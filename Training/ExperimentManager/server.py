"""aiohttp-based ExperimentManager server.

Serves the React SPA + REST + WebSocket. Accepts training-side ingest over
HTTP JSON (``/api/ingest/*``), broadcasts live updates over ``/ws``, exposes
the ``Runs/`` tree, per-run scalars/figures/meta, comments/star/color/delete,
and global GUI settings.

Binds to 127.0.0.1 only; auto-increments port when the default is occupied.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from aiohttp import WSMsgType, web

from . import storage
from .constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, RUN_DB_FILENAME
from .global_settings import GlobalSettings
from .run_scanner import (
    parse_hparams,
    resolve_run_dir,
    scan_run,
    scan_tree,
)
from .watcher import TreeWatcher

log = logging.getLogger("experiment_manager.server")

STATIC_ROOT = Path(__file__).resolve().parent / "frontend" / "dist"
DEFAULT_RUNS_ROOT = Path.cwd() / "Runs"

# ---------- in-memory live state ----------

class LiveState:
    """Latest snapshot of the currently registered training run.

    Only one run is considered "live" at a time (the last one that registered).
    Historical runs are read from their SQLite files by later-stage endpoints.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.registration: Optional[dict] = None
        self.snapshot: dict = self._empty_snapshot()
        self.pending_learning_rate: Optional[float] = None

    @staticmethod
    def _empty_snapshot() -> dict:
        return {
            "status": "idle",
            "connected": False,
            "run": None,
            "progress": {
                "overall": 0,
                "total": 0,
                "percent": 0.0,
                "epoch": 0,
                "epochs": 0,
                "step": 0,
                "steps_per_epoch": 0,
                "elapsed": 0.0,
                "rate": 0.0,
                "remaining": None,
            },
            "metrics": {},
            "custom_fields": [],
            "learning_rate": {"applied": None, "pending": None},
            "system": {},
            "server_time": time.time(),
        }

    async def register(self, payload: dict) -> None:
        async with self._lock:
            self.registration = payload
            self.snapshot = self._empty_snapshot()
            self.snapshot["connected"] = True
            self.snapshot["status"] = "idle"
            self.snapshot["run"] = {
                "dataset": payload.get("dataset"),
                "model": payload.get("model"),
                "run_name": payload.get("run_name"),
                "run_dir": payload.get("run_dir"),
            }
            self.snapshot["progress"]["epochs"] = int(payload.get("epochs", 0) or 0)
            self.snapshot["progress"]["steps_per_epoch"] = int(payload.get("steps_per_epoch", 0) or 0)
            self.snapshot["progress"]["total"] = int(payload.get("total_steps", 0) or 0)
            self.snapshot["custom_fields"] = list(payload.get("custom_fields") or [])
            self.pending_learning_rate = None
            self.snapshot["learning_rate"]["pending"] = None

    async def apply_update(self, payload: dict) -> dict:
        async with self._lock:
            self.snapshot["connected"] = True
            self.snapshot["status"] = payload.get("status", self.snapshot["status"])
            if "progress" in payload:
                self.snapshot["progress"].update(payload["progress"])
            if "metrics" in payload:
                self.snapshot["metrics"] = payload["metrics"]
            if "learning_rate" in payload:
                lr = payload["learning_rate"]
                self.snapshot["learning_rate"]["applied"] = lr.get("applied")
            self.snapshot["learning_rate"]["pending"] = self.pending_learning_rate
            if "system" in payload:
                self.snapshot["system"] = payload["system"]
            self.snapshot["server_time"] = time.time()
            return dict(self.snapshot)

    async def apply_status(self, status: str) -> dict:
        async with self._lock:
            self.snapshot["status"] = status
            self.snapshot["server_time"] = time.time()
            return dict(self.snapshot)

    async def apply_finish(self, status: str = "done") -> dict:
        async with self._lock:
            self.snapshot["status"] = status
            self.snapshot["connected"] = False
            self.snapshot["server_time"] = time.time()
            return dict(self.snapshot)

    async def set_pending_lr(self, lr: float) -> None:
        async with self._lock:
            self.pending_learning_rate = float(lr)
            self.snapshot["learning_rate"]["pending"] = float(lr)

    async def consume_pending_lr(self) -> Optional[float]:
        async with self._lock:
            lr = self.pending_learning_rate
            self.pending_learning_rate = None
            self.snapshot["learning_rate"]["pending"] = None
            return lr

    async def get_snapshot(self) -> dict:
        async with self._lock:
            return dict(self.snapshot)


# ---------- websocket broadcast ----------

class Broadcaster:
    def __init__(self) -> None:
        self._clients: set[web.WebSocketResponse] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message, allow_nan=False, default=str)
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                await self.unregister(ws)


# ---------- HTTP handlers ----------

async def handle_ingest_register(request: web.Request) -> web.Response:
    payload = await request.json()
    state: LiveState = request.app["live_state"]
    broadcaster: Broadcaster = request.app["broadcaster"]
    await state.register(payload)
    snap = await state.get_snapshot()
    await broadcaster.broadcast({"type": "live_snapshot", "payload": snap})
    return web.json_response({"ok": True})


async def handle_ingest_update(request: web.Request) -> web.Response:
    payload = await request.json()
    state: LiveState = request.app["live_state"]
    broadcaster: Broadcaster = request.app["broadcaster"]
    snap = await state.apply_update(payload)
    await broadcaster.broadcast({"type": "live_snapshot", "payload": snap})
    return web.json_response({"ok": True, "pending_lr": snap["learning_rate"]["pending"]})


async def handle_ingest_status(request: web.Request) -> web.Response:
    payload = await request.json()
    status = str(payload.get("status", "idle"))
    state: LiveState = request.app["live_state"]
    broadcaster: Broadcaster = request.app["broadcaster"]
    snap = await state.apply_status(status)
    await broadcaster.broadcast({"type": "live_snapshot", "payload": snap})
    return web.json_response({"ok": True})


async def handle_ingest_finish(request: web.Request) -> web.Response:
    payload = await request.json()
    status = str(payload.get("status", "done"))
    state: LiveState = request.app["live_state"]
    broadcaster: Broadcaster = request.app["broadcaster"]
    snap = await state.apply_finish(status)
    await broadcaster.broadcast({"type": "live_snapshot", "payload": snap})
    return web.json_response({"ok": True})


async def handle_get_pending_lr(request: web.Request) -> web.Response:
    """Trainer polls this to pick up browser-requested LR (consuming the value)."""
    state: LiveState = request.app["live_state"]
    lr = await state.consume_pending_lr()
    return web.json_response({"pending": lr})


async def handle_post_lr_request(request: web.Request) -> web.Response:
    """Browser posts a new learning-rate request."""
    payload = await request.json()
    try:
        lr = float(payload.get("learning_rate"))
    except (TypeError, ValueError):
        return web.json_response({"error": "learning_rate must be a number."}, status=400)
    if lr <= 0 or not (lr == lr):  # rejects NaN, non-positive
        return web.json_response({"error": "learning_rate must be positive."}, status=400)
    state: LiveState = request.app["live_state"]
    broadcaster: Broadcaster = request.app["broadcaster"]
    await state.set_pending_lr(lr)
    snap = await state.get_snapshot()
    await broadcaster.broadcast({"type": "live_snapshot", "payload": snap})
    return web.json_response({"ok": True, "pending": lr}, status=202)


async def handle_status(request: web.Request) -> web.Response:
    """Full snapshot for initial browser load."""
    state: LiveState = request.app["live_state"]
    snap = await state.get_snapshot()
    return web.json_response(snap)


# ---------- runs tree / summary / scalars / figures ----------

def _runs_root(request: web.Request) -> Path:
    return request.app["runs_root"]


def _resolve_run_dir_from_request(request: web.Request) -> Path:
    return resolve_run_dir(
        _runs_root(request),
        request.match_info["dataset"],
        request.match_info["model"],
        request.match_info["run"],
    )


def _open_run_db(run_dir: Path) -> Optional[Any]:
    db_path = run_dir / RUN_DB_FILENAME
    if not db_path.is_file():
        return None
    return storage.open_db(str(db_path))


async def handle_tree(request: web.Request) -> web.Response:
    watcher: TreeWatcher = request.app["watcher"]
    tree = watcher.snapshot()
    if not tree.get("datasets"):
        tree = await asyncio.get_event_loop().run_in_executor(None, scan_tree, _runs_root(request))
    return web.json_response(tree)


async def handle_run_summary(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    node = scan_run(
        request.match_info["dataset"],
        request.match_info["model"],
        request.match_info["run"],
        _runs_root(request),
    )
    conn = _open_run_db(run_dir)
    meta: dict[str, Any] = {}
    scalar_tags: list[str] = []
    figure_tags: list[str] = []
    if conn is not None:
        try:
            meta = storage.all_meta(conn)
            scalar_tags = storage.list_scalar_tags(conn)
            figure_tags = storage.list_figure_tags(conn)
        finally:
            conn.close()
    return web.json_response({
        "node": node,
        "meta": meta,
        "hparams": parse_hparams(meta),
        "scalar_tags": scalar_tags,
        "figure_tags": figure_tags,
    })


async def handle_run_scalars(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    tag = request.query.get("tag")
    if not tag:
        return web.json_response({"error": "missing tag"}, status=400)
    max_points = _int_or_none(request.query.get("max_points"))
    step_min = _int_or_none(request.query.get("step_min"))
    step_max = _int_or_none(request.query.get("step_max"))
    conn = _open_run_db(run_dir)
    if conn is None:
        return web.json_response({"tag": tag, "points": []})
    try:
        rows = storage.read_scalars(
            conn, tag, step_min=step_min, step_max=step_max, max_points=max_points
        )
    finally:
        conn.close()
    return web.json_response({
        "tag": tag,
        "points": [{"step": s, "wall_time": w, "value": v} for (s, w, v) in rows],
    })


async def handle_run_figures_index(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    tag = request.query.get("tag")
    if not tag:
        return web.json_response({"error": "missing tag"}, status=400)
    conn = _open_run_db(run_dir)
    if conn is None:
        return web.json_response({"tag": tag, "entries": []})
    try:
        entries = storage.list_figure_index(conn, tag)
    finally:
        conn.close()
    return web.json_response({
        "tag": tag,
        "entries": [{"step": s, "wall_time": w, "mime": m} for (s, w, m) in entries],
    })


async def handle_run_figure_blob(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.Response(status=400, text="invalid path")
    tag = request.query.get("tag")
    step = _int_or_none(request.query.get("step"))
    if not tag or step is None:
        return web.Response(status=400, text="missing tag or step")
    conn = _open_run_db(run_dir)
    if conn is None:
        return web.Response(status=404)
    try:
        result = storage.read_figure_blob(conn, tag, step)
    finally:
        conn.close()
    if result is None:
        return web.Response(status=404)
    blob, mime = result
    return web.Response(body=blob, content_type=mime)


async def handle_run_arch_txt(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.Response(status=400, text="invalid path")
    conn = _open_run_db(run_dir)
    arch: Optional[str] = None
    if conn is not None:
        try:
            arch = storage.get_meta(conn, "model_arch")
        finally:
            conn.close()
    if not arch:
        legacy = run_dir / "model_arch.txt"
        if legacy.is_file():
            arch = legacy.read_text(encoding="utf-8", errors="replace")
    if not arch:
        return web.Response(status=404, text="model_arch.txt not found")
    return web.Response(text=arch, content_type="text/plain")


# ---------- run mutations: comments / star / color / delete / open folder ----------

async def _write_run_meta(run_dir: Path, key: str, value: str) -> None:
    db_path = run_dir / RUN_DB_FILENAME
    conn = storage.open_db(str(db_path))
    try:
        storage.init_run_schema(conn)
        storage.set_meta(conn, key, value)
    finally:
        conn.close()


async def handle_post_comments(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    payload = await request.json()
    text = str(payload.get("comments", ""))
    (run_dir / "comments.txt").write_text(text, encoding="utf-8")
    await _write_run_meta(run_dir, "comments", text)
    _broadcast_tree_change(request)
    return web.json_response({"ok": True})


async def handle_post_star(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    payload = await request.json()
    starred = bool(payload.get("starred", False))
    await _write_run_meta(run_dir, "starred", "1" if starred else "0")
    _broadcast_tree_change(request)
    return web.json_response({"ok": True, "starred": starred})


async def handle_post_color(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    payload = await request.json()
    color = payload.get("color")
    if color is not None and not isinstance(color, str):
        return web.json_response({"error": "color must be string or null"}, status=400)
    await _write_run_meta(run_dir, "color", color or "")
    _broadcast_tree_change(request)
    return web.json_response({"ok": True, "color": color})


async def handle_delete_run(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    payload = await request.json()
    token = str(payload.get("confirm_token", ""))
    expected = _delete_token(request.match_info)
    if token != expected:
        return web.json_response(
            {"error": "confirmation token mismatch", "expected_token": expected}, status=403
        )
    shutil.rmtree(run_dir)
    _broadcast_tree_change(request)
    return web.json_response({"ok": True})


async def handle_get_delete_token(request: web.Request) -> web.Response:
    return web.json_response({"confirm_token": _delete_token(request.match_info)})


async def handle_open_folder(request: web.Request) -> web.Response:
    try:
        run_dir = _resolve_run_dir_from_request(request)
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)
    if not run_dir.is_dir():
        return web.json_response({"error": "not found"}, status=404)
    opener = _pick_folder_opener()
    if opener is None:
        return web.json_response({"error": "no opener available"}, status=501)
    try:
        subprocess.Popen(  # noqa: S603 - trusted arg vector
            [opener, str(run_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"ok": True})


def _pick_folder_opener() -> Optional[str]:
    if sys.platform == "darwin":
        return "open"
    if sys.platform.startswith("win"):
        return "explorer"
    for candidate in ("xdg-open", "gio"):
        if shutil.which(candidate):
            return candidate
    return None


def _delete_token(match_info) -> str:
    raw = f"{match_info['dataset']}/{match_info['model']}/{match_info['run']}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _broadcast_tree_change(request: web.Request) -> None:
    watcher: TreeWatcher = request.app["watcher"]
    # Kick the watcher on the next tick; nothing to await here.
    asyncio.get_event_loop().call_soon(lambda: None)
    request.app["broadcaster"]  # keep reference; watcher will diff on next scan
    del watcher  # silence linters


# ---------- global settings ----------

async def handle_get_global_settings(request: web.Request) -> web.Response:
    settings: GlobalSettings = request.app["global_settings"]
    return web.json_response(settings.all())


async def handle_post_global_settings(request: web.Request) -> web.Response:
    settings: GlobalSettings = request.app["global_settings"]
    payload = await request.json()
    if not isinstance(payload, dict):
        return web.json_response({"error": "expected object"}, status=400)
    settings.merge(payload)
    return web.json_response({"ok": True})


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------- websocket ----------

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=15.0)
    await ws.prepare(request)
    broadcaster: Broadcaster = request.app["broadcaster"]
    state: LiveState = request.app["live_state"]
    watcher: TreeWatcher = request.app["watcher"]
    await broadcaster.register(ws)
    try:
        snap = await state.get_snapshot()
        await ws.send_str(json.dumps({"type": "live_snapshot", "payload": snap}, default=str))
        tree = watcher.snapshot()
        await ws.send_str(json.dumps({"type": "tree_updated", "payload": tree}, default=str))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Client → server messages (subscribe_run, etc.) reserved for future use.
                pass
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws error: %s", ws.exception())
                break
    finally:
        await broadcaster.unregister(ws)
    return ws


# ---------- static fallback ----------

_FALLBACK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ExperimentManager</title>
<style>
body{background:#0f1720;color:#d5e2ee;font-family:system-ui,sans-serif;padding:2rem;max-width:720px;margin:auto}
code{background:#1a2431;padding:0.15rem 0.4rem;border-radius:0.3rem;color:#55ded0}
h1{color:#55ded0}
</style></head><body>
<h1>ExperimentManager server is running.</h1>
<p>The React SPA has not been built yet. Run:</p>
<pre><code>cd Training/ExperimentManager/frontend
npm install
npm run build</code></pre>
<p>Then reload this page. Live JSON snapshot: <a href="/api/status">/api/status</a></p>
</body></html>
"""


async def handle_root(request: web.Request) -> web.Response:
    index = STATIC_ROOT / "index.html"
    if index.is_file():
        return web.FileResponse(index)
    return web.Response(text=_FALLBACK_HTML, content_type="text/html")


# ---------- app factory ----------

def build_app(runs_root: Optional[Path] = None) -> web.Application:
    app = web.Application()
    app["runs_root"] = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    app["runs_root"].mkdir(parents=True, exist_ok=True)
    app["live_state"] = LiveState()
    app["broadcaster"] = Broadcaster()
    app["global_settings"] = GlobalSettings(app["runs_root"])
    app["watcher"] = TreeWatcher(app["runs_root"], app["broadcaster"])

    async def _on_startup(_app: web.Application) -> None:
        await _app["watcher"].start()

    async def _on_cleanup(_app: web.Application) -> None:
        await _app["watcher"].stop()
        _app["global_settings"].close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # live status + WS
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/ws", handle_ws)

    # ingest (trainer -> server)
    app.router.add_post("/api/ingest/register", handle_ingest_register)
    app.router.add_post("/api/ingest/update", handle_ingest_update)
    app.router.add_post("/api/ingest/status", handle_ingest_status)
    app.router.add_post("/api/ingest/finish", handle_ingest_finish)

    # learning rate
    app.router.add_get("/api/live/learning-rate", handle_get_pending_lr)
    app.router.add_post("/api/learning-rate", handle_post_lr_request)

    # tree + per-run
    app.router.add_get("/api/tree", handle_tree)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/summary", handle_run_summary)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/scalars", handle_run_scalars)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/figures", handle_run_figures_index)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/figure_blob", handle_run_figure_blob)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/arch_txt", handle_run_arch_txt)
    app.router.add_get("/api/runs/{dataset}/{model}/{run}/delete_token", handle_get_delete_token)
    app.router.add_post("/api/runs/{dataset}/{model}/{run}/comments", handle_post_comments)
    app.router.add_post("/api/runs/{dataset}/{model}/{run}/star", handle_post_star)
    app.router.add_post("/api/runs/{dataset}/{model}/{run}/color", handle_post_color)
    app.router.add_post("/api/runs/{dataset}/{model}/{run}/open_folder", handle_open_folder)
    app.router.add_delete("/api/runs/{dataset}/{model}/{run}", handle_delete_run)

    # global settings
    app.router.add_get("/api/global_settings", handle_get_global_settings)
    app.router.add_post("/api/global_settings", handle_post_global_settings)

    # static + SPA
    app.router.add_get("/", handle_root)
    if STATIC_ROOT.is_dir():
        app.router.add_static("/assets", STATIC_ROOT / "assets")
        # Fallback: any unknown non-API top-level path returns index.html for SPA routing.
        async def spa_catch_all(request: web.Request) -> web.Response:
            if request.match_info["tail"].startswith("api/") or request.match_info["tail"] == "ws":
                return web.Response(status=404)
            index = STATIC_ROOT / "index.html"
            if index.is_file():
                return web.FileResponse(index)
            return web.Response(status=404)
        app.router.add_get("/{tail:.*}", spa_catch_all)
    return app


def pick_free_port(host: str, start_port: int, max_tries: int = 200) -> int:
    """Return the first bindable port at or above ``start_port``."""
    import socket

    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in [{start_port}, {start_port + max_tries}).")


def run_server(
    host: str = DEFAULT_SERVER_HOST,
    start_port: int = DEFAULT_SERVER_PORT,
    runs_root: Optional[Path] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = pick_free_port(host, start_port)
    app = build_app(runs_root=runs_root)
    log.info(
        "ExperimentManager listening on http://%s:%d (runs_root=%s)",
        host, port, app["runs_root"],
    )
    web.run_app(app, host=host, port=port, print=None, access_log=None)
