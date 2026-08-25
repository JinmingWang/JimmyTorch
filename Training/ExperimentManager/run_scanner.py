"""Scan the ``Runs/`` directory tree and build a Dataset → Model → Run map.

Each run's metadata is read from its ``status_and_log.sqlite`` when present;
legacy runs (only ``comments.txt`` / ``model_arch.txt`` / no SQLite) still show
up as read-only nodes with best-effort metadata.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import storage
from .constants import RUN_DB_FILENAME


def _safe_read_meta(db_path: Path) -> dict[str, Any]:
    try:
        conn = storage.open_db(str(db_path))
        try:
            meta = storage.all_meta(conn)
        finally:
            conn.close()
        return meta
    except sqlite3.DatabaseError:
        return {}


def _legacy_meta(run_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    comments = run_dir / "comments.txt"
    if comments.is_file():
        try:
            meta["comments"] = comments.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    arch = run_dir / "model_arch.txt"
    if arch.is_file():
        try:
            meta["model_arch"] = arch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return meta


def scan_run(dataset_dir: str, model_dir: str, run_name: str, runs_root: Path) -> Optional[dict]:
    """Build a run node describing one leaf directory."""
    run_dir = runs_root / dataset_dir / model_dir / run_name
    if not run_dir.is_dir():
        return None
    db_path = run_dir / RUN_DB_FILENAME
    if db_path.is_file():
        meta = _safe_read_meta(db_path)
        has_db = True
    else:
        meta = _legacy_meta(run_dir)
        has_db = False
    status = meta.get("status") or ("done" if (run_dir / "best.pth").is_file() else "unknown")
    try:
        starred = bool(int(meta.get("starred", "0")))
    except (TypeError, ValueError):
        starred = False
    color = meta.get("color")
    return {
        "dataset": dataset_dir,
        "model": model_dir,
        "run_name": run_name,
        "path": f"{dataset_dir}/{model_dir}/{run_name}",
        "run_dir": str(run_dir),
        "has_db": has_db,
        "has_best": (run_dir / "best.pth").is_file(),
        "has_last": (run_dir / "last.pth").is_file(),
        "has_arch": (run_dir / "model_arch.txt").is_file() or bool(meta.get("model_arch")),
        "status": status,
        "starred": starred,
        "color": color,
        "created_at": _as_float(meta.get("created_at")),
        "updated_at": _as_float(meta.get("updated_at")),
        "closed_at": _as_float(meta.get("closed_at")),
    }


def scan_tree(runs_root: Path) -> dict[str, Any]:
    """Return a nested {dataset: {model: {run: node}}} mapping."""
    tree: dict[str, dict[str, dict[str, dict]]] = {}
    if not runs_root.is_dir():
        return {"runs_root": str(runs_root), "datasets": tree}
    for dataset_dir in sorted(_child_dirs(runs_root)):
        dataset_path = runs_root / dataset_dir
        models: dict[str, dict[str, dict]] = {}
        for model_dir in sorted(_child_dirs(dataset_path)):
            model_path = dataset_path / model_dir
            runs: dict[str, dict] = {}
            for run_name in sorted(_child_dirs(model_path)):
                node = scan_run(dataset_dir, model_dir, run_name, runs_root)
                if node is not None:
                    runs[run_name] = node
            if runs:
                models[model_dir] = runs
        if models:
            tree[dataset_dir] = models
    return {"runs_root": str(runs_root), "datasets": tree}


def _child_dirs(p: Path) -> list[str]:
    try:
        # Skip dotfiles and the shared global-settings sqlite.
        return [c.name for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except OSError:
        return []


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_hparams(meta: dict[str, Any]) -> Any:
    """Return the hparams as parsed JSON if possible, else the raw string."""
    raw = meta.get("hparams")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def sanitize_segment(s: str) -> str:
    """Reject empty and traversal segments; keep only plain path components."""
    if not s or s in ("..", ".") or "/" in s or "\\" in s or "\0" in s:
        raise ValueError(f"Invalid path segment: {s!r}")
    return s


def resolve_run_dir(runs_root: Path, dataset: str, model: str, run_name: str) -> Path:
    """Return the absolute run directory, guaranteed to be under ``runs_root``."""
    dataset = sanitize_segment(dataset)
    model = sanitize_segment(model)
    run_name = sanitize_segment(run_name)
    p = (runs_root / dataset / model / run_name).resolve()
    root_resolved = runs_root.resolve()
    try:
        p.relative_to(root_resolved)
    except ValueError as e:
        raise ValueError("Path escapes runs_root.") from e
    return p
