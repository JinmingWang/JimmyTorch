"""Training-side ExperimentManager client.

Drop-in replacement for the old ``ProgressManagerGUI``:
- Same public methods (``update``, ``consume_learning_rate_request``,
  ``mark_learning_rate_applied``, ``close``, ``snapshot``, ``url``, ``port``).
- Same public attributes used by the trainer (``overall_progress``,
  ``current_epoch``, ``current_step``, ``custom_fields``, ``epochs``,
  ``steps_per_epoch``, ``total_steps``).

Architecture: never hosts an HTTP server. A single background daemon thread
periodically POSTs the current snapshot to the ExperimentManager server and
polls it for browser-requested LR changes. All calls degrade silently if the
server is not running — training never blocks and never crashes because of
network issues. The authoritative log is written to SQLite by the paired
:class:`ExperimentLogger` (owned by ``JimmyExperiment``).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, List, Optional

from .constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT

# Optional system-stat helpers. Silently degrade when unavailable.
try:  # psutil is confirmed present in Py311.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

_pynvml: Any = None
_pynvml_handles: list = []
try:
    import pynvml as _pynvml_mod  # type: ignore
    _pynvml_mod.nvmlInit()
    _pynvml = _pynvml_mod
    _pynvml_handles = [
        _pynvml.nvmlDeviceGetHandleByIndex(i)
        for i in range(_pynvml.nvmlDeviceGetCount())
    ]
except Exception:
    _pynvml = None
    _pynvml_handles = []


def _sample_system() -> dict:
    """Return best-effort system stats. Missing fields are omitted."""
    out: dict[str, Any] = {}
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            out["cpu_mem_bytes"] = int(vm.used)
            out["cpu_mem_total"] = int(vm.total)
            out["cpu_util"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
    if _pynvml is not None and _pynvml_handles:
        try:
            h = _pynvml_handles[0]
            mem = _pynvml.nvmlDeviceGetMemoryInfo(h)
            util = _pynvml.nvmlDeviceGetUtilizationRates(h)
            out["gpu_mem_used"] = int(mem.used)
            out["gpu_mem_total"] = int(mem.total)
            out["gpu_util"] = int(util.gpu)
        except Exception:
            pass
    return out


class ExperimentManagerClient:
    """Drop-in replacement for ProgressManagerGUI, decoupled from the server."""

    def __init__(
        self,
        items_per_epoch: int,
        epochs: int,
        *,
        refresh_interval: float = 1.0,
        custom_fields: Optional[List[str]] = None,
        show_recent_steps: int = 200,
        host: str = DEFAULT_SERVER_HOST,
        port: int = DEFAULT_SERVER_PORT,
        dataset_name: str = "",
        model_name: str = "",
        run_name: str = "",
        run_dir: str = "",
    ) -> None:
        if items_per_epoch <= 0:
            raise ValueError("items_per_epoch must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive.")

        self.epochs = epochs
        self.steps_per_epoch = items_per_epoch
        self.total_steps = epochs * items_per_epoch
        self.show_recent_steps = show_recent_steps
        self.refresh_interval = refresh_interval
        self.custom_fields = [] if custom_fields is None else list(custom_fields)

        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.run_name = run_name
        self.run_dir = os.path.abspath(run_dir) if run_dir else ""

        self.overall_progress = 0
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()

        self._status = "idle"
        self._latest_values: dict[str, Optional[float]] = {}
        self._applied_lr: Optional[float] = None
        self._pending_lr: Optional[float] = None
        self._server_reachable = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reporter_thread: Optional[threading.Thread] = None
        self._closed = False

        self._register_and_start()

    # ---------- public property parity ----------

    @property
    def url(self) -> str:
        return self._base_url

    # ---------- public API used by JimmyTrainer ----------

    def update(self, current_epoch: int, current_step: int, **custom_values: float) -> None:
        """Record one completed batch (zero-indexed epoch and step)."""
        if not 0 <= current_epoch < self.epochs:
            raise IndexError("current_epoch is outside the configured range.")
        if not 0 <= current_step < self.steps_per_epoch:
            raise IndexError("current_step is outside the configured range.")

        values = {
            field: self._as_scalar(custom_values.get(field))
            for field in self.custom_fields
        }
        with self._lock:
            self.overall_progress += 1
            self.current_epoch = current_epoch
            self.current_step = current_step
            self._latest_values = values
            self._status = "training"

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def mark_learning_rate_applied(self, learning_rate: float) -> None:
        with self._lock:
            self._applied_lr = float(learning_rate)

    def consume_learning_rate_request(self) -> Optional[float]:
        """Return and clear the latest LR request (populated by the poller)."""
        with self._lock:
            lr = self._pending_lr
            self._pending_lr = None
            return lr

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0.0, time.time() - self.start_time)
            rate = self.overall_progress / elapsed if elapsed > 0 else 0.0
            remaining = ((self.total_steps - self.overall_progress) / rate) if rate > 0 else None
            return {
                "status": self._status,
                "connected": self._server_reachable,
                "run": {
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "run_name": self.run_name,
                    "run_dir": self.run_dir,
                },
                "progress": {
                    "overall": self.overall_progress,
                    "total": self.total_steps,
                    "percent": (self.overall_progress / self.total_steps * 100) if self.total_steps else 0.0,
                    "epoch": self.current_epoch + 1,
                    "epochs": self.epochs,
                    "step": self.current_step + 1,
                    "steps_per_epoch": self.steps_per_epoch,
                    "elapsed": elapsed,
                    "rate": rate,
                    "remaining": remaining,
                },
                "metrics": dict(self._latest_values),
                "custom_fields": list(self.custom_fields),
                "learning_rate": {
                    "applied": self._applied_lr,
                    "pending": self._pending_lr,
                },
                "system": _sample_system(),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        # One last best-effort finish ping.
        self._post_silently("/api/ingest/finish", {
            "run_dir": self.run_dir,
            "status": self._status if self._status in ("error", "done") else "done",
        })
        if self._reporter_thread is not None:
            self._reporter_thread.join(timeout=2.0)

    # ---------- internals ----------

    def _register_and_start(self) -> None:
        payload = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "custom_fields": list(self.custom_fields),
            "start_time": self.start_time,
        }
        self._post_silently("/api/ingest/register", payload)
        self._reporter_thread = threading.Thread(
            target=self._reporter_loop, name="ExpManagerClientReporter", daemon=True
        )
        self._reporter_thread.start()
        print(f"ExperimentManager client: reporting to {self._base_url}")

    def _reporter_loop(self) -> None:
        while not self._stop_event.is_set():
            snap = self.snapshot()
            update_payload = {
                "run_dir": self.run_dir,
                "status": snap["status"],
                "progress": snap["progress"],
                "metrics": snap["metrics"],
                "learning_rate": snap["learning_rate"],
                "system": snap["system"],
            }
            resp = self._post_silently("/api/ingest/update", update_payload)
            pending_lr = None
            if isinstance(resp, dict):
                pending_lr = resp.get("pending_lr")
            # Also fetch explicitly in case an update failed but LR arrived.
            fetched = self._get_silently("/api/live/learning-rate")
            if isinstance(fetched, dict) and fetched.get("pending") is not None:
                pending_lr = fetched["pending"]
            if pending_lr is not None:
                with self._lock:
                    self._pending_lr = float(pending_lr)
            self._stop_event.wait(self.refresh_interval)

    def _post_silently(self, path: str, payload: dict) -> Optional[dict]:
        try:
            body = json.dumps(payload, default=str).encode("utf-8")
            req = urllib.request.Request(
                self._base_url + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as r:
                self._server_reachable = True
                raw = r.read()
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return None
        except (urllib.error.URLError, OSError, TimeoutError):
            self._server_reachable = False
            return None

    def _get_silently(self, path: str) -> Optional[dict]:
        try:
            with urllib.request.urlopen(self._base_url + path, timeout=1.0) as r:
                self._server_reachable = True
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
            self._server_reachable = False
            return None

    @staticmethod
    def _as_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None
