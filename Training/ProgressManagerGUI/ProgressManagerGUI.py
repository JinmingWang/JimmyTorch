import json
import math
import mimetypes
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List


class ProgressManagerGUI:
    """A lightweight browser monitor for one active training process.

    The training thread calls :meth:`update` once per batch. The embedded HTTP
    server only reads compact, bounded snapshots, so it does not perform metric
    aggregation or write to disk while training is running.
    """

    def __init__(self,
                 items_per_epoch: int,
                 epochs: int,
                 refresh_interval: float = 1.0,
                 custom_fields: List[str] | None = None,
                 show_recent_steps: int = 200,
                 host: str = "127.0.0.1",
                 start_port: int = 9000,
                 auto_start: bool = True):
        if items_per_epoch <= 0:
            raise ValueError("items_per_epoch must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if show_recent_steps <= 0:
            raise ValueError("show_recent_steps must be positive.")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive.")
        if not 0 < start_port < 65536:
            raise ValueError("start_port must be between 1 and 65535.")

        self.epochs = epochs
        self.steps_per_epoch = items_per_epoch
        self.total_steps = epochs * items_per_epoch
        self.show_recent_steps = show_recent_steps
        self.refresh_interval = refresh_interval
        self.custom_fields = [] if custom_fields is None else list(custom_fields)
        self.host = host
        self.start_port = start_port

        self.overall_progress = 0
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        self._events: deque[dict[str, Any]] = deque(maxlen=show_recent_steps)
        self._pending_learning_rate: float | None = None
        self._applied_learning_rate: float | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self.port: int | None = None

        if auto_start:
            self.start()

    @property
    def url(self) -> str | None:
        if self.port is None:
            return None
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the local dashboard server on the first available port."""
        with self._lock:
            if self._server is not None:
                return

        server = self._create_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        with self._lock:
            self._server = server
            self._server_thread = thread
            self.port = server.server_port

        print(f"Training monitor: {self.url}")

    def update(self, current_epoch: int, current_step: int, **custom_values: float) -> None:
        """Record one completed batch using zero-indexed epoch and step inputs."""
        if not 0 <= current_epoch < self.epochs:
            raise IndexError("current_epoch is outside the configured training range.")
        if not 0 <= current_step < self.steps_per_epoch:
            raise IndexError("current_step is outside the configured epoch range.")

        now = time.time()
        values = {
            field: self._as_scalar(custom_values.get(field))
            for field in self.custom_fields
        }
        with self._lock:
            self.overall_progress += 1
            self.current_epoch = current_epoch
            self.current_step = current_step
            self._events.append({
                "global_step": self.overall_progress,
                "epoch": current_epoch + 1,
                "step": current_step + 1,
                "elapsed": now - self.start_time,
                "values": values,
            })

    def request_learning_rate(self, learning_rate: Any) -> float:
        """Queue the latest browser learning-rate request for the training thread."""
        try:
            value = float(learning_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("Learning rate must be a positive finite number.") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rate must be a positive finite number.")

        with self._lock:
            self._pending_learning_rate = value
        return value

    def consume_learning_rate_request(self) -> float | None:
        """Return and clear the latest LR request. Call this from the training thread."""
        with self._lock:
            learning_rate = self._pending_learning_rate
            self._pending_learning_rate = None
            return learning_rate

    def mark_learning_rate_applied(self, learning_rate: float) -> None:
        """Record the LR successfully applied by the training thread."""
        with self._lock:
            self._applied_learning_rate = float(learning_rate)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe bounded view of the current training state."""
        now = time.time()
        with self._lock:
            elapsed = max(0.0, now - self.start_time)
            rate = self.overall_progress / elapsed if elapsed > 0 else 0.0
            remaining_steps = max(0, self.total_steps - self.overall_progress)
            return {
                "status": "closed" if self._closed else "running",
                "refresh_interval": self.refresh_interval,
                "custom_fields": list(self.custom_fields),
                "progress": {
                    "overall": self.overall_progress,
                    "total": self.total_steps,
                    "percent": self.overall_progress / self.total_steps * 100,
                    "epoch": self.current_epoch + 1,
                    "epochs": self.epochs,
                    "step": self.current_step + 1,
                    "steps_per_epoch": self.steps_per_epoch,
                    "elapsed": elapsed,
                    "rate": rate,
                    "remaining": remaining_steps / rate if rate > 0 else None,
                },
                "learning_rate": {
                    "applied": self._applied_learning_rate,
                    "pending": self._pending_learning_rate,
                },
                "recent_events": list(self._events),
            }

    def close(self) -> None:
        """Stop accepting dashboard requests without affecting the training process."""
        with self._lock:
            self._closed = True
            server = self._server
            thread = self._server_thread
            self._server = None
            self._server_thread = None

        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _create_server(self) -> ThreadingHTTPServer:
        handler = self._make_request_handler()
        port = self.start_port
        while True:
            try:
                return ThreadingHTTPServer((self.host, port), handler)
            except OSError:
                port += 1
                if port > 65535:
                    raise RuntimeError("No available TCP port remains at or above start_port.")

    def _make_request_handler(self):
        manager = self
        static_dir = Path(__file__).resolve().parent

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/status":
                    self._send_json(HTTPStatus.OK, manager.snapshot())
                    return
                if self.path in ("/", "/index.html", "/styles.css", "/app.js"):
                    asset_name = "index.html" if self.path == "/" else self.path.lstrip("/")
                    self._send_asset(static_dir / asset_name)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

            def do_POST(self):
                if self.path != "/api/learning-rate":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(content_length))
                    learning_rate = manager.request_learning_rate(payload.get("learning_rate"))
                except (AttributeError, ValueError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {
                        "error": "learning_rate must be a positive finite number.",
                    })
                    return
                self._send_json(HTTPStatus.ACCEPTED, {
                    "pending": learning_rate,
                    "message": "Learning rate will apply before the next training batch.",
                })

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send_asset(self, path: Path) -> None:
                if not path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Dashboard asset not found."})
                    return
                content = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                content = json.dumps(payload, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        return RequestHandler

    @staticmethod
    def _as_scalar(value: Any) -> float | None:
        if value is None:
            return None
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        return scalar if math.isfinite(scalar) else None