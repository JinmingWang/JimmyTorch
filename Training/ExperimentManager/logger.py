"""ExperimentLogger — the training-side writer for a single run's SQLite log."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from . import storage
from .constants import (
    RUN_DB_FILENAME,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_IDLE,
)


class ExperimentLogger:
    """Writes scalars, figures, and metadata to ``<run_dir>/status_and_log.sqlite``.

    All writes are local-first; the ExperimentManager server (Stage 2) reads
    this file and streams to the browser. Safe to use even when no server is running.
    """

    def __init__(
        self,
        run_dir: str,
        dataset_name: str,
        model_name: str,
        run_name: str,
    ) -> None:
        self.run_dir = run_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.run_name = run_name

        os.makedirs(run_dir, exist_ok=True)
        self._db_path = os.path.join(run_dir, RUN_DB_FILENAME)
        self._conn = storage.open_db(self._db_path)
        storage.init_run_schema(self._conn)

        storage.set_meta(self._conn, "dataset_name", dataset_name)
        storage.set_meta(self._conn, "model_name", model_name)
        storage.set_meta(self._conn, "run_name", run_name)
        if storage.get_meta(self._conn, "created_at") is None:
            storage.set_meta(self._conn, "created_at", str(time.time()))
        storage.set_meta(self._conn, "updated_at", str(time.time()))
        storage.set_meta(self._conn, "status", STATUS_IDLE)

    # ---------- scalars ----------

    def log_scalar(self, tag: str, step: int, value: float) -> None:
        v = self._coerce(value)
        if v is None:
            return
        storage.record_scalar(self._conn, tag, int(step), v)
        self._touch()

    def log_scalars(self, step: int, **values: float) -> None:
        if not values:
            return
        cleaned: dict[str, float] = {}
        for k, v in values.items():
            coerced = self._coerce(v)
            if coerced is not None:
                cleaned[k] = coerced
        if not cleaned:
            return
        storage.record_scalars_batch(self._conn, int(step), cleaned)
        self._touch()

    @staticmethod
    def _coerce(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        import math as _m
        return v if _m.isfinite(v) else None

    # ---------- figures ----------

    def log_figure(self, tag: str, step: int, fig: Any) -> None:
        """Encode a matplotlib ``Figure`` as PNG bytes and store it."""
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        blob = buf.getvalue()
        storage.record_figure(self._conn, tag, int(step), blob, mime="image/png")
        self._touch()

    # ---------- metadata ----------

    def log_hparams(self, hparams: dict) -> None:
        storage.set_meta(self._conn, "hparams", json.dumps(hparams, default=str))
        self._touch()

    def set_comments(self, comments: str) -> None:
        storage.set_meta(self._conn, "comments", comments)
        self._touch()

    def set_model_arch(self, arch_text: str) -> None:
        storage.set_meta(self._conn, "model_arch", arch_text)
        self._touch()

    def set_status(self, status: str) -> None:
        storage.set_meta(self._conn, "status", status)
        self._touch()

    def set_meta(self, key: str, value: str) -> None:
        storage.set_meta(self._conn, key, value)
        self._touch()

    def get_meta(self, key: str) -> Optional[str]:
        return storage.get_meta(self._conn, key)

    # ---------- lifecycle ----------

    def close(self, final_status: str = STATUS_DONE) -> None:
        try:
            storage.set_meta(self._conn, "status", final_status)
            storage.set_meta(self._conn, "closed_at", str(time.time()))
        finally:
            self._conn.close()

    def close_with_error(self, message: str = "") -> None:
        try:
            storage.set_meta(self._conn, "status", STATUS_ERROR)
            if message:
                storage.set_meta(self._conn, "error_message", message)
            storage.set_meta(self._conn, "closed_at", str(time.time()))
        finally:
            self._conn.close()

    # ---------- internals ----------

    def _touch(self) -> None:
        storage.set_meta(self._conn, "updated_at", str(time.time()))

    @property
    def db_path(self) -> str:
        return self._db_path
