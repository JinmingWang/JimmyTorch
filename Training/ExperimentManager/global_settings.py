"""Global GUI settings persisted to ``Runs/Experiment_GUI_Status.sqlite``.

Stores JSON-encoded key/value pairs for cross-run UI state: theme, per-curve
xlim/ylim/smoothing/collapsed, node checked states, group colors, etc.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from . import storage
from .constants import GLOBAL_DB_FILENAME


class GlobalSettings:
    """Single-DB key-value store, thread-safe via a coarse lock."""

    def __init__(self, runs_root: Path) -> None:
        self._path = runs_root / GLOBAL_DB_FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _ensure(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = storage.open_db(str(self._path))
            storage.init_global_schema(self._conn)
        return self._conn

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            conn = self._ensure()
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            conn = self._ensure()
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, default=str)),
            )
            conn.commit()

    def all(self) -> dict[str, Any]:
        with self._lock:
            conn = self._ensure()
            out: dict[str, Any] = {}
            for row in conn.execute("SELECT key, value FROM settings"):
                try:
                    out[row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    out[row["key"]] = row["value"]
            return out

    def merge(self, patch: dict[str, Any]) -> None:
        for k, v in patch.items():
            self.set(k, v)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
