"""Low-level SQLite storage for per-run experiment logs and global GUI state.

Design notes:
- One SQLite file per run at ``<run_dir>/status_and_log.sqlite`` (WAL mode).
- One global file at ``Runs/Experiment_GUI_Status.sqlite`` for cross-run UI state.
- Scalars are bounded per-tag by reservoir sampling: newest ``recent_keep`` rows
  are never evicted; when a tag exceeds ``cap`` rows, one row is dropped uniformly
  at random from the older history. This mimics TensorBoard's space-saving policy
  while keeping the live tail exact for real-time curves.
- Figures are stored as raw blobs (PNG bytes) with a cap per tag; oldest are dropped.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Iterable, Optional, Sequence

from .constants import (
    DEFAULT_FIGURE_CAP,
    DEFAULT_SCALAR_CAP,
    DEFAULT_SCALAR_RECENT_KEEP,
)


# ---------- connection ----------

def open_db(path: str) -> sqlite3.Connection:
    """Open (creating parent dirs) a SQLite DB in WAL mode with sensible pragmas."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


# ---------- schema ----------

_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS scalars (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scalars_tag_step ON scalars(tag, step);

CREATE TABLE IF NOT EXISTS figures (
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    mime      TEXT    NOT NULL,
    blob      BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_figures_tag_step ON figures(tag, step);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS reservoir_state (
    tag  TEXT PRIMARY KEY,
    seen INTEGER NOT NULL,
    cap  INTEGER NOT NULL
);
"""


_GLOBAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_run_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_RUN_SCHEMA)
    conn.commit()


def init_global_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_GLOBAL_SCHEMA)
    conn.commit()


# ---------- meta ----------

def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row["value"]


def all_meta(conn: sqlite3.Connection) -> dict:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}


# ---------- scalars ----------

def _bump_seen(conn: sqlite3.Connection, tag: str, cap: int) -> int:
    conn.execute(
        "INSERT INTO reservoir_state(tag, seen, cap) VALUES(?, 1, ?) "
        "ON CONFLICT(tag) DO UPDATE SET seen = seen + 1",
        (tag, cap),
    )
    return conn.execute(
        "SELECT seen FROM reservoir_state WHERE tag=?", (tag,)
    ).fetchone()["seen"]


def record_scalar(
    conn: sqlite3.Connection,
    tag: str,
    step: int,
    value: float,
    wall_time: Optional[float] = None,
    cap: int = DEFAULT_SCALAR_CAP,
    recent_keep: int = DEFAULT_SCALAR_RECENT_KEEP,
) -> None:
    """Insert one scalar. Evicts one older row at random when the tag exceeds ``cap``."""
    if wall_time is None:
        wall_time = time.time()
    conn.execute(
        "INSERT INTO scalars(tag, step, wall_time, value) VALUES(?, ?, ?, ?)",
        (tag, step, wall_time, value),
    )
    _bump_seen(conn, tag, cap)
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM scalars WHERE tag=?", (tag,)
    ).fetchone()["c"]
    if count > cap:
        conn.execute(
            """
            DELETE FROM scalars
            WHERE rowid IN (
                SELECT rowid FROM scalars
                WHERE tag = ?
                  AND rowid NOT IN (
                    SELECT rowid FROM scalars
                    WHERE tag = ?
                    ORDER BY step DESC
                    LIMIT ?
                  )
                ORDER BY RANDOM()
                LIMIT 1
            )
            """,
            (tag, tag, recent_keep),
        )
    conn.commit()


def record_scalars_batch(
    conn: sqlite3.Connection,
    step: int,
    values: dict,
    wall_time: Optional[float] = None,
    cap: int = DEFAULT_SCALAR_CAP,
    recent_keep: int = DEFAULT_SCALAR_RECENT_KEEP,
) -> None:
    """Insert many tag→value pairs at the same step in one transaction."""
    if wall_time is None:
        wall_time = time.time()
    for tag, value in values.items():
        conn.execute(
            "INSERT INTO scalars(tag, step, wall_time, value) VALUES(?, ?, ?, ?)",
            (tag, step, wall_time, float(value)),
        )
        _bump_seen(conn, tag, cap)
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM scalars WHERE tag=?", (tag,)
        ).fetchone()["c"]
        if count > cap:
            conn.execute(
                """
                DELETE FROM scalars
                WHERE rowid IN (
                    SELECT rowid FROM scalars
                    WHERE tag = ?
                      AND rowid NOT IN (
                        SELECT rowid FROM scalars
                        WHERE tag = ?
                        ORDER BY step DESC
                        LIMIT ?
                      )
                    ORDER BY RANDOM()
                    LIMIT 1
                )
                """,
                (tag, tag, recent_keep),
            )
    conn.commit()


def read_scalars(
    conn: sqlite3.Connection,
    tag: str,
    step_min: Optional[int] = None,
    step_max: Optional[int] = None,
    max_points: Optional[int] = None,
) -> list[tuple[int, float, float]]:
    """Return ``(step, wall_time, value)`` rows, optionally range-filtered and
    stride-downsampled to at most ``max_points`` while preserving the endpoints."""
    sql = "SELECT step, wall_time, value FROM scalars WHERE tag=?"
    params: list = [tag]
    if step_min is not None:
        sql += " AND step >= ?"
        params.append(step_min)
    if step_max is not None:
        sql += " AND step <= ?"
        params.append(step_max)
    sql += " ORDER BY step ASC"
    rows = [(r["step"], r["wall_time"], r["value"]) for r in conn.execute(sql, params)]
    if max_points is not None and len(rows) > max_points and max_points >= 2:
        stride = (len(rows) - 1) / (max_points - 1)
        rows = [rows[int(round(i * stride))] for i in range(max_points)]
    return rows


def list_scalar_tags(conn: sqlite3.Connection) -> list[str]:
    return [r["tag"] for r in conn.execute("SELECT DISTINCT tag FROM scalars ORDER BY tag")]


# ---------- figures ----------

def record_figure(
    conn: sqlite3.Connection,
    tag: str,
    step: int,
    blob: bytes,
    mime: str = "image/png",
    wall_time: Optional[float] = None,
    cap: int = DEFAULT_FIGURE_CAP,
) -> None:
    """Insert a figure blob; drop oldest for this tag when the cap is exceeded."""
    if wall_time is None:
        wall_time = time.time()
    conn.execute(
        "INSERT INTO figures(tag, step, wall_time, mime, blob) VALUES(?, ?, ?, ?, ?)",
        (tag, step, wall_time, mime, blob),
    )
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM figures WHERE tag=?", (tag,)
    ).fetchone()["c"]
    if count > cap:
        conn.execute(
            """
            DELETE FROM figures WHERE rowid IN (
                SELECT rowid FROM figures WHERE tag=? ORDER BY step ASC LIMIT ?
            )
            """,
            (tag, count - cap),
        )
    conn.commit()


def list_figure_index(conn: sqlite3.Connection, tag: str) -> list[tuple[int, float, str]]:
    return [
        (r["step"], r["wall_time"], r["mime"])
        for r in conn.execute(
            "SELECT step, wall_time, mime FROM figures WHERE tag=? ORDER BY step ASC",
            (tag,),
        )
    ]


def read_figure_blob(conn: sqlite3.Connection, tag: str, step: int) -> Optional[tuple[bytes, str]]:
    row = conn.execute(
        "SELECT blob, mime FROM figures WHERE tag=? AND step=? LIMIT 1", (tag, step)
    ).fetchone()
    return None if row is None else (row["blob"], row["mime"])


def list_figure_tags(conn: sqlite3.Connection) -> list[str]:
    return [r["tag"] for r in conn.execute("SELECT DISTINCT tag FROM figures ORDER BY tag")]
