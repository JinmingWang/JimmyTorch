"""Read compact, per-run comparison values without loading model weights or figures."""
from __future__ import annotations

import ast
from contextlib import closing
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

from .constants import RUN_DB_FILENAME
from .run_scanner import resolve_run_dir


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return value
        except OverflowError:
            pass
    return None


def _literal(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError, OverflowError):
        return None


def _numeric_hparams(params):
    result = {}

    def visit(value, path):
        if isinstance(value, str) and value.startswith("-- ") and " Config --\n" in value:
            # DynamicConfig.__str__: optimizer representations can span several lines.
            fields = {}
            for line in value.splitlines()[1:]:
                key, sep, raw = line.partition(": ")
                if sep and key.isidentifier():
                    fields[key] = _literal(raw)
            value = fields
        elif isinstance(value, str) and value.startswith("Config of "):
            # Older DynamicConfig stored dict reprs containing Python class objects.
            raw = value.partition(": ")[2]
            value = _literal(re.sub(r"<class '[\w.]+'\s*>", "None", raw))
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, str(key)])
        elif path and _number(value) is not None:
            # Escape literal dots so a nested path and a flat custom key stay distinct.
            result[".".join(key.replace("\\", "\\\\").replace(".", "\\.") for key in path)] = value

    visit(params, [])
    return result


def _metrics(conn):
    # The indexed last-step lookup also breaks ties for repeated steps. Aggregate
    # over retained samples in SQLite so long histories never go over the wire.
    rows = conn.execute("""
        SELECT tag, MIN(value), MAX(value),
               (SELECT value FROM scalars AS s
                WHERE s.tag = series.tag AND s.value BETWEEN ?1 AND ?2
                ORDER BY step DESC, rowid DESC LIMIT 1)
        FROM scalars AS series WHERE value BETWEEN ?1 AND ?2 GROUP BY tag
    """, (-sys.float_info.max, sys.float_info.max))
    return {tag: {"lowest": low, "highest": high, "last": last} for tag, low, high, last in rows}


def _throughput(conn):
    # LR is written against the trainer's global iteration, including distributed runs.
    first = conn.execute("SELECT step, wall_time FROM scalars WHERE tag='LR' "
                         "ORDER BY step, rowid LIMIT 1").fetchone()
    last = conn.execute("SELECT step, wall_time FROM scalars WHERE tag='LR' "
                        "ORDER BY step DESC, rowid DESC LIMIT 1").fetchone()
    if first and last and last[0] > first[0] and last[1] > first[1]:
        return _number((last[0] - first[0]) / (last[1] - first[1]))
    return None


def _read_run(run_dir):
    measurements, hparams, metrics = {}, {}, {}
    for filename in ("last.pth", "best.pth"):
        path = run_dir / filename
        if path.is_file() and not path.is_symlink():
            measurements["checkpoint_mib"] = path.stat().st_size / (1024 ** 2)
            break

    profile = run_dir / "model_profile.json"
    if profile.is_file() and not profile.is_symlink():
        try:
            values = json.loads(profile.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                for name in ("parameters", "forward_macs", "forward_flops"):
                    value = _number(values.get(name))
                    if value is not None and value >= 0:
                        measurements[name] = value
        except (ValueError, OSError, RecursionError):
            pass

    db = run_dir / RUN_DB_FILENAME
    if db.is_file() and not db.is_symlink():
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
            conn.execute("BEGIN")
            try:
                meta = dict(conn.execute("SELECT key, value FROM meta WHERE key IN "
                                         "('parameters','hparams','model_profile','iterations_per_second')"))
            except sqlite3.DatabaseError:
                meta = {}
            try:
                recorded = json.loads(meta.get("model_profile") or "{}")
                if isinstance(recorded, dict):
                    for key in ("parameters", "forward_macs", "forward_flops", "checkpoint_mib"):
                        value = _number(recorded.get(key))
                        if value is not None and value >= 0:
                            measurements.setdefault(key, value)
                if "parameters" not in measurements and meta.get("parameters"):
                    count = int(meta["parameters"])
                    if _number(count) is not None and count >= 0:
                        measurements["parameters"] = count
            except (ValueError, TypeError, RecursionError):
                pass
            try:
                hparams = _numeric_hparams(json.loads(meta.get("hparams") or "{}"))
            except (ValueError, TypeError, RecursionError):
                pass
            try:
                metrics = _metrics(conn)
                rate = _number(float(meta.get("iterations_per_second") or 0))
                if not rate or rate <= 0:
                    rate = _throughput(conn)
                if rate is not None:
                    measurements["iterations_per_second"] = rate
            except (sqlite3.DatabaseError, ValueError, TypeError):
                pass
    return {"measurements": measurements, "hparams": hparams, "metrics": metrics}


def comparison_rows(root, datasets):
    rows = []
    for dataset, models in datasets.items():
        for model, runs in models.items():
            for run in runs:
                try:
                    directory = resolve_run_dir(root, dataset, model, run)
                    values = _read_run(directory)
                except (OSError, ValueError, sqlite3.DatabaseError):
                    continue
                rows.append({"dataset": dataset, "model": model, "run_name": run, **values})
    return rows
