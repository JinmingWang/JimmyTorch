"""JimmyTorch ExperimentManager package.

Stage 2 exposes:
- ``ExperimentLogger`` — per-run SQLite writer (Stage 1).
- ``ExperimentManagerClient`` — drop-in replacement for the old
  ``ProgressManagerGUI`` that reports to a decoupled aiohttp server.
- ``run_server`` — programmatic entry point for launching the server.
"""
from .client import ExperimentManagerClient
from .constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    GLOBAL_DB_FILENAME,
    RUN_DB_FILENAME,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_EVALUATING,
    STATUS_IDLE,
    STATUS_TRAINING,
)
from .logger import ExperimentLogger


def __getattr__(name):
    if name in {"build_app", "run_server"}:
        from . import server
        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ExperimentLogger",
    "ExperimentManagerClient",
    "build_app",
    "run_server",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "GLOBAL_DB_FILENAME",
    "RUN_DB_FILENAME",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_EVALUATING",
    "STATUS_IDLE",
    "STATUS_TRAINING",
]
