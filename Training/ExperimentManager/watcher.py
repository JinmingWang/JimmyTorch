"""Polling watcher that emits WS ``tree_updated`` events on changes.

Deliberately naïve: rescans the ``Runs/`` tree every ``poll_interval`` seconds
and broadcasts the full tree when its JSON signature changes. This is O(runs)
per tick which is fine for typical local dev (hundreds of runs at most).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .run_scanner import scan_tree

log = logging.getLogger("experiment_manager.watcher")


class TreeWatcher:
    def __init__(self, runs_root: Path, broadcaster: Any, poll_interval: float = 2.0) -> None:
        self.runs_root = runs_root
        self.broadcaster = broadcaster
        self.poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._last_signature: str = ""
        self._last_tree: dict[str, Any] = {"runs_root": str(runs_root), "datasets": {}}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="ExpManagerTreeWatcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def snapshot(self) -> dict[str, Any]:
        return dict(self._last_tree)

    async def _loop(self) -> None:
        while True:
            try:
                tree = await asyncio.get_event_loop().run_in_executor(None, scan_tree, self.runs_root)
                sig = json.dumps(tree, sort_keys=True, default=str)
                if sig != self._last_signature:
                    self._last_signature = sig
                    self._last_tree = tree
                    await self.broadcaster.broadcast({"type": "tree_updated", "payload": tree})
            except Exception:
                log.exception("tree watcher error")
            await asyncio.sleep(self.poll_interval)
