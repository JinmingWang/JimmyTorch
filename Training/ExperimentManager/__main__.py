"""Entry point: ``python -m Training.ExperimentManager``."""
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from .server import run_server


def main() -> None:
    p = argparse.ArgumentParser(description="JimmyTorch ExperimentManager server.")
    p.add_argument("--host", default=DEFAULT_SERVER_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    p.add_argument(
        "--runs-root",
        default=None,
        help="Path to the Runs/ directory. Defaults to ./Runs relative to cwd.",
    )
    args = p.parse_args()
    runs_root = Path(args.runs_root) if args.runs_root else None
    run_server(host=args.host, start_port=args.port, runs_root=runs_root)


if __name__ == "__main__":
    main()
