#!/usr/bin/env python3
"""Thin adapter to the authoritative Simple Semantic Loop validator."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR = REPO_ROOT / "scripts" / "simple_semantic_loop" / "validate_message.ts"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("review", nargs="?", type=Path)
    parser.add_argument("--derive", action="store_true")
    args = parser.parse_args()
    if args.derive == (args.review is not None):
        parser.error("provide review, or use --derive, but not both")
    command = [
        "node",
        str(VALIDATOR),
        "closure-derive" if args.derive else "closure",
        "--task",
        str(args.task),
        "--pretty",
    ]
    if args.review is not None:
        command.extend(["--review", str(args.review)])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
