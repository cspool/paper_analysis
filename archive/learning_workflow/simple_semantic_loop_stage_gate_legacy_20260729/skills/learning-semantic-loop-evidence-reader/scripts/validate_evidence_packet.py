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
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--task-only", action="store_true")
    args = parser.parse_args()
    if args.task_only == (args.packet is not None):
        parser.error("choose exactly one of --task-only or --packet")
    command = [
        "node",
        str(VALIDATOR),
        "evidence-task" if args.task_only else "evidence",
        "--task",
        str(args.task),
        "--pretty",
    ]
    if args.packet is not None:
        command.extend(["--packet", str(args.packet)])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
