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
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--json-output", action="store_true")
    args = parser.parse_args()
    command = [
        "node",
        str(VALIDATOR),
        "direction",
        "--task",
        str(args.task),
        "--review",
        str(args.review),
    ]
    if not args.json_output:
        command.append("--pretty")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
