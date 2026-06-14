#!/usr/bin/env python3
"""Convert PDFs to Markdown through the existing Marker installation."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MARKER_ROOT = Path("/home/descfly/Desktop/marker")
DEFAULT_MARKER_PYTHON = Path("/home/descfly/miniconda3/bin/python3")
DEFAULT_OUTPUT = Path.cwd() / "paper_markdown"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Marker command without running conversion.",
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Convert papers with /home/descfly/Desktop/marker without moving its environment. "
            "Unknown options are passed through to Marker."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Convert one PDF.")
    single.add_argument("pdf", help="PDF path.")
    add_common_arguments(single)

    batch = subparsers.add_parser("batch", help="Convert all PDFs directly inside a folder.")
    batch.add_argument("folder", help="Folder containing PDFs.")
    batch.add_argument("--workers", type=int, help="Marker worker process count.")
    batch.add_argument("--skip-existing", action="store_true", help="Skip existing outputs.")
    add_common_arguments(batch)

    args, marker_args = parser.parse_known_args()
    return args, marker_args


def validate_marker(marker_root: Path, marker_python: Path, script_name: str) -> Path:
    marker_script = marker_root / script_name
    if not marker_root.is_dir():
        raise FileNotFoundError(f"Marker root not found: {marker_root}")
    if not marker_python.is_file():
        raise FileNotFoundError(f"Marker Python not found: {marker_python}")
    if not marker_script.is_file():
        raise FileNotFoundError(f"Marker script not found: {marker_script}")
    return marker_script


def ensure_valid_passthrough(marker_args: list[str]) -> None:
    blocked = {"--output_dir", "--output-dir"}
    if any(arg.split("=", 1)[0] in blocked for arg in marker_args):
        raise ValueError("Use this wrapper's --output/-o option instead of --output_dir.")


def print_command(command: list[str], cwd: Path) -> None:
    print(f"Working directory: {cwd}")
    print("Command:")
    print("  " + shlex.join(command))


def run_marker(command: list[str], marker_root: Path, dry_run: bool) -> int:
    print_command(command, marker_root)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=marker_root, check=False).returncode


def single_command(
    args: argparse.Namespace,
    marker_args: list[str],
    marker_root: Path,
    marker_python: Path,
) -> int:
    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    output = Path(args.output).expanduser().resolve()
    marker_script = validate_marker(marker_root, marker_python, "convert_single.py")
    command = [
        str(marker_python),
        str(marker_script),
        str(pdf),
        "--output_dir",
        str(output),
        *marker_args,
    ]
    return run_marker(command, marker_root, args.dry_run)


def batch_command(
    args: argparse.Namespace,
    marker_args: list[str],
    marker_root: Path,
    marker_python: Path,
) -> int:
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    pdfs = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found directly inside: {folder}")

    output = Path(args.output).expanduser().resolve()
    marker_script = validate_marker(marker_root, marker_python, "convert.py")

    print(f"Selected {len(pdfs)} PDF(s) from {folder}")
    with tempfile.TemporaryDirectory(prefix="paper-analysis-marker-") as temp_dir:
        staging = Path(temp_dir)
        for pdf in pdfs:
            (staging / pdf.name).symlink_to(pdf)

        command = [
            str(marker_python),
            str(marker_script),
            str(staging),
            "--output_dir",
            str(output),
        ]
        if args.workers is not None:
            command.extend(["--workers", str(args.workers)])
        if args.skip_existing:
            command.append("--skip_existing")
        command.extend(marker_args)
        return run_marker(command, marker_root, args.dry_run)


def main() -> int:
    args, marker_args = parse_args()
    marker_root = Path(
        os.environ.get("MARKER_ROOT", str(DEFAULT_MARKER_ROOT))
    ).expanduser().resolve()
    marker_python = Path(
        os.environ.get("MARKER_PYTHON", str(DEFAULT_MARKER_PYTHON))
    ).expanduser().resolve()

    try:
        ensure_valid_passthrough(marker_args)
        if args.mode == "single":
            return single_command(args, marker_args, marker_root, marker_python)
        return batch_command(args, marker_args, marker_root, marker_python)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
