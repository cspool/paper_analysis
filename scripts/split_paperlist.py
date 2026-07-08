#!/usr/bin/env python3
"""Split a paperlist markdown by ## headings into separate files for batch download."""

import argparse
import os
import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Convert a heading into a safe filename."""
    # Remove markdown formatting
    name = name.strip()
    # Remove leading ## and whitespace
    name = re.sub(r"^##\s+", "", name)
    # Replace problematic chars
    name = name.replace("/", "-").replace("\\", "-").replace(":", " -")
    name = name.replace("?", "").replace("*", "").replace('"', "")
    name = name.replace("<", "").replace(">", "").replace("|", "-")
    # Collapse whitespace
    name = re.sub(r"\s+", "_", name.strip())
    return name


def split_paperlist(input_path: Path, output_dir: Path) -> list[Path]:
    """Split a paperlist markdown into per-section files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find all ## section boundaries
    section_starts: list[tuple[int, str]] = []  # (line_index, heading)
    for i, line in enumerate(lines):
        if re.match(r"^##\s+", line):
            section_starts.append((i, line))

    if not section_starts:
        print("No ## headings found.")
        return []

    # Build sections: each spans from its start line to the line before the next heading
    output_files = []
    for idx, (start, heading) in enumerate(section_starts):
        end = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
        section_lines = lines[start:end]

        # Preserve top-level headings (# only) that appear between ## sections
        fname = sanitize_filename(heading) + ".md"
        out_path = output_dir / fname

        content = "\n".join(section_lines).strip() + "\n"
        out_path.write_text(content, encoding="utf-8")
        output_files.append(out_path)
        print(f"  [{len(section_lines):4d} lines] {fname}")

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a paperlist markdown by ## headings."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Paperlist markdown file to split.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/split_paperlists/).",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}")
        raise SystemExit(1)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_path.parent / "split_paperlists"
    )

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"\nSplitting by ## sections:\n")

    files = split_paperlist(input_path, output_dir)
    print(f"\nDone. {len(files)} files written to {output_dir}")


if __name__ == "__main__":
    main()
