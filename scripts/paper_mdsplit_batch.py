#!/usr/bin/env python3
"""Split MD files from path1 subdirectories using mdsplit CLI, then copy companion JPGs to path2.

mdsplit derives each output filename from the heading text, so a heading it cannot turn
into a filename makes it drop that section and everything after it -- silently, with
returncode 0. Marker produces such headings occasionally: a LaTeX ``$$...$$`` block
promoted to level 1, or an OCR degeneration that repeats a phrase for thousands of
characters. To keep that from reaching the corpus unnoticed this script demotes
unusable level-1 headings to level 2 in a scratch copy before splitting, and then
reconciles the split bytes against the source and fails loudly when they disagree.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# mdsplit builds filenames from heading text; anything longer than this cannot survive
# the filesystem name limit once mdsplit adds its own decoration.
MAX_HEADING_CHARS = 120

# Fraction of the source .md bytes the split files must add up to. The corpus audit put
# healthy papers at >=100% and only a handful between 90% and 100%.
MIN_BYTE_RETENTION = 0.90

_WORD = re.compile(r"\w", re.UNICODE)


def unusable_heading(body):
    """True when mdsplit cannot build a filename from this level-1 heading text.

    Three forms have been seen in Marker output:
    - a LaTeX ``$$...$$`` block promoted to level 1
    - an OCR degeneration repeating a phrase for thousands of characters
    - decoration only, such as ``# #####``, which leaves mdsplit with an empty
      filename and makes it raise ValueError (killing the whole file's split)
    """
    if "$$" in body:
        return True
    if len(body) > MAX_HEADING_CHARS:
        return True
    # mdsplit strips markdown/HTML before building the name; no word character left
    # means it ends up with "" and raises.
    if not _WORD.search(re.sub(r"<[^>]*>", "", body)):
        return True
    return False


def sanitize_headings(md_file, scratch_dir):
    """Copy md_file into scratch_dir, demoting unusable level-1 headings to level 2.

    Returns (path_to_use, demoted_count). The original file is never modified.
    """
    text = md_file.read_text(encoding="utf-8", errors="replace")
    out, demoted = [], 0
    for line in text.split("\n"):
        if line.startswith("# ") and unusable_heading(line[2:]):
            out.append("#" + line)  # "# x" -> "## x"
            demoted += 1
        else:
            out.append(line)
    if not demoted:
        return md_file, 0
    scratch = scratch_dir / md_file.name
    scratch.write_text("\n".join(out), encoding="utf-8")
    return scratch, demoted


def split_bytes(out_dir):
    return sum(f.stat().st_size for f in out_dir.glob("*.md"))


def main():
    parser = argparse.ArgumentParser(
        description="For each subdir under path1, run mdsplit on its .md file "
                    "and copy all .jpg/.jpeg files to the corresponding output subdir under path2."
    )
    parser.add_argument("path1", help="Source directory containing subdirectories (each with 1 MD + JPGs)")
    parser.add_argument("path2", help="Output base directory")
    args = parser.parse_args()

    src_base = Path(args.path1)
    dst_base = Path(args.path2)

    if not src_base.is_dir():
        print(f"Error: {src_base} is not a directory", file=sys.stderr)
        sys.exit(1)

    dst_base.mkdir(parents=True, exist_ok=True)

    subdirs = sorted(d for d in src_base.iterdir() if d.is_dir())
    if not subdirs:
        print(f"No subdirectories found in {src_base}")
        sys.exit(0)

    failures = []

    for subdir in subdirs:
        name = subdir.name
        print(f"[{name}]")

        md_files = sorted(subdir.glob("*.md"))
        if not md_files:
            print(f"  SKIP: no .md file found")
            continue

        if len(md_files) > 1:
            print(f"  NOTE: {len(md_files)} .md files present, splitting only {md_files[0].name}")

        md_file = md_files[0]
        out_dir = dst_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="mdsplit_") as scratch:
            split_src, demoted = sanitize_headings(md_file, Path(scratch))
            if demoted:
                print(f"  demoted {demoted} unusable level-1 heading(s) before splitting")

            # Run mdsplit
            cmd = [
                "mdsplit",
                str(split_src.resolve()),
                "--max-level", "1",
                "--output", str(out_dir.resolve()),
                "--force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  mdsplit FAILED: {result.stderr.strip()}", file=sys.stderr)
            failures.append((name, "mdsplit returned non-zero"))
        else:
            if result.stdout.strip():
                print(f"  mdsplit: {result.stdout.strip()}")

        # mdsplit exits 0 even when it silently drops sections, so reconcile the bytes.
        src_bytes = md_file.stat().st_size
        dst_bytes = split_bytes(out_dir)
        if src_bytes and dst_bytes < src_bytes * MIN_BYTE_RETENTION:
            pct = dst_bytes * 100 // src_bytes
            print(
                f"  CONTENT LOSS: split kept {dst_bytes}/{src_bytes} bytes ({pct}%) "
                f"-- sections were dropped, do not use this split as-is",
                file=sys.stderr,
            )
            failures.append((name, f"kept only {pct}% of source bytes"))

        # Copy JPGs
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for ext in exts:
            for jpg in subdir.glob(ext):
                shutil.copy2(jpg, out_dir / jpg.name)
                print(f"  copied: {jpg.name}")

    if failures:
        print(f"\nDone with {len(failures)} problem paper(s):", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        print(
            "\nRe-check these against papers_md/<batch>/<title>/ before running "
            "run_all_papers.py on them.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
