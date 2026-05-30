#!/usr/bin/env python3
"""Split MD files from path1 subdirectories using mdsplit CLI, then copy companion JPGs to path2."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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

    for subdir in subdirs:
        name = subdir.name
        print(f"[{name}]")

        md_files = sorted(subdir.glob("*.md"))
        if not md_files:
            print(f"  SKIP: no .md file found")
            continue

        md_file = md_files[0]
        out_dir = dst_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run mdsplit
        cmd = [
            "mdsplit",
            str(md_file.resolve()),
            "--max-level", "1",
            "--output", str(out_dir.resolve()),
            "--force",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  mdsplit FAILED: {result.stderr.strip()}", file=sys.stderr)
        else:
            if result.stdout.strip():
                print(f"  mdsplit: {result.stdout.strip()}")

        # Copy JPGs
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for ext in exts:
            for jpg in subdir.glob(ext):
                shutil.copy2(jpg, out_dir / jpg.name)
                print(f"  copied: {jpg.name}")

    print("Done.")


if __name__ == "__main__":
    main()
