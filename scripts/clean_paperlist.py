#!/usr/bin/env python3
"""
Clean split paperlist files:
  1. Remove survey content entirely (Papers.md only, the whole "### 1. Survey Papers" section).
  2. Remove all paper entries from 2024 and before (all files EXCEPT News.md).
  3. News.md is kept as-is.
"""

import argparse
import re
from pathlib import Path


# Venue names that appear before a year
VENUES = (
    r"Arxiv|ICLR|ICML|NeurIPS|NIPS|EMNLP|ACL|NAACL|TACL|AAAI|CVPR|ICCV|ECCV|"
    r"SIGMOD|ISCA|MICRO|HPCA|ASPLOS|OSDI|SOSP|EuroSys|NSDI|ATC|FAST|SC|PPoPP|"
    r"CGO|PACT|DAC|ICCAD|USENIX\s+ATC|ICLR|ICML|NeurIPS"
)

# Years to remove (paper entries from 2024 and before)
OLD_YEARS = r"2019|2020|2021|2022|2023|2024"

# Regex: Venue YYYY (e.g. "Arxiv 2024", "ICLR 2023", "[ASPLOS 2024]")
RE_VENUE_YEAR = re.compile(
    rf"(?:\b|\[)(?:{VENUES})[\s']+(?:{OLD_YEARS})(?:\b|\])",
    re.IGNORECASE,
)

# Broader: just a year 2019-2024 preceded by a venue-like word boundary
RE_OLD_YEAR_LOOSE = re.compile(
    rf"(?:Arxiv|ICLR|ICML|NeurIPS|NIPS|EMNLP|ACL|NAACL|TACL|AAAI|CVPR|ICCV|"
    rf"ECCV|SIGMOD|ISCA|MICRO|HPCA|ASPLOS|OSDI|SOSP|EuroSys|NSDI|ATC|FAST|SC|"
    rf"PPoPP|CGO|PACT|DAC|ICCAD|USENIX)\s*'?\s*(?:2019|2020|2021|2022|2023|2024)",
    re.IGNORECASE,
)

# Also catch short year abbreviations like '24  '23  '22 etc.
RE_SHORT_YEAR = re.compile(
    rf"(?:'2[0-4])(?:\b|$)",
)


def line_is_pre2025(line: str) -> bool:
    """Return True if this line is a paper entry from 2024 or before."""
    return bool(RE_OLD_YEAR_LOOSE.search(line))


def clean_papers_md(path: Path, dry_run: bool) -> int:
    """Clean Papers.md: remove survey section + pre-2025 entries."""
    lines = path.read_text(encoding="utf-8").splitlines()

    # Find survey section boundaries
    survey_start = None
    survey_end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("### 1. Survey Papers"):
            survey_start = i
        if survey_start is not None and line.strip().startswith("### 2. "):
            survey_end = i
            break

    if survey_start is None:
        print(f"  WARNING: Could not find '### 1. Survey Papers' in {path.name}")
        survey_start = -1
        survey_end = -1

    # Build cleaned output
    kept = []
    removed_survey = 0
    removed_old = 0

    for i, line in enumerate(lines):
        # Skip survey section
        if survey_start is not None and survey_start <= i < (survey_end or len(lines)):
            removed_survey += 1
            continue

        # Skip pre-2025 paper entries
        if line_is_pre2025(line):
            removed_old += 1
            # Also skip the next line if it's a GitHub badge line
            continue

        kept.append(line)

    # Clean up: collapse 3+ consecutive blank lines into 2
    cleaned = []
    blank_count = 0
    for line in kept:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    output = "\n".join(cleaned) + "\n"

    if dry_run:
        print(f"  [{path.name}] Would remove {removed_survey} survey lines + "
              f"{removed_old} pre-2025 entries ({len(lines)} -> {len(cleaned)} lines)")
    else:
        path.write_text(output, encoding="utf-8")
        print(f"  [{path.name}] Removed {removed_survey} survey lines + "
              f"{removed_old} pre-2025 entries ({len(lines)} -> {len(cleaned)} lines)")

    return removed_survey + removed_old


def clean_generic(path: Path, dry_run: bool) -> int:
    """Clean a generic split file: remove pre-2025 entries only."""
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    removed = 0

    for line in lines:
        if line_is_pre2025(line):
            removed += 1
            continue
        kept.append(line)

    if dry_run:
        if removed > 0:
            print(f"  [{path.name}] Would remove {removed} pre-2025 entries "
                  f"({len(lines)} -> {len(kept)} lines)")
    else:
        if removed > 0:
            output = "\n".join(kept) + "\n"
            path.write_text(output, encoding="utf-8")
            print(f"  [{path.name}] Removed {removed} pre-2025 entries "
                  f"({len(lines)} -> {len(kept)} lines)")

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean split paperlists: remove surveys and pre-2025 papers."
    )
    parser.add_argument(
        "split_dir",
        type=Path,
        help="Directory containing the split paperlist .md files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not modify files.",
    )
    args = parser.parse_args()

    split_dir = args.split_dir.expanduser().resolve()
    if not split_dir.is_dir():
        print(f"Error: not a directory: {split_dir}")
        return 1

    all_files = sorted(split_dir.glob("*.md"))
    total_removed = 0

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Cleaning {len(all_files)} files in {split_dir}\n")

    for f in all_files:
        if f.name == "News.md":
            print(f"  [News.md] Keeping as-is (user requested)")
            continue

        if f.name == "Papers.md":
            removed = clean_papers_md(f, dry_run=args.dry_run)
        else:
            removed = clean_generic(f, dry_run=args.dry_run)
        total_removed += removed

    print(f"\nDone. {'Would remove' if args.dry_run else 'Removed'} "
          f"{total_removed} total lines across all files.")


if __name__ == "__main__":
    main()
