#!/usr/bin/env python3
"""Batch split MD files from a repo directory's sub-repos by ## headings."""

import argparse
import re
import sys
from pathlib import Path

# Mapping from experiment MD name (without .md) to output subdirectory
EXP_NAME_TO_DIR = {
    "实验_编译框架":   "编译实验笔记",
    "实验_算法pipeline": "算法实验笔记",
    "实验_芯片设计":   "芯片实验笔记",
    "实验_硬件架构":   "硬件实验笔记",
    "实验_kernel调度": "kernel实验笔记",
    "实验_Serving调度": "系统实验笔记",
}

# Mapping from knowledge MD name (without .md) to output subdirectory
KNOW_NAME_TO_DIR = {
    "知识库_编译框架":   "编译知识笔记",
    "知识库_算法pipeline": "算法知识笔记",
    "知识库_系统架构":   "系统知识笔记",
    "知识库_芯片设计":   "芯片知识笔记",
    "知识库_硬件架构":   "硬件知识笔记",
    "知识库_kernel调度": "kernel知识笔记",
}

NOTES_BASE = Path("/data3/paper_analysis")


def sanitize_filename(name: str, max_len: int = 100) -> str:
    name = name.strip()
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def split_md(src_path: Path, dst_dir: Path) -> int:
    """Split a single MD file by ## headings. Returns number of sections written."""
    content = src_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    sections = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            title = line[3:].strip()
            sections.append((i, title))

    if not sections:
        print(f"  No ## headings found, skipping")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, (start_line, title) in enumerate(sections):
        end_line = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
        body = "\n".join(lines[start_line:end_line]).strip()
        out_path = dst_dir / (sanitize_filename(title) + ".md")
        out_path.write_text(body + "\n", encoding="utf-8")
        print(f"    -> {out_path}")

    return len(sections)


def resolve_output_dir(repo_name: str, md_stem: str) -> Path | None:
    """Resolve the output directory for a given MD file from a repo."""
    if repo_name == "idea_repo":
        return NOTES_BASE / "idea_notes"

    if repo_name == "experiment_repo":
        sub = EXP_NAME_TO_DIR.get(md_stem)
        if sub is None:
            print(f"  WARNING: unknown experiment name '{md_stem}', skipping", file=sys.stderr)
            return None
        return NOTES_BASE / "experiment_notes" / sub

    if repo_name == "knowledge_repo":
        sub = KNOW_NAME_TO_DIR.get(md_stem)
        if sub is None:
            print(f"  WARNING: unknown knowledge name '{md_stem}', skipping", file=sys.stderr)
            return None
        return NOTES_BASE / "knowledge_notes" / sub

    print(f"  WARNING: unknown repo '{repo_name}', skipping", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch split MD files from experiment_repo/idea_repo/knowledge_repo "
                    "by ## headings."
    )
    parser.add_argument(
        "root_repo_path",
        help="Root path containing experiment_repo/, idea_repo/, knowledge_repo/ subdirectories",
    )
    args = parser.parse_args()

    root = Path(args.root_repo_path)
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    repo_dirs = {
        "experiment_repo": root / "experiment_repo",
        "idea_repo": root / "idea_repo",
        "knowledge_repo": root / "knowledge_repo",
    }

    total_md = 0
    total_sections = 0

    for repo_name, repo_dir in repo_dirs.items():
        if not repo_dir.is_dir():
            print(f"[{repo_name}] not found, skipping")
            continue

        md_files = sorted(repo_dir.glob("*.md"))
        if not md_files:
            print(f"[{repo_name}] no .md files, skipping")
            continue

        print(f"[{repo_name}] ({len(md_files)} files)")
        for md_path in md_files:
            md_stem = md_path.stem  # filename without .md
            dst_dir = resolve_output_dir(repo_name, md_stem)
            if dst_dir is None:
                continue

            print(f"  {md_path.name} -> {dst_dir}")
            n = split_md(md_path, dst_dir)
            total_md += 1
            total_sections += n

    print(f"\nDone. Processed {total_md} files, wrote {total_sections} sections total.")


if __name__ == "__main__":
    main()
