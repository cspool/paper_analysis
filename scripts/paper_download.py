#!/usr/bin/env python3
"""Download papers by title through the existing agent_research downloader."""

import argparse
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_BACKEND = Path("/data3/agent_research/download_papers.py")
DEFAULT_OUTPUT = Path.cwd() / "paper_downloads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download papers by exact title or from title-list files. "
            "Uses /data3/agent_research/download_papers.py in place."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--title",
        action="append",
        help="Paper title to download. Repeat --title to download several papers.",
    )
    source.add_argument(
        "--file",
        action="append",
        dest="title_files",
        help=(
            "Title-list file. Repeat --file for several files. Plain Markdown "
            "headings are skipped; other non-empty lines and paper links are supported."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"PDF output directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("PAPER_DOWNLOAD_BACKEND", str(DEFAULT_BACKEND)),
        help="Existing download_papers.py path.",
    )
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between papers.")
    parser.add_argument("--max-time", type=int, default=60, help="Request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print parsed titles only.")
    parser.add_argument("--no-arxiv-fallback", action="store_true")
    parser.add_argument("--no-oa-fallback", action="store_true")
    parser.add_argument("--no-dblp-fallback", action="store_true")
    parser.add_argument("--no-known-public-fallback", action="store_true")
    parser.add_argument(
        "--skip-source-discovery",
        action="store_true",
        help="Skip probing source pages from Markdown links.",
    )
    return parser.parse_args()


def load_backend(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Downloader backend not found: {path}")

    spec = importlib.util.spec_from_file_location("agent_research_download_papers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load downloader backend: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_title_line(line: str) -> str:
    title = line.strip()
    title = re.sub(r"^#{1,6}\s+", "", title)
    title = re.sub(r"^(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s*)?", "", title)
    title = re.sub(r"^[>|]\s*", "", title)
    title = re.sub(r"[*_`]+", "", title)
    return title.strip(" \t|")


def title_entry(title: str, origin: str, line_no: int) -> dict[str, Any]:
    return {
        "title": title,
        "source_url": "",
        "pdf_url": "",
        "source_id": "",
        "input_file": origin,
        "line": line_no,
    }


def entries_from_file(path: Path, backend: ModuleType) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Title-list file not found: {path}")

    entries: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue

        link_entries = []
        for text, raw_url in backend.LINK_PATTERN.findall(line):
            if backend.is_skipped_link(text, raw_url):
                continue
            pdf_url, source_id = backend.pdf_url_for(raw_url)
            link_entries.append(
                {
                    "title": backend.title_from_line(line, text),
                    "source_url": backend.normalize_url(raw_url),
                    "pdf_url": backend.normalize_url(pdf_url or raw_url),
                    "source_id": source_id,
                    "input_file": str(path),
                    "line": line_no,
                }
            )

        if link_entries:
            entries.extend(link_entries)
            continue

        if re.match(r"^#{1,6}\s+", stripped):
            continue

        title = clean_title_line(line)
        if title:
            entries.append(title_entry(title, str(path), line_no))

    return entries


def deduplicate_entries(
    entries: list[dict[str, Any]], backend: ModuleType
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen = set()
    for entry in entries:
        key = backend.normalize_title(entry["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def collect_entries(args: argparse.Namespace, backend: ModuleType) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.title:
        for idx, raw_title in enumerate(args.title, 1):
            title = raw_title.strip()
            if title:
                entries.append(title_entry(title, "<command-line>", idx))
    else:
        for raw_path in args.title_files:
            entries.extend(entries_from_file(Path(raw_path).expanduser().resolve(), backend))
    return deduplicate_entries(entries, backend)


def main() -> int:
    args = parse_args()
    try:
        backend = load_backend(Path(args.backend).expanduser().resolve())
        entries = collect_entries(args, backend)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print("Error: no paper titles found.", file=sys.stderr)
        return 2

    output_dir = Path(args.output).expanduser().resolve()
    print(f"Parsed {len(entries)} unique paper title(s)")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        for idx, entry in enumerate(entries, 1):
            print(f"[{idx}/{len(entries)}] {entry['title']}")
            if entry["source_url"]:
                print(f"  source: {entry['source_url']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    results = []

    for idx, entry in enumerate(entries, 1):
        print(f"\n[{idx}/{len(entries)}] {entry['title'][:100]}")
        result = backend.download_entry(
            entry,
            output_dir,
            args.max_time,
            use_arxiv_fallback=not args.no_arxiv_fallback,
            use_oa_fallback=not args.no_oa_fallback,
            use_dblp_fallback=not args.no_dblp_fallback,
            use_known_public_fallback=not args.no_known_public_fallback,
            skip_source_discovery=args.skip_source_discovery,
        )
        results.append(result)
        backend.save_results(results_file, results)

        if result["status"] in ("downloaded", "exists"):
            print(f"  {result['status'].upper()} ({result['size']} bytes)")
        else:
            print(f"  FAILED: {result['reason']}")

        if idx < len(entries):
            time.sleep(max(0.0, args.delay))

    success = sum(result["status"] in ("downloaded", "exists") for result in results)
    print(f"\nDone. {success}/{len(results)} available. Results: {results_file}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
