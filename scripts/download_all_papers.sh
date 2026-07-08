#!/bin/bash
# Download papers from all split paperlists
# Usage:
#   bash scripts/download_all_papers.sh           # download all
#   bash scripts/download_all_papers.sh --dry-run  # preview only
#   bash scripts/download_all_papers.sh Serving.md # download single file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SPLIT_DIR="$ROOT/temp/split_paperlists"
DOWNLOAD_DIR="$ROOT/paper_downloads"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    shift
fi

download_one() {
    local md_file="$1"
    local name="${md_file%.md}"
    local name="${name##*/}"
    echo "=== [$name] ==="
    python3 "$ROOT/scripts/paper_download.py" \
        --file "$SPLIT_DIR/$md_file" \
        -o "$DOWNLOAD_DIR/$name" \
        $DRY_RUN
    echo ""
}

if [[ $# -gt 0 ]]; then
    # Download specified files only
    for f in "$@"; do
        download_one "$f"
    done
else
    # Download all
    echo "============================================"
    echo "Downloading all papers from $SPLIT_DIR"
    echo "Output to: $DOWNLOAD_DIR"
    [[ -n "$DRY_RUN" ]] && echo "MODE: DRY RUN (preview only)"
    echo "============================================"
    echo ""

    for f in "$SPLIT_DIR"/*.md; do
        download_one "$(basename "$f")"
    done
fi

echo "All done."
