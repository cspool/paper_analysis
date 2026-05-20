#!/usr/bin/env python3
"""Convert all PDFs under papers/paper_2026/ to Markdown using pymupdf4llm."""

import os
import sys
import time
import pymupdf4llm

PDF_DIR = "papers/paper_2025"

def main():
    pdf_files = []
    for root, dirs, files in os.walk(PDF_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
    pdf_files.sort()

    total = len(pdf_files)
    print(f"Found {total} PDF files to convert\n")

    ok, fail = 0, 0
    for i, pdf_path in enumerate(pdf_files):
        md_path = pdf_path.rsplit(".", 1)[0] + ".md"
        name = os.path.basename(pdf_path)
        try:
            t0 = time.time()
            markdown = pymupdf4llm.to_markdown(
                pdf_path,
                write_images=False,
                embed_images=False,
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            elapsed = time.time() - t0
            ok += 1
            print(f"[{i+1}/{total}] OK  {elapsed:.1f}s  {name}", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{total}] FAIL  {name}: {e}", flush=True)

    print(f"\nDone. {ok} success, {fail} failed.")

if __name__ == "__main__":
    main()
