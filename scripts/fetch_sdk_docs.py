#!/usr/bin/env python3
"""Fetch SDK/tool documentation (GitHub README or web page) as .md files."""

import re
import subprocess
import sys
from pathlib import Path

BASE = Path("/data3/paper_analysis/papers_pdf/2026-07-02_downloads")

# Map: directory_name -> [(title, url), ...]
SDK_ENTRIES = {
    "General_ML_Workloads": [
        ("LiteRT", "https://ai.google.dev/edge/litert"),
        ("ExecuTorch", "https://github.com/pytorch/executorch"),
        ("ONNX Runtime", "https://onnxruntime.ai/"),
        ("MNN", "https://github.com/alibaba/MNN"),
        ("NCNN", "https://github.com/Tencent/ncnn"),
    ],
    "Vendor-Specific_SDKs": [
        ("Qualcomm QNN", "https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"),
        ("Apple Core ML", "https://developer.apple.com/documentation/coreml"),
        ("FluidAudio", "https://github.com/FluidInference/FluidAudio"),
        ("NVIDIA TensorRT", "https://developer.nvidia.com/tensorrt"),
        ("Intel OpenVINO", "https://github.com/openvinotoolkit/openvino"),
        ("MediaTek NeuroPilot", "https://neuropilot.mediatek.com/"),
    ],
    "LLM_&_GenAI_Specialized": [
        ("llama.cpp", "https://github.com/ggerganov/llama.cpp"),
        ("MLC LLM", "https://github.com/mlc-ai/mlc-llm"),
        ("TensorRT-LLM", "https://github.com/NVIDIA/TensorRT-LLM"),
        ("mllm", "https://github.com/UbiquitousLearning/mllm"),
        ("MLX LM", "https://github.com/ml-explore/mlx-lm"),
        ("OmniInfer", "https://github.com/omnimind-ai/OmniInfer-VLM"),
        ("RunAnywhere", "https://github.com/RunanywhereAI/runanywhere-sdks"),
        ("Off Grid", "https://github.com/alichherawalla/off-grid-mobile-ai"),
    ],
}

GITHUB_RE = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$")


def github_readme_url(url: str) -> str | None:
    """Convert a GitHub repo URL to its raw README URL. Tries main, then master."""
    m = GITHUB_RE.search(url.rstrip("/"))
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    for branch in ("main", "master"):
        raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
        if _url_ok(raw):
            return raw
    return None


def _url_ok(url: str) -> bool:
    r = subprocess.run(
        ["curl", "-sL", "--connect-timeout", "10", "--max-time", "30",
         "-o", "/dev/null", "-w", "%{http_code}", url],
        capture_output=True, text=True, timeout=40,
    )
    return r.stdout.strip() == "200"


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '', name).strip()


def fetch_and_save(url: str, out_path: Path) -> bool:
    """Fetch URL content and save as markdown."""
    cmd = [
        "curl", "-sL", "--connect-timeout", "15", "--max-time", "60",
        "-H", "User-Agent: Mozilla/5.0",
        "-o", str(out_path),
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        return False
    if out_path.stat().st_size < 100:
        return False
    return True


def main():
    total_ok, total_fail = 0, 0

    for dir_name, entries in SDK_ENTRIES.items():
        out_dir = BASE / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for title, url in entries:
            fname = sanitize_filename(title) + ".md"
            out_path = out_dir / fname

            if out_path.exists() and out_path.stat().st_size > 500:
                print(f"  SKIP (exists): {dir_name}/{fname}")
                total_ok += 1
                continue

            # Try GitHub README first
            raw = github_readme_url(url)
            if raw:
                print(f"  GITHUB: {title} → {raw[:80]}...")
                ok = fetch_and_save(raw, out_path)
            else:
                # Fetch web page as-is
                print(f"  WEB:    {title} → {url[:80]}...")
                ok = fetch_and_save(url, out_path)

            if ok:
                print(f"    ✅ {out_path.stat().st_size} bytes")
                total_ok += 1
            else:
                print(f"    ❌ failed")
                total_fail += 1

    print(f"\nDone: {total_ok} ok, {total_fail} failed")


if __name__ == "__main__":
    main()
