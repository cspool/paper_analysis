#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel multi-config paper analysis launcher.

Launches N worker processes, each handling papers from a different PAPER_BASE_DIR.
Each worker records its own logs / status / progress, and passes its own
EXPERIMENT_DIR / IDEA_DIR / KNOWLEDGE_DIR to the paper-experiment-idea and
paper-knowledge skills.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/data3/paper_analysis").resolve()
ROOT_REPO = Path("/data3/paper_analysis/repos").resolve()
CLAUDE_CMD = "claude"
STOP_ON_FAILURE = False
MODEL_NAME = None

# Number of parallel worker processes
MAX_WORKERS = 3

# ── Per-config definitions ──────────────────────────────────────────────
# Each config = one PAPER_BASE_DIR → one worker process.
# Add / remove entries to match your paper directories and repos.

CONFIGS = [
    {
        "name": "moe_batch_1",
        "paper_base_dir": ROOT / "papers_md" / "md_moe_batch_1",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_moe_batch_1",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_moe_batch_1",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_moe_batch_1.json",
        "experiment_dir": ROOT_REPO / "repo_moe_batch_1" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_moe_batch_1" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_moe_batch_1" / "knowledge_repo",
    },
    {
        "name": "moe_batch_2",
        "paper_base_dir": ROOT / "papers_md" / "md_moe_batch_2",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_moe_batch_2",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_moe_batch_2",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_moe_batch_2.json",
        "experiment_dir": ROOT_REPO / "repo_moe_batch_2" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_moe_batch_2" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_moe_batch_2" / "knowledge_repo",
    },
    {
        "name": "moe_batch_3",
        "paper_base_dir": ROOT / "papers_md" / "md_moe_batch_3",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_moe_batch_3",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_moe_batch_3",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_moe_batch_3.json",
        "experiment_dir": ROOT_REPO / "repo_moe_batch_3" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_moe_batch_3" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_moe_batch_3" / "knowledge_repo",
    },
    {
        "name": "moe_batch_4",
        "paper_base_dir": ROOT / "papers_md" / "md_moe_batch_4",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_moe_batch_4",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_moe_batch_4",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_moe_batch_4.json",
        "experiment_dir": ROOT_REPO / "repo_moe_batch_4" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_moe_batch_4" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_moe_batch_4" / "knowledge_repo",
    },
    {
        "name": "multimodal_kernel_batch_1",
        "paper_base_dir": ROOT / "papers_md" / "md_multimodal_kernel_batch_1",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_multimodal_kernel_batch_1",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_multimodal_kernel_batch_1",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_multimodal_kernel_batch_1.json",
        "experiment_dir": ROOT_REPO / "repo_multimodal_kernel_batch_1" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_multimodal_kernel_batch_1" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_multimodal_kernel_batch_1" / "knowledge_repo",
    },
    {
        "name": "multimodal_kernel_batch_2",
        "paper_base_dir": ROOT / "papers_md" / "md_multimodal_kernel_batch_2",
        "log_dir": ROOT / "paper_extract_checkpoints" / "logs_multimodal_kernel_batch_2",
        "status_dir": ROOT / "paper_extract_checkpoints" / "status_multimodal_kernel_batch_2",
        "progress_file": ROOT / "paper_extract_checkpoints" / "progress_multimodal_kernel_batch_2.json",
        "experiment_dir": ROOT_REPO / "repo_multimodal_kernel_batch_2" / "experiment_repo",
        "idea_dir": ROOT_REPO / "repo_multimodal_kernel_batch_2" / "idea_repo",
        "knowledge_dir": ROOT_REPO / "repo_multimodal_kernel_batch_2" / "knowledge_repo",
    },
]


# ══════════════════════════════════════════════════════════════════════════
# Utility functions (unchanged from original, now parameterised)
# ══════════════════════════════════════════════════════════════════════════

def render_claude_stream_event(event, collected_text):
    event_type = event.get("type")

    if event_type == "system":
        subtype = event.get("subtype")
        if subtype == "init":
            print(
                f"\n[init] model={event.get('model')} "
                f"session_id={event.get('session_id')}\n",
                flush=True,
            )
        return

    if event_type == "stream_event":
        se = event.get("event", {})
        if se.get("type") == "content_block_delta":
            delta = se.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    print(text, end="", flush=True)
                    collected_text.append(text)
        return

    if event_type == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    print(text, end="", flush=True)
                    collected_text.append(text)
            elif block.get("type") == "tool_use":
                name = block.get("name", "unknown_tool")
                short = summarize_tool_input(name, block.get("input", {}))
                print(f"\n[tool_use] {name} {short}\n", flush=True)
        return

    if event_type == "user":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                label = "ERROR" if block.get("is_error") else "OK"
                print(f"\n[tool_result] {label}\n", flush=True)
        return

    if event_type == "result":
        print(
            f"\n[result] subtype={event.get('subtype','')} "
            f"turns={event.get('num_turns')} "
            f"duration_ms={event.get('duration_ms')}\n",
            flush=True,
        )
        result_text = event.get("result")
        if result_text:
            print(result_text, end="", flush=True)
            collected_text.append(result_text)
        return


def summarize_tool_input(tool_name, tool_input, max_len=180):
    if not isinstance(tool_input, dict):
        text = str(tool_input)
        return text[:max_len]

    if tool_name in {"Read", "Write", "Edit", "MultiEdit"}:
        path = tool_input.get("file_path") or tool_input.get("path")
        return f"path={path}"

    if tool_name == "Bash":
        cmd = " ".join(str(tool_input.get("command", "")).split())
        return f"cmd={cmd[:max_len]}"

    if tool_name == "Glob":
        return f"path={tool_input.get('path')} pattern={tool_input.get('pattern')}"

    if tool_name == "Grep":
        return f"path={tool_input.get('path')} pattern={tool_input.get('pattern')}"

    if tool_name == "WebSearch":
        query = tool_input.get("query", "")
        return f"query={query[:max_len]}"

    if tool_name == "WebFetch":
        return f"url={tool_input.get('url', '')}"

    if tool_name == "Skill":
        return f"skill={tool_input.get('skill') or tool_input.get('name')}"

    text = " ".join(str(tool_input).split())
    return text[:max_len]


def ensure_dirs(config):
    for key in ("log_dir", "status_dir"):
        d = config[key]
        d.mkdir(parents=True, exist_ok=True)
    config["progress_file"].parent.mkdir(parents=True, exist_ok=True)
    for key in ("experiment_dir", "idea_dir", "knowledge_dir"):
        d = config[key]
        d.mkdir(parents=True, exist_ok=True)


def load_titles(paper_base_dir):
    if not paper_base_dir.exists():
        raise FileNotFoundError(f"Paper base dir not found: {paper_base_dir}")
    titles = sorted(
        p.name for p in paper_base_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not titles:
        raise ValueError(f"No paper directories found in: {paper_base_dir}")
    return titles


def load_progress(progress_file):
    if not progress_file.exists():
        return {"done": [], "failed": [], "runs": {}, "last_updated": None}
    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RuntimeError(
            f"progress.json is corrupted: {progress_file}\n"
            f"Please inspect or delete it before rerunning."
        )


def save_progress(progress, progress_file):
    tmp = progress_file.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(progress_file)


def safe_name(index, title, max_len=90):
    cleaned = []
    for c in title:
        if c.isalnum() or c in "-_.":
            cleaned.append(c)
        else:
            cleaned.append("_")
    name = "".join(cleaned)
    name = "_".join(part for part in name.split("_") if part)
    if len(name) > max_len:
        name = name[:max_len]
    return f"{index:03d}__{name}"


def make_prompt(index, total, title, config):
    paper_dir = config["paper_base_dir"] / title
    return f"""
你现在只处理第 {index}/{total} 篇论文。

论文标题：
{title}

路径参数（调用 skill 时必须传入）：

- paper_dir: {paper_dir}
- experiment_dir: {config['experiment_dir']}
- idea_dir: {config['idea_dir']}
- knowledge_dir: {config['knowledge_dir']}

请在当前这一个 Claude Code context 中，严格按顺序执行下面两个 skill：

第一步：
调用 paper-experiment-idea skill，传入以下参数：paper_dir={paper_dir}、experiment_dir={config['experiment_dir']}、idea_dir={config['idea_dir']}，论文标题为 "{title}"

第二步：
调用 paper-knowledge skill，传入以下参数：paper_dir={paper_dir}、knowledge_dir={config['knowledge_dir']}，论文标题为 "{title}"

硬性要求：
1. 每篇论文只使用这一个 context。
2. 必须先完整执行 paper-experiment-idea。
3. 等 paper-experiment-idea 完成后，才执行 paper-knowledge。
4. 不要并行执行两个 skill。
5. 不要为两个 skill 额外指定输出格式、输出文件、分析维度或额外约束。
6. 两个 skill 的输出规则已经在 skill 内部定义，严格遵守 skill 自身要求即可。
7. 当前 prompt 只负责指定论文标题、调用顺序、路径参数和状态记录。
8. 不要读取或引用其他论文的分析结果。
9. 如果第一步失败，不要继续第二步。
10. 最后请只在终端输出一个状态块，用于调度器记录执行状态。

状态块格式必须如下：

RUN_ONE_STATUS_BEGIN
paper_index: {index}
paper_total: {total}
paper_title: {title}
paper-experiment-idea: DONE 或 FAILED
paper-knowledge: DONE 或 FAILED 或 SKIPPED
overall: DONE 或 FAILED
RUN_ONE_STATUS_END
"""


def parse_status_block(stdout_text):
    begin = "RUN_ONE_STATUS_BEGIN"
    end = "RUN_ONE_STATUS_END"
    if begin not in stdout_text or end not in stdout_text:
        return {
            "status_block_found": False,
            "raw_status_block": None,
            "overall": "UNKNOWN",
            "paper-experiment-idea": "UNKNOWN",
            "paper-knowledge": "UNKNOWN",
        }
    block = stdout_text.split(begin, 1)[1].split(end, 1)[0].strip()
    parsed = {"status_block_found": True, "raw_status_block": block}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


# ══════════════════════════════════════════════════════════════════════════
# Per-paper runner
# ══════════════════════════════════════════════════════════════════════════

def run_one(index, total, title, config):
    base_name = safe_name(index, title)
    log_file = config["log_dir"] / f"{base_name}.stream.jsonl"
    status_file = config["status_dir"] / f"{base_name}.json"

    prompt = make_prompt(index, total, title, config)

    cmd = [
        CLAUDE_CMD,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--permission-mode", "acceptEdits",
        "--allowedTools",
        "Read,Glob,Grep,LS,Bash,Write,Edit,MultiEdit,WebSearch,WebFetch",
    ]
    if MODEL_NAME:
        cmd.extend(["--model", MODEL_NAME])

    started_at = datetime.now().isoformat()

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    collected_text = []
    with open(log_file, "w", encoding="utf-8") as log_fp:
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                collected_text.append(line)
                continue
            render_claude_stream_event(event, collected_text)

    returncode = proc.wait()
    ended_at = datetime.now().isoformat()
    stdout_text = "".join(collected_text)
    parsed_status = parse_status_block(stdout_text)

    status = {
        "paper_index": index,
        "paper_total": total,
        "paper_title": title,
        "started_at": started_at,
        "ended_at": ended_at,
        "returncode": returncode,
        "log_file": str(log_file),
        "status_block": parsed_status,
        "process_status": "DONE" if returncode == 0 else "FAILED",
    }

    status_file.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if returncode != 0:
        raise RuntimeError(
            f"Claude Code failed on paper {index}: {title}\n"
            f"Return code: {returncode}\n"
            f"See log: {log_file}\n"
            f"See status: {status_file}"
        )

    if parsed_status.get("overall") == "FAILED":
        raise RuntimeError(
            f"Skill-level status reported FAILED on paper {index}: {title}\n"
            f"See log: {log_file}\n"
            f"See status: {status_file}"
        )

    return status


# ══════════════════════════════════════════════════════════════════════════
# Per-config worker  (runs in a child process)
# ══════════════════════════════════════════════════════════════════════════

def run_config(config):
    """Process every paper under *one* config sequentially.  Called by the pool."""
    ensure_dirs(config)

    paper_base_dir = config["paper_base_dir"]
    progress_file = config["progress_file"]

    titles = load_titles(paper_base_dir)
    total = len(titles)

    progress = load_progress(progress_file)
    done_set = set(progress.get("done", []))

    config_name = config["name"]
    print(f"[{config_name}] Total papers: {total}", flush=True)

    for index, title in enumerate(titles, start=1):
        key = f"{index:03d}:{title}"
        if key in done_set:
            print(f"[{config_name}] SKIP {index}/{total}: {title}", flush=True)
            continue

        try:
            status = run_one(index, total, title, config)

            progress.setdefault("done", []).append(key)
            # Clean up any stale failed records for this paper (from prior retries)
            progress["failed"] = [
                f for f in progress.get("failed", []) if f["paper"] != key
            ]
            progress.setdefault("runs", {})[key] = status
            progress["last_updated"] = datetime.now().isoformat()
            save_progress(progress, progress_file)

            print(f"[{config_name}] DONE {index}/{total}: {title}\n", flush=True)

        except Exception as e:
            failed_record = {
                "paper": key,
                "error": str(e),
                "time": datetime.now().isoformat(),
            }
            progress.setdefault("failed", []).append(failed_record)
            progress["last_updated"] = datetime.now().isoformat()
            save_progress(progress, progress_file)

            print(f"[{config_name}] FAILED {index}/{total}: {title}", flush=True)
            print(str(e), flush=True)

            if STOP_ON_FAILURE:
                print(f"[{config_name}] STOP_ON_FAILURE=True, stopping.", flush=True)
                break
            else:
                continue

    # Summary
    done_count = len(progress.get("done", []))
    failed_count = len(progress.get("failed", []))
    return {
        "config": config_name,
        "total": total,
        "done": done_count,
        "failed": failed_count,
    }


# ══════════════════════════════════════════════════════════════════════════
# Master launcher
# ══════════════════════════════════════════════════════════════════════════

def main():
    if not CONFIGS:
        print("No configs defined. Add entries to CONFIGS list.", file=sys.stderr)
        sys.exit(1)

    workers = min(MAX_WORKERS, len(CONFIGS))
    print(f"Launching {workers} parallel worker(s) for {len(CONFIGS)} config(s)")
    print(f"STOP_ON_FAILURE: {STOP_ON_FAILURE}")
    print(f"Root: {ROOT}")
    print("-" * 60)

    for i, cfg in enumerate(CONFIGS):
        print(f"  [{i}] {cfg['name']}: {cfg['paper_base_dir']}")
    print("-" * 60)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_config, cfg): cfg["name"] for cfg in CONFIGS
        }

        for future in as_completed(future_map):
            name = future_map[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[MASTER] CONFIG DONE  {name}: "
                    f"{result['done']}/{result['total']} done, "
                    f"{result['failed']} failed",
                    flush=True,
                )
            except Exception:
                print(f"[MASTER] CONFIG CRASHED  {name}:", flush=True)
                traceback.print_exc()
                results.append({"config": name, "total": "?", "done": "?", "failed": "CRASHED"})

    print("\n" + "=" * 60)
    print("ALL CONFIGS PROCESSED")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['config']:30s}  done={str(r['done']):>4s}  "
            f"failed={str(r['failed']):>7s}  total={str(r['total']):>4s}"
        )


if __name__ == "__main__":
    main()
