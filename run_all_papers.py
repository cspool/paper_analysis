#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
from pathlib import Path
from datetime import datetime


# =========================
# Basic config
# =========================

ROOT = Path("/data3/paper_analysis").resolve()

TITLE_FILE = ROOT / "papers" / "paper_titles_2025.md"

LOG_DIR = ROOT / "outputs" / "logs"
STATUS_DIR = ROOT / "outputs" / "status"
PROGRESS = ROOT / "outputs" / "progress.json"

CLAUDE_CMD = "claude"

STOP_ON_FAILURE = False
# True:
#   某篇论文失败后停止，不继续下一篇。
# False:
#   某篇论文失败后记录状态，继续下一篇。

# 路径参数 —— 传递给 paper-experiment-idea 和 paper-knowledge skill
PAPER_DIR = ROOT / "papers" / "paper_2025"
EXPERIMENT_DIR = ROOT / "repo_2025" / "experiment_repo"
IDEA_DIR = ROOT / "repo_2025" / "idea_repo"
KNOWLEDGE_DIR = ROOT / "repo_2025" / "knowledge_repo"

MODEL_NAME = None
# 如果你的 deepseekv4pro 已经在 Claude Code 环境中配置好，这里保持 None。
# 如果你需要显式指定模型，可以改成：
# MODEL_NAME = "deepseekv4pro"


# =========================
# Utility functions
# =========================

def render_claude_stream_event(event, collected_text):
    """
    轻量前台可视化：
    - 显示 Claude 正文
    - 显示工具调用
    - 不显示 tool_result 详细内容
    - 不显示大量 system/user/event 噪音
    """

    event_type = event.get("type")

    # 初始化信息：只显示一次关键信息
    if event_type == "system":
        subtype = event.get("subtype")

        if subtype == "init":
            print(
                f"\n[init] "
                f"model={event.get('model')} "
                f"session_id={event.get('session_id')}\n",
                flush=True
            )

        return

    # 流式文本增量
    if event_type == "stream_event":
        se = event.get("event", {})
        se_type = se.get("type")

        if se_type == "content_block_delta":
            delta = se.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    print(text, end="", flush=True)
                    collected_text.append(text)

        return

    # assistant 完整消息：正文 + tool_use
    if event_type == "assistant":
        msg = event.get("message", {})
        content = msg.get("content", [])

        for block in content:
            block_type = block.get("type")

            # Claude 正文
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    print(text, end="", flush=True)
                    collected_text.append(text)

            # 工具调用：只显示工具名和简短输入
            elif block_type == "tool_use":
                name = block.get("name", "unknown_tool")
                tool_input = block.get("input", {})

                short_input = summarize_tool_input(name, tool_input)

                print(
                    f"\n[tool_use] {name} {short_input}\n",
                    flush=True
                )

        return

    # user 事件通常是 tool_result，不详细打印
    if event_type == "user":
        msg = event.get("message", {})
        content = msg.get("content", [])

        for block in content:
            if block.get("type") == "tool_result":
                is_error = block.get("is_error", False)

                if is_error:
                    print("\n[tool_result] ERROR\n", flush=True)
                else:
                    print("\n[tool_result] OK\n", flush=True)

        return

    # 最终结果
    if event_type == "result":
        subtype = event.get("subtype", "")
        duration_ms = event.get("duration_ms")
        num_turns = event.get("num_turns")

        print(
            f"\n[result] subtype={subtype} "
            f"turns={num_turns} "
            f"duration_ms={duration_ms}\n",
            flush=True
        )

        result_text = event.get("result")
        if result_text:
            print(result_text, end="", flush=True)
            collected_text.append(result_text)

        return

def summarize_tool_input(tool_name, tool_input, max_len=180):
    """
    把 tool_use 的 input 压缩成一行，避免前台刷屏。
    """

    if not isinstance(tool_input, dict):
        text = str(tool_input)
        return text[:max_len]

    # Read
    if tool_name == "Read":
        path = tool_input.get("file_path") or tool_input.get("path")
        return f"path={path}"

    # Write
    if tool_name == "Write":
        path = tool_input.get("file_path") or tool_input.get("path")
        return f"path={path}"

    # Edit / MultiEdit
    if tool_name in {"Edit", "MultiEdit"}:
        path = tool_input.get("file_path") or tool_input.get("path")
        return f"path={path}"

    # Bash
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        cmd = " ".join(str(cmd).split())
        if len(cmd) > max_len:
            cmd = cmd[:max_len] + "..."
        return f"cmd={cmd}"

    # Glob
    if tool_name == "Glob":
        pattern = tool_input.get("pattern")
        path = tool_input.get("path")
        return f"path={path} pattern={pattern}"

    # Grep
    if tool_name == "Grep":
        pattern = tool_input.get("pattern")
        path = tool_input.get("path")
        return f"path={path} pattern={pattern}"

    # WebSearch
    if tool_name == "WebSearch":
        query = tool_input.get("query", "")
        if len(query) > max_len:
            query = query[:max_len] + "..."
        return f"query={query}"

    # WebFetch
    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        return f"url={url}"

    # Skill
    if tool_name == "Skill":
        skill_name = tool_input.get("skill") or tool_input.get("name")
        return f"skill={skill_name}"

    # Task
    if tool_name == "Task":
        description = tool_input.get("description", "")
        prompt = tool_input.get("prompt", "")
        text = description or prompt
        text = " ".join(str(text).split())
        if len(text) > max_len:
            text = text[:max_len] + "..."
        return f"{text}"

    # fallback
    text = str(tool_input)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text

def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)


def load_titles():
    if not TITLE_FILE.exists():
        raise FileNotFoundError(f"Title file not found: {TITLE_FILE}")

    titles = []
    for line in TITLE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        titles.append(line)

    if not titles:
        raise ValueError(f"No paper titles found in: {TITLE_FILE}")

    return titles


def load_progress():
    if not PROGRESS.exists():
        return {
            "done": [],
            "failed": [],
            "runs": {},
            "last_updated": None
        }

    try:
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RuntimeError(
            f"progress.json is corrupted: {PROGRESS}\n"
            f"Please inspect or delete it before rerunning."
        )


def save_progress(progress):
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(PROGRESS)


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


def make_prompt(index, total, title):
    return f"""
你现在只处理第 {index}/{total} 篇论文。

论文标题：
{title}

路径参数（调用 skill 时必须传入）：

- paper_dir: {PAPER_DIR}
- experiment_dir: {EXPERIMENT_DIR}
- idea_dir: {IDEA_DIR}
- knowledge_dir: {KNOWLEDGE_DIR}

请在当前这一个 Claude Code context 中，严格按顺序执行下面两个 skill：

第一步：
调用 paper-experiment-idea skill，传入以下参数：paper_dir={PAPER_DIR}、experiment_dir={EXPERIMENT_DIR}、idea_dir={IDEA_DIR}，论文标题为 "{title}"

第二步：
调用 paper-knowledge skill，传入以下参数：paper_dir={PAPER_DIR}、knowledge_dir={KNOWLEDGE_DIR}，论文标题为 "{title}"

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
    """
    解析 Claude Code 最后输出的状态块。
    注意：这是模型报告的状态，不替代 subprocess returncode。
    subprocess returncode 仍然是进程级成功/失败依据。
    """
    begin = "RUN_ONE_STATUS_BEGIN"
    end = "RUN_ONE_STATUS_END"

    if begin not in stdout_text or end not in stdout_text:
        return {
            "status_block_found": False,
            "raw_status_block": None,
            "overall": "UNKNOWN",
            "paper-experiment-idea": "UNKNOWN",
            "paper-knowledge": "UNKNOWN"
        }

    block = stdout_text.split(begin, 1)[1].split(end, 1)[0].strip()

    parsed = {
        "status_block_found": True,
        "raw_status_block": block
    }

    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()

    return parsed


# =========================
# Core runner
# =========================

def run_one(index, total, title):
    base_name = safe_name(index, title)
    log_file = LOG_DIR / f"{base_name}.stream.jsonl"
    status_file = STATUS_DIR / f"{base_name}.json"

    prompt = make_prompt(index, total, title)

    cmd = [
        CLAUDE_CMD,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Read,Glob,Grep,LS,Bash,Write,Edit,MultiEdit,WebSearch,WebFetch"
    ]

    if MODEL_NAME:
        cmd.extend(["--model", MODEL_NAME])

    started_at = datetime.now().isoformat()

    print("=" * 80)
    print(f"RUN {index}/{total}")
    print(f"TITLE: {title}")
    print("LIVE MODE: stream-json")
    print(f"STEP 1: /paper-experiment-idea")
    print(f"STEP 2: /paper-knowledge")
    print(f"LOG: {log_file}")
    print(f"STATUS: {status_file}")
    print("=" * 80)

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )

    collected_text = []

    with open(log_file, "w", encoding="utf-8") as log_fp:
        for line in process.stdout:
            # 完整 JSONL 仍然保存到 log，方便之后排查
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

    returncode = process.wait()
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
        "process_status": "DONE" if returncode == 0 else "FAILED"
    }

    status_file.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8"
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


# =========================
# Main loop
# =========================

def main():
    ensure_dirs()

    titles = load_titles()
    total = len(titles)

    progress = load_progress()
    done_set = set(progress.get("done", []))

    print(f"Project root: {ROOT}")
    print(f"Total papers: {total}")
    print(f"Execution mode: single scheduler, strict sequential run_one")
    print(f"Per paper: one context, two skills sequentially")
    print(f"STOP_ON_FAILURE: {STOP_ON_FAILURE}")
    print()

    for index, title in enumerate(titles, start=1):
        key = f"{index:03d}:{title}"

        if key in done_set:
            print(f"SKIP {index}/{total}: {title}")
            continue

        try:
            # 关键顺序点：
            # run_one 返回前，不会进入下一篇论文。
            status = run_one(index, total, title)

            progress.setdefault("done", []).append(key)
            progress.setdefault("runs", {})[key] = status
            progress["last_updated"] = datetime.now().isoformat()
            save_progress(progress)

            print(f"DONE {index}/{total}: {title}")
            print()

            if index % 10 == 0:
                print("-" * 80)
                print(f"BATCH CHECKPOINT: finished up to paper {index}/{total}")
                print("-" * 80)

        except Exception as e:
            failed_record = {
                "paper": key,
                "error": str(e),
                "time": datetime.now().isoformat()
            }

            progress.setdefault("failed", []).append(failed_record)
            progress["last_updated"] = datetime.now().isoformat()
            save_progress(progress)

            print(f"FAILED {index}/{total}: {title}")
            print(str(e))
            print()

            if STOP_ON_FAILURE:
                print("STOP_ON_FAILURE=True, stop processing remaining papers.")
                break
            else:
                print("STOP_ON_FAILURE=False, continue to next paper.")
                continue

    print("All possible papers processed under current configuration.")


if __name__ == "__main__":
    main()