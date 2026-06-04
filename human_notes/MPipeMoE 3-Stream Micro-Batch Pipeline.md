# MPipeMoE 3-Stream Micro-Batch Pipeline

- 导出时间：2026-06-04 11:32 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：/data3/paper_analysis/human_notes/MPipeMoE 3-Stream Micro-Batch Pipeline.md
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
<environment_context>
  <cwd>/data3/paper_analysis</cwd>
  <shell>bash</shell>
  <current_date>2026-06-04</current_date>
  <timezone>Asia/Shanghai</timezone>
  <filesystem><workspace_roots><root>/data3/paper_analysis</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry><entry access="write"><path>/data3/paper_analysis</path></entry><entry access="write"><special>:slash_tmp</special></entry><entry access="write"><special>:tmpdir</special></entry><entry access="read"><path>/data3/paper_analysis/.git</path></entry><entry access="read"><path>/data3/paper_analysis/.agents</path></entry><entry access="read"><path>/data3/paper_analysis/.codex</path></entry></file_system></permission_profile></filesystem>
</environment_context>

### 002 User
# Context from my IDE setup:

## Active file: draft/review_draft.md

## Active selection of the file:
| 微批次流水线 | MPipeMoE 3-Stream Micro-Batch Pipeline* | **3 CUDA stream (comp/comm/mem)**；offline interference profiling (μ/σ/η slowdown因子)；4种memory reuse策略 (S1-S4) + n∈{1,2,4,8,16}自适应选择→min T_total | σ≈1（计算几乎不受干扰→通信-计算重叠可行）；N大(64 GPU): S4最优（避免memory bandwidth竞争） | Q2.4, vault: MPipeMoE (56.3) |
## Open tabs:
- review_draft.md: draft/review_draft.md
- L6_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L6_horizon_summary.md
- L5_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L5_horizon_summary.md
- L4_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L4_horizon_summary.md
- L3_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L3_horizon_summary.md

## My request for Codex:
教我:'| 微批次流水线 | MPipeMoE 3-Stream Micro-Batch Pipeline* | **3 CUDA stream (comp/comm/mem)**；offline interference profiling (μ/σ/η slowdown因子)；4种memory reuse策略 (S1-S4) + n∈{1,2,4,8,16}自适应选择→min T_total | σ≈1（计算几乎不受干扰→通信-计算重叠可行）；N大(64 GPU): S4最优（避免memory bandwidth竞争） | Q2.4, vault: MPipeMoE (56.3) |'

### 003 Codex Final
这行的核心意思是：

MPipeMoE 把一个 MoE layer 的训练过程拆成很多 `micro-batch`，然后用 3 条 CUDA stream 把“通信、计算、内存拷贝”尽量并行起来；再根据离线 profiling 得到的干扰系数，自动选择最合适的流水线粒度 `n` 和内存复用策略 `S1-S4`，目标是最小化总训练时间 `T_total`。

**先抓住一张图**

```text
原始 MoE 层:
All-to-All Dispatch S  ->  Expert FFN C  ->  All-to-All Collect R
        通信                    计算                  通信

MPipeMoE:
把 batch 切成 n 个 micro-batch:
mb0, mb1, mb2, ...

comm stream:  S0 ---- R0 ---- S1 ---- R1 ---- S2 ---- R2
comp stream:       C0 ---- C1 ---- C2 ---- C3
mem  stream:  D2H/H2D/offload/prefetch, 仅部分策略需要
```

Annotations:

`S` 是 token dispatch 的 All-to-All 通信，`C` 是 expert FFN 计算，`R` 是 token collect 的 All-to-All 通信。`comm stream` 跑 NCCL 通信，`comp stream` 跑 GEMM/FFN，`mem stream` 跑 CPU-GPU 拷贝。核心是让 `S/R` 和 `C` 重叠，而不是串行等完。

**这几个符号怎么读**

`n ∈ {1,2,4,8,16}`：把一个 batch 切成几个 micro-batch。  
`n` 太小，重叠不充分；`n` 太大，kernel launch overhead 和 GPU under-utilization 增加。论文/笔记里常见结论是 batch 越大，最优 `n` 越大。

`μ / σ / η`：三类操作并行时的 slowdown 因子。  
`μ` 描述通信受干扰，`σ` 描述计算受干扰，`η` 描述 memory copy 受干扰。

```text
单独执行:
W_comp, W_comm, W_mem

并行执行:
T_comp ≈ work_comp / (σ * W_comp)
T_comm ≈ work_comm / (μ * W_comm)
T_mem  ≈ work_mem  / (η * W_mem)
```

Annotations:

如果 `σ≈1`，说明计算几乎不被通信或拷贝拖慢，所以“通信-计算重叠”很值得做。如果 `η` 或通信和 mem copy 相关 slowdown 很差，说明通信和 CPU-GPU 拷贝在抢 memory bandwidth，这时三流并行反而可能拖慢。

**S1-S4 是什么**

它们都是 memory reuse 策略。MPipeMoE 发现不同 micro-batch 的中间张量生命周期错开，所以可以共用 buffer，降低 activation/temp buffer 内存。但 buffer 被覆盖后，反向传播还要用这些张量，所以要“恢复”。

| 策略 | `T_DI` 怎么恢复 | `T_M` 怎么恢复 | 直觉 |
|---|---|---|---|
| S1 | CPU offload | CPU offload | 多用 mem copy |
| S2 | 通信重放 | CPU offload | 通信 + mem copy |
| S3 | CPU offload | 重计算 | mem copy + compute |
| S4 | 通信重放 | 重计算 | 避免前向 mem copy |

所以你表里说：

`N大(64 GPU): S4最优（避免memory bandwidth竞争）`

意思是大规模 GPU 下通信压力已经很大，如果还让 `mem stream` 做大量 D2H/H2D 拷贝，会和通信抢带宽。S4 不靠 CPU offload 恢复，而是用“重新 All-to-All + 重计算”恢复张量，因此避开 memory copy 带宽竞争；重计算开销在通信瓶颈下还能被隐藏一部分。

**选择逻辑**

```pseudo
best = None
for S in {S1, S2, S3, S4}:
  for n in {1, 2, 4, 8, 16}:
    b = B / n
    T = performance_model(B=b, strategy=S, slowdown=(μ, σ, η))
    best = min(best, (T, n, S))
return best
```

Annotations:

`B` 是 batch tokens，`b` 是每个 micro-batch 的大小。模型估算每种 `(n, S)` 的端到端时间，选 `T_total` 最小的组合。这里 `N` 是设备规模，`n` 是流水线粒度，两个不要混。

**一句话背下来**

MPipeMoE = “把 MoE 的 All-to-All 和 expert FFN 切成 micro-batch，用 CUDA 多 stream 重叠；再用 slowdown profiling 判断哪些重叠值得做，最后在 `n` 和 `S1-S4` 里选最快且省内存的配置。”

证据来自本地 vault：  
[CUDA Stream Overlapping](/data3/paper_analysis/knowledge_notes/kernel知识笔记/CUDA%20Stream%20Overlapping%20for%20Computation-Communication%20Overlap.md) score 2225.0；[Pipeline Performance Model](/data3/paper_analysis/knowledge_notes/kernel知识笔记/Pipeline%20Paradigm%20and%20Performance%20Model%20for%20Multi-Stream%20GPU%20Execution.md) score 6931.0；[MPipeMoE kernel note](/data3/paper_analysis/experiment_notes/kernel实验笔记/MPipeMoE_%20Memory%20Efficient%20MoE%20for%20Pre-trained%20Models%20with%20Adaptive%20Pipeline%20Parallelism.md) score 1789.5。

### 004 User
# Context from my IDE setup:

## Active file: draft/review_draft.md

## Open tabs:
- review_draft.md: draft/review_draft.md
- L6_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L6_horizon_summary.md
- L5_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L5_horizon_summary.md
- L4_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L4_horizon_summary.md
- L3_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L3_horizon_summary.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 

### 005 User
<skill>
<name>export-conversation-notes</name>
<path>/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md</path>
---
name: export-conversation-notes
description: Export the currently available Codex/chat session record into a target Markdown file, defaulting to human_notes/. Save only user inputs and Codex final outputs. Use when the user asks to save, archive, output, or record the current conversation, dialogue history, or analysis session in the paper_analysis workspace. If the target Markdown file already exists and is non-empty, append incrementally without modifying, deleting, reformatting, or replacing any existing content.
---

# Export Conversation Notes

## Overview

Save the current session's available user inputs and Codex final outputs to the requested target location. Default to `human_notes/` when no explicit directory is provided. Default to Chinese for headings and status text. Do not save process notes, status updates, tool calls, command outputs, hidden instructions, or intermediate work logs.

## Workflow

1. Identify the target path.
   - save under `human_notes/`.
   - If no filename can be identified, use `<session-name>.md` and <session-name> is the name of the current session.

2. Prepare the target file.
   - Work relative to the current workspace root.
   - Ensure the destination directory exists.
   - Sanitize only the filename portion by replacing filesystem-hostile characters such as `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, and `|` with safe separators, then trim whitespace.
   - Save to the resolved target path.

3. Capture the available conversation record.
   - Output only user inputs and Codex final outputs in chronological order with speaker labels.
   - Include all available user messages exactly as written.
   - Include only Codex final responses that answer the user, not interim progress updates or tool-facing content.
   - Do not include tool calls, shell commands, command outputs, file edit logs, errors from tools, status updates, planning chatter, hidden system/developer/policy/runtime instructions, or intermediate reasoning.
   - Do not summarize, compress, paraphrase, normalize, or reorganize the saved user inputs and final outputs.
   - If earlier user inputs or final outputs are unavailable because context was compacted or not exposed to Codex, state this limitation briefly before the saved conversation.

4. Write the Markdown file.
   - If the file does not exist, create it with the new-file template.
   - If the file exists but is empty, write the new-file template.
   - If the file exists and is non-empty, enter incremental mode: append a new dated section only at the end of the file.
   - In incremental mode, never modify, delete, reorder, summarize, normalize, reformat, or replace any existing content, even if the existing note has typos, duplicate headings, stale metadata, or inconsistent formatting.
   - In incremental mode, use an append-only edit. With `apply_patch`, add only new lines after the existing final line.
   - Keep the saved record readable, but preserve the available user inputs and final outputs over brevity.
   - If the record is too long for one edit, append it in multiple consecutive chunks until all currently available user inputs and final outputs are saved.

## Markdown Template

For a new file, use this structure:

```md
# <session-or-paper-title>

- 导出时间：<YYYY-MM-DD HH:MM TZ>
- 来源：当前 Codex session 可用对话上下文
- 保存路径：<resolved-target-path>
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
<用户消息原文>

### 002 Codex Final
<Codex 最终回复原文>
```

For appending to an existing non-empty file, add this block at the end of the file without changing earlier content:

```md
---

## 对话记录补充：<YYYY-MM-DD HH:MM TZ>

<continue the same chronological format, saving only user inputs and Codex final outputs>
```

## Completion Response

After saving, respond briefly with the output path and whether the file was created or appended. Mention any uncertainty about target inference or incomplete available conversation context.

</skill>
