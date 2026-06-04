# Infera Runtime 编译-调度协同讲解

- 导出时间：2026-06-04 11:20 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：human_notes/Infera Runtime 编译-调度协同讲解.md
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
| 编译-调度协同 | Infera Runtime (SelectKernels→FuseKernels→LaunchKernel)* | SelectKernels: G(u) asynchrony wavefront metric递归传播children并发收益→选zero in-degree + max G(u) data blocks + 回归模型选最优kernel版本（IPC_est, TLP≥4约束）；FuseKernels: SASS binary level warp级水平融合（BAR.SYNC重组+warp交错排布）；LaunchKernel: HKQ→GDRCopy→DKQ→CDP daemon kernel fire-and-forget (<10μs) | **最深度的硬件-软件协同dispatcher——绕过DMA (GDRCopy)、GPU内自主launch (CDP)、SASS级binary manipulation** | Q2.4, vault: Infera Runtime (59.0/95.5/117.3) |
## Open tabs:
- review_draft.md: draft/review_draft.md
- L6_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L6_horizon_summary.md
- L5_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L5_horizon_summary.md
- L4_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L4_horizon_summary.md
- L3_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L3_horizon_summary.md

## My request for Codex:
教我:'| 编译-调度协同 | Infera Runtime (SelectKernels→FuseKernels→LaunchKernel)* | SelectKernels: G(u) asynchrony wavefront metric递归传播children并发收益→选zero in-degree + max G(u) data blocks + 回归模型选最优kernel版本（IPC_est, TLP≥4约束）；FuseKernels: SASS binary level warp级水平融合（BAR.SYNC重组+warp交错排布）；LaunchKernel: HKQ→GDRCopy→DKQ→CDP daemon kernel fire-and-forget (<10μs) | **最深度的硬件-软件协同dispatcher——绕过DMA (GDRCopy)、GPU内自主launch (CDP)、SASS级binary manipulation** | Q2.4, vault: Infera Runtime (59.0/95.5/117.3) |'

### 003 Codex Final
这行可以读成一句话：

**Infera Runtime 是一个把“编译器生成的 kernel 变体”和“运行时调度决策”深度绑在一起的 dispatcher：先选该跑谁，再把能融合的 kernel 在 SASS/warp 层融合，最后用低延迟 GPU-side launch 路径发射。**

**1. SelectKernels：决定“现在跑哪个 kernel、用哪个版本”**
这里把计算图看成 DAG。`zero in-degree` 是当前依赖都满足、已经可以执行的节点。  
`G(u) asynchrony wavefront metric` 可以理解为：如果现在调度节点 `u`，它会释放多少后续并发机会。

一个简化模型是：

```text
G(u) = local_gain(u) + sum(child_gain(v) for v in children(u))
```

也就是说，调度器不只看当前 kernel 快不快，还看它跑完以后能不能让更多 children 同时变 ready。  
所以它会优先选：

```text
ready_nodes = nodes with zero in-degree
pick u with max G(u)
pick best kernel version by regression model
subject to IPC_est high enough and TLP >= 4
```

`IPC_est` 是预测指令吞吐，`TLP >= 4` 是保证线程级并行度别太低，否则 kernel 版本虽然单看省资源，但可能填不满 GPU。

**2. FuseKernels：把多个 kernel 在 SASS / warp 层融合**
这里的厉害点是它不是普通 CUDA source-level fusion，而是更底层的 **SASS binary level** 操作。  
SASS 是 NVIDIA GPU 最底层汇编级指令。

`warp级水平融合` 可以理解为：不是把两个算子粗暴串起来，而是把不同 kernel 的 warp 重新排布，让它们在同一个 fused kernel 里交错执行。

关键难点是 `BAR.SYNC`。  
`BAR.SYNC` 是线程块内同步屏障。融合后，如果屏障位置不对，不同 warp 会互相等死或者破坏原来的同步语义，所以需要重组同步点。

简单时间线：

```text
Before:
K1: warp0 warp1 | BAR.SYNC | warp2
K2: warp0 warp1 | BAR.SYNC | warp2

After fused:
FusedK:
  K1.warp0
  K2.warp0
  K1.warp1
  K2.warp1
  adjusted BAR.SYNC
  K1.warp2
  K2.warp2
```

它的目标是减少 kernel launch 次数、提升 occupancy、把空闲 warp 槽位填起来。

**3. LaunchKernel：用低延迟路径发射 kernel**
`HKQ → GDRCopy → DKQ → CDP daemon kernel fire-and-forget`

可以这样拆：

- `HKQ`: host kernel queue，CPU 侧准备好的 kernel 队列。
- `GDRCopy`: 低延迟 CPU-GPU 内存拷贝路径，常用于绕开普通 `cudaMemcpy` / runtime 调用链的高开销。严格说不是“没有数据移动”，而是避开常规 DMA/runtime 路径的调度开销。
- `DKQ`: device kernel queue，GPU 侧能看到的 kernel 队列。
- `CDP daemon kernel`: GPU 上常驻的 daemon kernel，用 CUDA Dynamic Parallelism 在 GPU 内部继续 launch 子 kernel。
- `fire-and-forget`: CPU 把任务塞进去后不用每次同步等待，GPU 自己接着发射。

所以 `<10μs` 说的是 launch latency 被压得很低。

**整行的核心意思**
传统 dispatcher 多半停留在“CPU runtime 根据依赖图发 kernel”。  
这里的 Infera Runtime 更激进：它同时做 **图调度、kernel 版本选择、二进制级 kernel 融合、低延迟 GPU-side launch**。

你表格里那句“最深度的硬件-软件协同 dispatcher”强调的正是这个：它不是单纯调度算法，而是把调度器一路下探到 GPU binary、warp、barrier、launch queue 和 CDP 机制。

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
