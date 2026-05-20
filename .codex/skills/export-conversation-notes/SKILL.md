---
name: export-conversation-notes
description: Export the current visible Codex/chat conversation into a Markdown note named from the current paper title and save it under conv_notes/. Use when the user asks to save, archive, output, or record the current conversation, dialogue history, or analysis session for a paper in the paper_analysis workspace. If the target Markdown file already exists and is non-empty, append incrementally without modifying, deleting, reformatting, or replacing any existing content.
---

# Export Conversation Notes

## Overview

Save the current visible conversation as a paper-specific Markdown note in `conv_notes/`. Default to Chinese for headings and status text, and preserve technical details such as paper filenames, paths, decisions, commands, errors, and generated artifacts.

## Workflow

1. Identify the current paper title.
   - Prefer a paper title or path explicitly named by the user.
   - If the IDE/context provides an active or obviously current paper file, use its basename without the extension, for example `papers/paper_2026/71-TokenFlow Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling.md` becomes `71-TokenFlow Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling`.
   - If several paper tabs are visible but none is clearly current, ask one concise clarification question before writing.
   - Do not invent a title. If no paper can be identified, use `未命名论文对话记录` only after stating the uncertainty.

2. Prepare the output path.
   - Work relative to the current workspace root.
   - Ensure `conv_notes/` exists.
   - Sanitize the filename by replacing filesystem-hostile characters such as `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, and `|` with safe separators, then trim whitespace.
   - Save to `conv_notes/<current-paper-title>.md`.

3. Capture the conversation record.
   - Use only the conversation content visible in the current context. If context was compacted or earlier turns are unavailable, say that the record covers the visible conversation.
   - Prefer a chronological speaker-labeled transcript when the conversation is short enough.
   - For long sessions, write a faithful structured record instead of a lossy one-line summary: user requests, Codex actions, important tool results, files changed, errors, decisions, and remaining questions.
   - Do not add paper claims, experiment details, or conclusions that were not present in the conversation.

4. Write the Markdown file.
   - If the file does not exist, create it with the new-file template.
   - If the file exists but is empty, write the new-file template.
   - If the file exists and is non-empty, enter incremental mode: append a new dated section only at the end of the file.
   - In incremental mode, never modify, delete, reorder, summarize, normalize, reformat, or replace any existing content, even if the existing note has typos, duplicate headings, stale metadata, or inconsistent formatting.
   - In incremental mode, use an append-only edit. With `apply_patch`, add only new lines after the existing final line.
   - Keep the note readable and directly useful as a future memory of the work session.

## Markdown Template

For a new file, use this structure:

```md
# <current-paper-title>

- 导出时间：<YYYY-MM-DD HH:MM TZ>
- 来源：当前 Codex 可见对话上下文
- 保存路径：conv_notes/<current-paper-title>.md

## 对话记录

### User
<用户消息或结构化摘要>

### Codex
<Codex 回复、执行动作、关键工具结果或结构化摘要>

## 已产生的文件或修改

- <路径或 N/A>

## 后续待办

- <待办或 N/A>
```

For appending to an existing non-empty file, add this block at the end of the file without changing earlier content:

```md
---

## 对话记录补充：<YYYY-MM-DD HH:MM TZ>

<same sections as needed>
```

## Completion Response

After saving, respond briefly with the output path and whether the file was created or appended. Mention any uncertainty about title inference or incomplete visible conversation context.
