---
name: paper-single-analysis
description: Analyze one academic paper from a user-provided paper title, title number, or local paper identifier by searching the current workspace for the corresponding full text, reading the core sections, verifying official code/artifact evidence when possible, extracting baseline defects and implementation details, and explaining any new pipeline or kernel execution flow in a Chinese dictionary-style Markdown report. Use when Codex is asked to analyze a single local paper, especially architecture, systems, hardware, accelerator, compiler/runtime, sparse/quantization, LLM acceleration, serving, or GPU kernel papers.
---

# Paper Single Analysis

## Overview

Use this skill to analyze exactly one paper from the current working directory. Default to Chinese output and write like a computer architecture, systems, or top-conference reviewer: focus on the paper's real problem, baseline limitation, design response, implementation evidence, experimental basis, and the concrete pipeline or kernel mechanism introduced by the paper.

## Workflow

1. Identify the target paper from the user's title, paper number, filename, or local identifier.
2. Search the current workspace first. Use `rg --files`, title fragments, paper numbers, and likely directories such as `paper_2026/`, `paper-fulltext-repo-analysis/`, or other local paper folders.
3. If the user gives a number, map it through local title lists such as `paper_titles.md` before searching for the full text.
4. Open the matching paper source, preferring full text files such as PDF, HTML, Markdown, text extraction, conference page snapshots, or downloaded paper pages.
5. Read the core sections: Abstract, Introduction / Background / Motivation, Method / Design / Architecture / Approach, and Implementation / Experiment / Evaluation.
6. Verify open-source repository or artifact status when evidence is available. Prefer links from the paper, official project page, conference artifact page, author/lab homepage, GitHub, Hugging Face, or Papers with Code. If external network or evidence is unavailable, state the uncertainty explicitly.
7. Write one complete Markdown file named from the paper title. Sanitize characters that are invalid for the local filesystem.

## Analysis Requirements

Ground every claim in the paper text or official artifact evidence. Do not infer implementation details, metrics, or repository status beyond the available sources.

For the core contribution and baseline comparison, answer:

- What specific bottleneck exists in prior methods, systems, hardware architectures, or baselines.
- Whether the bottleneck comes from compute, memory access, communication, scheduling, data layout, sparsity, precision, parallelism, energy, area, or another factor.
- Why the paper argues existing approaches are insufficient.
- Under what application scenario, model scale, hardware platform, workload, or system condition the issue becomes important.

For methodological design, answer:

- What the method's core mechanism is.
- How the design directly counters the baseline limitation.
- Whether it introduces new dataflow, hardware units, scheduling, compiler optimization, sparsity pattern, approximate computation, cache policy, training/inference strategy, or system co-design.
- What trade-offs the design accepts.

For implementation and evaluation, answer:

- How the baseline is implemented.
- How the proposed design is implemented.
- What hardware platform, simulator, framework, model, dataset, benchmark, kernel, runtime, RTL, FPGA/ASIC/GPU prototype, or artifact is used.
- What key parameters, assumptions, and limitations affect the experiments.
- How performance, energy, area, throughput, latency, accuracy, cost, or scalability gains are measured.

For pipeline or kernel extraction, answer:

- What new pipeline, kernel, operator packaging, execution flow, scheduling path, dataflow, or runtime path the paper introduces.
- How the new pipeline/kernel works in simple terms.
- How one concrete request, batch, tensor, token, tile, sparse block, KV cache block, or GPU kernel sequence flows through it.
- If the paper does not introduce a named pipeline or kernel, describe the closest concrete system path or write that no explicit new pipeline/kernel is provided.

## Output Format

Create exactly one Markdown file for the paper. Use this dictionary-style structure and avoid tables:

```md
论文标题：<Paper Title>

    开源仓库确认：
        - 状态：已找到 / 疑似相关 / 未找到明确开源仓库 / 无法确认
        - 链接：<URL 或 N/A>
        - 说明：<官方仓库、非官方复现、作者主页、artifact 页面等判断依据>

    1、论文工作：
        - 论文要解决的核心问题：<内容>
        - 论文的主要贡献：<内容>
        - 论文所处背景：<内容>

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：<内容>
        - 论文的设计方法：<内容>
        - 方法如何对冲 Baseline 缺陷：<内容>
        - 关键 trade-off：<内容>

    3、论文实现：
        - Baseline 如何实现：<内容>
        - 新设计如何实现：<内容>
        - 实验 / 实现平台：<内容>
        - 关键实验设置与指标：<内容>

    4、pipeline/kernel解析：
        - 新pipeline/kernel是什么：<内容>
        - 新pipeline/kernel的执行流例子：<内容>
```

## Quality Rules

- Analyze only one paper per invocation unless the user explicitly asks otherwise.
- Search local workspace content before using external sources.
- Do not rely only on title, abstract, snippets, or secondary summaries when full text is locally available.
- If full text, implementation details, experiments, or repository evidence are missing, write the uncertainty instead of inventing details.
- Preserve technical terms such as `Baseline`, `trade-off`, `Implementation`, and benchmark names when they improve clarity.
- Preserve concrete pipeline/kernel names from the paper. If no exact name exists, use a descriptive name and clearly mark it as an inferred description.
- Prefer concise but information-dense Chinese. Avoid broad praise, generic summaries, and unsupported claims.
