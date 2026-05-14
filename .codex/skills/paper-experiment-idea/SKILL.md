---
name: paper-experiment-idea
description: Analyze a local AI systems, LLM inference, accelerator, compiler, runtime, serving, or architecture paper from a title, filename, number, or workspace identifier; find the original paper PDF or paired full text in paper_2026, read implementation/evaluation and method sections, classify reproducibility-relevant experiments into the corresponding experiment_repo layer files, and append baseline-vs-method design ideas to idea_repo/idea库.md. Use when the user asks to build or update experiment and idea knowledge bases from a paper.
---

# Paper Experiment Idea

## Overview

Use this skill to turn one paper into two reusable Chinese knowledge-base entries:

- `experiment_repo/实验_<层次>.md`: implementation and experiment evidence for reproducibility.
- `idea_repo/idea库.md`: the paper's baseline pain point and method idea across the model-to-hardware stack.

Default to the current workspace. Search `paper_2026/` first for the original PDF and any paired extracted full text (`.txt`, `.md`). Use title lists, downloaded PDFs, extracted text files, and Markdown notes only to identify or support the target paper.

## Workflow

1. Identify the target paper from the user's title, title number, filename, or local identifier.
2. Search `paper_2026/` with `rg --files`, title fragments, paper numbers, and likely filename variants to find the original PDF. If a same-title `.txt` or `.md` extraction exists, use it as the readable full text while treating the PDF as the source paper.
3. Prefer readable full text in this order: paired extracted `.txt` / `.md` from `paper_2026/`, PDF text extraction from the original PDF, paper HTML, then other local notes. Do not rely only on abstract or secondary summaries when full text is available.
4. Read the sections most relevant to reproducibility: `Implementation`, `Experiment`, `Evaluation`, `Experimental Setup`, `Artifact`, `Method`, `Design`, `Architecture`, `Introduction`, `Motivation`, and `Background`.
5. Extract how the paper implements baselines, how it implements the proposed design, and what hardware, simulator, framework, models, datasets, or benchmarks it uses.
6. Classify the implementation and experiments into one or more layers below. Multiple layers are allowed.
7. Append experiment entries only to the matching layer Markdown files under `experiment_repo/`, and append the design-idea entry only to `idea_repo/idea库.md`. Create missing directories or files if needed. Do not overwrite existing entries.
8. Report any uncertain, missing, or approximate classification in the final response and, when useful, inside the appended entry under `分层说明`.

## Evidence Rules

- Ground claims in the paper text. If a detail is absent, write `论文未明确说明` instead of inventing it.
- Preserve concrete names for models, datasets, frameworks, simulators, benchmarks, chips, GPUs, RTL tools, kernels, and metrics.
- If the paper gives a simulator or artifact name but no link, search the paper text first for URLs. Use web search only when needed and allowed; otherwise write that the link could not be confirmed.
- Treat "baseline implementation" as the concrete version the paper evaluates, not merely the general prior-work idea.
- Keep the writing concise, technical, and useful for later reproduction.

## Experiment Layer Taxonomy

For every matched layer, append the paper to the corresponding file under `experiment_repo/` using the exact field order below. If the paper belongs to multiple layers, write one entry in each corresponding layer file. If the implementation only partially matches a layer, still use the closest layer and add a short `分层说明`.

### 算法pipeline

Use when the implementation accelerates inference with sparsity, compression, quantization, distillation, shared or approximate weights/activations, a new model pipeline, or algorithm-level model changes while trying to preserve accuracy.

Append to `experiment_repo/实验_算法pipeline.md`:

```md
## <论文标题>

- 属于算法pipeline的实现是什么？实验比较什么？
- 硬件平台是什么，配置是什么。
- 模型是什么。数据集和bench分别是什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

### Serving调度

Use when the implementation modifies an open-source serving framework or request scheduler to improve SLO, throughput, latency, batching, KV cache use, routing, or multi-request execution.

Append to `experiment_repo/实验_Serving调度.md`:

```md
## <论文标题>

- 属于Serving调度的实现是什么？实验比较什么？
- 硬件平台是什么，配置是什么。
- 开源Serving框架是什么。修改了什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

### 编译框架

Use when the implementation modifies a compiler framework, graph compiler, code generator, auto-tuner, or kernel generation system to automate optimized kernels or guarantee performance properties.

Append to `experiment_repo/实验_编译框架.md`:

```md
## <论文标题>

- 属于编译框架的实现是什么？实验比较什么？
- 硬件平台是什么，配置是什么。
- 开源编译框架是什么。修改了什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

### kernel调度

Use when the implementation introduces new kernels, operators, GPU/runtime scheduling, sparse/quantized/communication kernels, transfer scheduling, or backend execution on GPUs, heterogeneous platforms, PIM, or accelerators.

Append to `experiment_repo/实验_kernel调度.md`:

```md
## <论文标题>

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
- 后端平台是什么，配置是什么。
- 评估性能的软件/脚本是什么。修改了什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

### 硬件架构

Use when the implementation is an RTL IP block, hardware module, simulator change, ISA/tile instruction, runtime hardware support, pipeline, memory subsystem, PIM support, or accelerator architecture evaluated under workloads, often with area/power/performance.

Append to `experiment_repo/实验_硬件架构.md`:

```md
## <论文标题>

- 属于硬件架构的实现是什么？实验比较什么？
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
- 模拟器模拟什么的性能，修改了什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

### 芯片设计

Use when the implementation is tied to physical chip organization or technology, such as chiplet, multi-chip module, wafer-scale engine, DRAM/HBM organization, ReRAM, near-memory/processing-in-memory arrays, interconnect packaging, or physical architecture optimization.

Append to `experiment_repo/实验_芯片设计.md`:

```md
## <论文标题>

- 属于芯片设计的实现是什么？实验比较什么？
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
- 模拟器模拟什么的性能，修改了什么。
- 分层说明：<完全匹配 / 部分匹配 / 最接近原因>
```

## Idea Entry

Append one entry to `idea_repo/idea库.md` after experiment classification. Use the paper's `Introduction`, `Motivation`, `Background`, and `Method` / `Design` sections to explain what concrete design solves the baseline pain point. The idea entry should be based on the layers actually selected for the paper, but still describe the baseline and paper method across the model inference algorithm, system framework, compiler framework, kernel scheduling, and hardware architecture stack as far as the paper provides evidence.

Use this exact structure:

```md
## <论文标题>

- baseline方法是什么？
  <说明baseline方法，并给出baseline在模型推理算法-系统框架-编译框架-kernel调度-硬件架构全栈的执行例子。缺失层次写"论文未明确说明"。>

- 论文方法是什么？如何对应解决Baseline的缺陷？
  <说明论文方法，并对比baseline给出论文方法在模型推理算法-系统框架-编译框架-kernel调度-硬件架构全栈的执行例子。缺失层次写"论文未明确说明"。>
```

The stack example should be concrete rather than exhaustive. A good answer follows one request, token, tensor, tile, kernel, memory transfer, or chip-level data path through as many stack layers as the paper supports.

## Quality Checklist

Before finishing:

- Verify every appended file path exists and contains the new entry.
- Verify the target paper was resolved through `paper_2026/` whenever possible, and state when only non-`paper_2026` evidence was available.
- Ensure the same paper is not duplicated in a target file unless the user explicitly asks to add another pass.
- Confirm each experiment layer entry names the comparison target or baseline, the proposed implementation, and the evaluated hardware/simulator/framework when available.
- Confirm the idea entry directly maps baseline defects to method design choices.
- In the final response, list which files were updated and note any missing evidence or approximate layer matches.
