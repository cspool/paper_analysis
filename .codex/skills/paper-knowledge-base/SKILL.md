---
name: paper-knowledge-base
description: Extract and maintain a layered academic terminology knowledge base from local AI systems, LLM inference, accelerator, compiler/runtime, serving, GPU kernel, or architecture papers. Use when the user provides a paper title, title number, filename, or workspace identifier and asks to learn knowledge, extract key terms, update knowledge_repo, or build Chinese knowledge-base entries that explain what each term is, decompose it from the assigned layer's perspective with a concrete process or pseudocode example, and describe how it is generally implemented or used, using web search when helpful to make the wording concrete and precise.
---

# Paper Knowledge Base

## Overview

Use this skill to turn one target paper into layered Chinese terminology entries under `knowledge_repo/`. The goal is not a broad paper summary; it is to capture reusable academic terms, explain each term at the correct system-stack layer, decompose how the term works from that layer's perspective, describe how the term is generally implemented or used, and connect the term to the paper that used it.

Default to the current workspace. Search `paper_2026/` first for the original PDF and any paired extracted full text (`.txt`, `.md`). Use title lists, downloaded PDFs, extracted text files, and Markdown notes only to identify or support the target paper.

## Workflow

1. Identify the target paper from the user's title, title number, filename, or local identifier.
2. Search `paper_2026/` with `rg --files`, title fragments, paper numbers, and likely filename variants to find the original PDF. If the user gives a number, map it through local title lists such as `paper_titles.md` first.
3. Prefer readable full text in this order: paired extracted `.txt` / `.md` from `paper_2026/`, PDF text extraction from the original PDF, official paper HTML, then other local notes. Do not rely only on abstracts or secondary summaries when full text is available.
4. Read the sections most useful for terminology extraction: `Background`, `Preliminary`, `Motivation`, `Baseline`, `Introduction`, `Method`, `Design`, `Architecture`, and `Approach`. Read implementation/evaluation only when needed to understand a term's concrete use.
5. Extract key academic terms that are reusable beyond a single sentence: mechanisms, modules, scheduling concepts, kernels, frameworks, algorithms, model components, hardware components, memory/interconnect concepts, and system policies. For each term, capture what it is, a layer-specific decomposition with a concrete operating flow, pseudocode, or computation process, and how it is generally implemented or used. Combine the paper description with web search when it helps make the explanation concrete and precise.
6. Classify each term into the closest layer in the taxonomy below. If a term cannot be assigned precisely, do not skip it: choose the most likely or closest layer based on the paper context and mark the assignment as approximate in the final report. Multiple layers are allowed only when the term has materially different meanings or examples at each layer.
7. Update the corresponding Markdown files under `knowledge_repo/`. Create `knowledge_repo/` and any missing `知识库_<层次>.md` files if needed. Do not overwrite unrelated content.
8. If a layer file already contains the same term or a clear alias, merge the new paper-specific explanation into the existing entry instead of creating a duplicate. Add the paper title to `涉及论文标题`.
9. In the final response, list the updated files, the resolved paper source, and any terms whose classification or explanation was uncertain.

## Evidence Rules

- Ground explanations in the paper text. If a term is named but not explained, infer only from nearby paper context and mark the gap clearly.
- Use web search when it helps make a term explanation, layer-specific decomposition, or implementation/use description concrete and precise, especially when the local paper text is terse. Prefer official docs, papers, project pages, or authoritative references, and keep the entry grounded in the target paper's usage.
- If evidence is missing, write `论文未明确说明` rather than inventing details.
- Preserve concrete names for models, frameworks, kernels, simulators, chips, interconnects, memory systems, scheduling policies, and metrics.
- Keep each term concise but useful for later research reuse. Avoid generic one-line dictionary definitions when the paper provides a more specific usage.
- For `从<层次>角度拆解术语...`, explain how the term works inside the assigned layer rather than giving a generic definition. Prefer an operating flow. For `kernel调度` and `算法pipeline`, include pseudocode or a concrete computation process when the paper gives enough detail.
- For `术语一般如何实现？如何使用？`, combine the paper's concrete description with reliable background knowledge when needed. Prefer implementation/use mechanisms such as chip interconnect organization, hardware modules, kernel execution, compiler optimization, serving/runtime behavior, or model pipeline steps. If the paper and references do not give enough context, write `论文未明确说明`.
- For imprecise layer matches, explain the closest-layer reasoning briefly in the final response; only add that reasoning to the entry itself when it helps future reuse.
- Treat aliases carefully. Merge obvious variants such as `KV Cache`, `KVCache`, and `Key-Value Cache`; keep the most standard visible term as the heading and mention aliases in the explanation.

## Entry Format

Use the destination layer's exact format for new entries. Each definition, decomposition, and implementation/use field explicitly asks for web search to make the answer concrete and precise; keep that wording in the field names. If an existing entry uses a compatible local format, preserve useful old content while migrating or extending it toward these fields.

```md
## <术语名>

术语是什么？通过联网搜索让回答具体和精准。
<解释该术语本身是什么。>

从<层次>角度拆解术语，<按目标层次要求给出运转流程、伪代码、具体计算过程或具体例子。>通过联网搜索让回答具体和精准。
<说明该术语从当前归属层次看如何发挥作用。优先结合论文描述给出具体流程；必要时用可靠网络来源补充通用理解。>

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
<结合论文描述和必要的可靠背景知识，说明该术语通常如何落地、实现或使用。>

涉及论文标题：
- <论文标题>
```

When updating an existing term:

- Preserve useful old explanation and append the new paper-specific nuance.
- Add or revise `术语是什么？通过联网搜索让回答具体和精准。`, the layer-specific `从<层次>角度拆解术语...通过联网搜索让回答具体和精准。` field, and `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` when the new paper gives clearer or different evidence.
- Append the paper title only if it is not already listed.
- Avoid duplicate headings for the same term in the same layer file.

Use the exact field names for the destination layer:

- `知识库_芯片设计.md`: `术语是什么？通过联网搜索让回答具体和精准。` / `从芯片设计角度拆解术语，比如术语术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`
- `知识库_硬件架构.md`: `术语解释` / `术语是什么？通过联网搜索让回答具体和精准。` / `从硬件架构角度拆解术语，比如术语术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`
- `知识库_kernel调度.md`: `术语是什么？通过联网搜索让回答具体和精准。` / `从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`
- `知识库_编译框架.md`: `术语是什么？通过联网搜索让回答具体和精准。` / `从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`
- `知识库_系统架构.md`: `术语是什么？通过联网搜索让回答具体和精准。` / `从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`
- `知识库_算法pipeline.md`: `术语是什么？通过联网搜索让回答具体和精准。` / `从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。` / `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。` / `涉及论文标题`

For `知识库_硬件架构.md`, preserve the extra `术语解释` field because the source specification includes it.

## Layer Taxonomy

### 芯片设计

Write to `knowledge_repo/知识库_芯片设计.md`.

Use for physical chip organization and chip-level technology terms: `chiplet`, multi-chip module, wafer-scale engine (`WSE`), `NvLink` or equivalent chip/interconnect fabric, HBM/DRAM physical organization, ReRAM or near-memory arrays, package-level bandwidth, die-to-die links, memory channels, and physical architecture constraints.

### 硬件架构

Write to `knowledge_repo/知识库_硬件架构.md`.

Use for hardware platforms and architectural modules above physical chip layout: multi-GPU systems, heterogeneous `XU+YU` style systems, GPU, CPU, NPU, FPGA, PIM, accelerators, memory hierarchy, compute units, hardware queues, on-chip buffers, DMA engines, and architecture-level execution resources.

### kernel调度

Write to `knowledge_repo/知识库_kernel调度.md`.

Use for operator/runtime execution mechanisms and same-level academic terms: sparse kernels, attention kernels, activation quantization or sparsity implementation, tensor/pipeline/data/expert parallelism when described as kernel/runtime mapping, operator-to-hardware alignment, kernel pipeline overlap, GPU thread/block/warp implementation, PIM calls, heterogeneous kernel execution, communication kernels, layout transforms, and backend scheduling.

### 编译框架

Write to `knowledge_repo/知识库_编译框架.md`.

Use for framework, compiler, optimization-method, and same-level academic terms: Triton, PyTorch, SGLang internals when discussed as framework/compiler machinery, graph compilers, code generation, auto-tuning, scheduling IR, integer linear programming, polyhedral optimization, greedy or heuristic optimization, offline/online operator optimization, and kernel generation systems.

### 系统架构

Write to `knowledge_repo/知识库_系统架构.md`.

Use for serving/runtime systems, multi-request resource management, and same-level academic terms: vLLM, `xxServe` systems, serverless frameworks, request scheduling, batching, routing, layer/token-level scheduling, dynamic scheduling, preemptive scheduling, SLO-aware scheduling, SM partitioning, dynamic SM allocation, prefix sharing, shared KV cache, KV cache eviction/offloading, and hardware-aware multi-request scheduling.

### 算法pipeline

Write to `knowledge_repo/知识库_算法pipeline.md`.

Use for model pipeline, algorithm pipeline, and same-level academic terms: MoE, Transformer, VLM, LLM, DiT, LoRA, SFT, generative recommendation, decoding algorithms, model inference pipeline stages, sparsity, compression, quantization, distillation, shared or approximate weights/activations, token pruning, speculative decoding, and accuracy-preserving algorithm changes.

## Quality Checklist

Before finishing:

- Verify every updated file path exists and contains the new or merged entry.
- Verify the target paper was resolved through `paper_2026/` whenever possible.
- Check that each term has the required fields for its destination layer, especially `术语是什么？通过联网搜索让回答具体和精准。`, the exact `从<层次>角度拆解术语...通过联网搜索让回答具体和精准。` field, `术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。`, and `涉及论文标题`.
- Check that existing terms were merged rather than duplicated.
- Check that every `涉及论文标题` includes the current paper title.
- Note any approximate layer choices, closest-layer reasoning, and missing paper evidence in the final response.
