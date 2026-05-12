---
name: paper-knowledge-base
description: Extract and maintain a layered academic terminology knowledge base from local AI systems, LLM inference, accelerator, compiler/runtime, serving, GPU kernel, or architecture papers. Use when the user provides a paper title, title number, filename, or workspace identifier and asks to learn knowledge, extract key terms, update knowledge_repo, or build Chinese knowledge-base entries from paper Background, Preliminary, Motivation, Baseline, Method, Design, or Architecture sections.
---

# Paper Knowledge Base

## Overview

Use this skill to turn one target paper into layered Chinese terminology entries under `knowledge_repo/`. The goal is not a broad paper summary; it is to capture reusable academic terms, explain each term at the correct system-stack layer, and connect the term to the paper that used it.

Default to the current workspace. Search `paper_2026/` first for the original PDF and any paired extracted full text (`.txt`, `.md`). Use title lists, downloaded PDFs, extracted text files, and Markdown notes only to identify or support the target paper.

## Workflow

1. Identify the target paper from the user's title, title number, filename, or local identifier.
2. Search `paper_2026/` with `rg --files`, title fragments, paper numbers, and likely filename variants to find the original PDF. If the user gives a number, map it through local title lists such as `paper_titles.md` first.
3. Prefer readable full text in this order: paired extracted `.txt` / `.md` from `paper_2026/`, PDF text extraction from the original PDF, official paper HTML, then other local notes. Do not rely only on abstracts or secondary summaries when full text is available.
4. Read the sections most useful for terminology extraction: `Background`, `Preliminary`, `Motivation`, `Baseline`, `Introduction`, `Method`, `Design`, `Architecture`, and `Approach`. Read implementation/evaluation only when needed to understand a term's concrete use.
5. Extract key academic terms that are reusable beyond a single sentence: mechanisms, modules, scheduling concepts, kernels, frameworks, algorithms, model components, hardware components, memory/interconnect concepts, and system policies.
6. Classify each term into the closest layer in the taxonomy below. Multiple layers are allowed only when the term has materially different meanings or examples at each layer.
7. Update the corresponding Markdown files under `knowledge_repo/`. Create `knowledge_repo/` and any missing `知识库_<层次>.md` files if needed. Do not overwrite unrelated content.
8. If a layer file already contains the same term or a clear alias, merge the new paper-specific explanation into the existing entry instead of creating a duplicate. Add the paper title to `涉及论文标题`.
9. In the final response, list the updated files, the resolved paper source, and any terms whose classification or explanation was uncertain.

## Evidence Rules

- Ground explanations in the paper text. If a term is named but not explained, infer only from nearby paper context and mark the gap clearly.
- Use web search only when the local paper text is insufficient to explain a general academic term and network access is available or explicitly requested. Prefer official docs, papers, project pages, or authoritative references.
- If evidence is missing, write `论文未明确说明` rather than inventing details.
- Preserve concrete names for models, frameworks, kernels, simulators, chips, interconnects, memory systems, scheduling policies, and metrics.
- Keep each term concise but useful for later research reuse. Avoid generic one-line dictionary definitions when the paper provides a more specific usage.
- Treat aliases carefully. Merge obvious variants such as `KV Cache`, `KVCache`, and `Key-Value Cache`; keep the most standard visible term as the heading and mention aliases in the explanation.

## Entry Format

Use this format for new entries. If an existing entry uses a compatible local format, preserve its style while keeping these four fields.

```md
## <术语名>

术语解释：
<解释该术语是什么，处于哪一层，论文中为什么需要它。>

术语关联术语的使用例子：
<说明该术语如何与相关术语共同使用，最好给出论文中的一个具体数据流、请求流、kernel、模型或硬件路径例子。>

涉及论文标题：
- <论文标题>
```

When updating an existing term:

- Preserve useful old explanation and append the new paper-specific nuance.
- Add a new example if the new paper uses the term differently.
- Append the paper title only if it is not already listed.
- Avoid duplicate headings for the same term in the same layer file.

## Layer Taxonomy

### 芯片设计

Write to `knowledge_repo/知识库_芯片设计.md`.

Use for physical chip organization and chip-level technology terms: `chiplet`, multi-chip module, wafer-scale engine (`WSE`), `NvLink` or equivalent chip/interconnect fabric, HBM/DRAM physical organization, ReRAM or near-memory arrays, package-level bandwidth, die-to-die links, memory channels, and physical architecture constraints.

### 硬件架构

Write to `knowledge_repo/知识库_硬件架构.md`.

Use for hardware platforms and architectural modules above physical chip layout: multi-GPU systems, heterogeneous `XU+YU` style systems, GPU, CPU, NPU, FPGA, PIM, accelerators, memory hierarchy, compute units, hardware queues, on-chip buffers, DMA engines, and architecture-level execution resources.

### kernel调度

Write to `knowledge_repo/知识库_kernel调度.md`.

Use for operator/runtime execution mechanisms: sparse kernels, attention kernels, activation quantization or sparsity implementation, tensor/pipeline/data/expert parallelism when described as kernel/runtime mapping, operator-to-hardware alignment, kernel pipeline overlap, GPU thread/block/warp implementation, PIM calls, heterogeneous kernel execution, communication kernels, layout transforms, and backend scheduling.

### 编译框架

Write to `knowledge_repo/知识库_编译框架.md`.

Use for framework, compiler, and optimization-method terms: Triton, PyTorch, SGLang internals when discussed as framework/compiler machinery, graph compilers, code generation, auto-tuning, scheduling IR, integer linear programming, polyhedral optimization, greedy or heuristic optimization, offline/online operator optimization, and kernel generation systems.

### 系统架构

Write to `knowledge_repo/知识库_系统架构.md`.

Use for serving/runtime systems and multi-request resource management: vLLM, `xxServe` systems, serverless frameworks, request scheduling, batching, routing, layer/token-level scheduling, dynamic scheduling, preemptive scheduling, SLO-aware scheduling, SM partitioning, dynamic SM allocation, prefix sharing, shared KV cache, KV cache eviction/offloading, and hardware-aware multi-request scheduling.

### 算法pipeline

Write to `knowledge_repo/知识库_算法pipeline.md`.

Use for model and algorithm pipeline terms: MoE, Transformer, VLM, LLM, DiT, LoRA, SFT, generative recommendation, decoding algorithms, model inference pipeline stages, sparsity, compression, quantization, distillation, shared or approximate weights/activations, token pruning, speculative decoding, and accuracy-preserving algorithm changes.

## Quality Checklist

Before finishing:

- Verify every updated file path exists and contains the new or merged entry.
- Verify the target paper was resolved through `paper_2026/` whenever possible.
- Check that each term has the four required fields.
- Check that existing terms were merged rather than duplicated.
- Check that every `涉及论文标题` includes the current paper title.
- Note any approximate layer choices or missing paper evidence in the final response.
