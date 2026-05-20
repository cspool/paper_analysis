---
name: obsidian-keyword-explainer
description: "Search Obsidian notes under knowledge_notes, experiment_notes, idea_notes, and papers for user keywords or semantically segmented paragraph parts, then synthesize detailed Chinese explanations covering what each part is, why it is needed, and how or when to use it. Give concrete examples for every question as formulas, pseudocode, flowcharts, or scheduling timeline diagrams with detailed annotations. If a paper title is provided, search it as context and read referenced documents without explaining the title itself. For paragraph input, split into logical semantic units, explain each unit, then integrate the whole paragraph. Use web search for uncertain keywords, aliases, definitions, or missing context."
---

# Obsidian Keyword Explainer

## Overview

Use this skill to answer keyword or paragraph explanation requests from the Obsidian vault, especially prompts like:

`使用obsidian的API在knowledge_notes、experiment_notes、idea_notes、papers下搜寻<关键词>，具体解释其是什么？为什么需要？如何使用或什么场景使用？`

Default to using the Obsidian MCP/API tools, not filesystem search. The task is read-only unless the user explicitly asks to create, edit, or update notes. When the user provides a paper title, treat the title as a context-search query: use it to locate and read relevant notes, but do not create a standalone explanation section for the title unless the user explicitly asks. When the user provides a paragraph instead of explicit keywords, split it by logical semantics into reusable search units, explain each unit independently, and then explain how the whole paragraph works as one argument or mechanism. When the keyword itself is ambiguous, note evidence is insufficient, or aliases/definitions are uncertain, use web search to supplement and disambiguate. Do not artificially shorten the answer; provide enough detail and concrete examples to make each question useful. Express concrete examples as formulas, pseudocode, flowcharts, or scheduling timeline diagrams whenever possible, and add detailed annotation explaining how to read each formula, pseudocode block, flowchart, or timeline.

## Workflow

1. Extract the target `<关键词>` values from the user request. Support one keyword, multiple keywords, or a full paragraph. Also extract an optional paper title when the request says or implies a specific paper title, such as `论文标题`, `paper title`, `分析论文`, quoted paper names, or a title-like phrase near the requested explanation. If there is no usable keyword, paragraph, or paper title context, ask for it briefly.
2. If the user provides a paragraph, split it into logical semantic units before searching:
   - Keep named concepts, technical terms, algorithms, systems, modules, metrics, assumptions, and causal claims as separate units.
   - Split compound clauses when they express different mechanisms, motivations, implementation steps, or scenarios.
   - Preserve important modifier context, such as `LLM inference`, `GPU`, `KV cache`, `serving`, `sparse`, `compiler`, or `hardware-aware`, so the search unit is not too generic.
   - Deduplicate aliases and near-duplicates, but keep materially different meanings separate.
   - In the final answer, show the segmentation result and briefly explain why each unit was selected.
3. If a paper title is available, search the title before or alongside the keywords. Treat the title as `context_query`, not as an explanation target:
   - Search the title and likely normalized variants, such as subtitle-free title, acronym, Chinese/English title, and filename-like fragments.
   - Read the most relevant documents pointed to by the title search results with `mcp__obsidian__.vault_read`.
   - Use these title-hit documents as context to interpret keyword meaning, disambiguate aliases, and explain how the requested concepts appear in that paper.
   - Do not output `## <论文标题>` as a keyword explanation unless the user explicitly asks to explain the title itself.
4. Process each keyword or semantic unit independently, then synthesize cross-unit relationships only after the per-unit explanations are complete.
5. Search only these vault folders: `knowledge_notes/`, `experiment_notes/`, `idea_notes/`, and `papers/`.
6. Prefer `mcp__obsidian__.search_simple` with the raw keyword, semantic unit, or paper-title context query and `contextLength` around `160`-`240`; filter returned `filename` values to the four target folder prefixes.
7. If results are sparse, too broad, or path filtering matters, use `mcp__obsidian__.search_query` with a folder constraint plus keyword checks against `path` and `content`. Escape regex metacharacters in the keyword before putting it in a `regexp`.

```json
{
  "and": [
    {
      "or": [
        { "regexp": ["^knowledge_notes/", { "var": "path" }] },
        { "regexp": ["^experiment_notes/", { "var": "path" }] },
        { "regexp": ["^idea_notes/", { "var": "path" }] },
        { "regexp": ["^papers/", { "var": "path" }] }
      ]
    },
    {
      "or": [
        { "regexp": ["<escaped-keyword>", { "var": "path" }] },
        { "regexp": ["<escaped-keyword>", { "var": "content" }] }
      ]
    }
  ]
}
```

8. If needed, use `mcp__obsidian__.vault_list` to confirm folder names and `mcp__obsidian__.vault_read` to read the most relevant notes. Read full notes for the best matches instead of relying only on search snippets. For title-search results, read full notes that appear to describe the paper, its experiments, its ideas, or its terminology even if they do not contain every target keyword.
9. Synthesize across the three note types:
   - `knowledge_notes/`: definitions, mechanisms, terminology background.
   - `experiment_notes/`: implementation evidence, baselines, metrics, reproducibility details, empirical use.
   - `idea_notes/`: motivations, pain points, design ideas, scenarios where the term is useful.
   - `papers/`: paper full text, title context, method details, evaluation setup, examples, and terminology as used in the source paper.
10. If exact hits are missing, try obvious aliases, Chinese/English variants, spacing/hyphen variants, singular/plural variants, and acronym expansions inferred from the user's keyword, semantic unit, title context, or title-hit documents. State when no direct note evidence was found.
11. Use web search for uncertain keywords or weak note evidence, including ambiguous terms, acronym expansion, term origin, current common usage, paper title verification, or missing implementation context. Prefer authoritative sources such as papers, official docs, project documentation, standards, and well-known technical references. Cite web sources with links in the final answer when used.
12. For every keyword or semantic unit and every major question (`是什么？`, `为什么需要？`, `如何使用？什么场景使用？`), include at least one concrete example. Express each example as one of:
    - Formula: use Obsidian/MathJax-compatible syntax for metrics, cost models, tensor shapes, memory/latency/throughput analysis, algorithmic relationships, or probability/objective functions.
    - Pseudocode: use for algorithms, schedulers, runtime policies, compiler passes, kernel execution, data processing, or decision logic.
    - Flowchart: use Mermaid `flowchart` blocks for system workflows, pipeline stages, request paths, component interactions, or usage scenarios.
    - Scheduling timeline diagram: use Mermaid `sequenceDiagram` or `gantt` blocks for kernel scheduling, pipeline overlap, prefill-decode multiplexing, heterogeneous CPU/GPU/PIM/FPGA execution, communication-computation overlap, batching windows, or resource occupancy over time.
    After each formula, pseudocode block, flowchart, or scheduling timeline diagram, add detailed annotations:
    - Formula annotations must explain every variable, unit/dimension when relevant, assumptions, and what changing each key term means. Write formulas in Obsidian/MathJax-compatible Markdown: use display math with `$$` on separate lines; use inline math with `$...$`; do not wrap formulas in code fences; keep Chinese prose outside math blocks; prefer ASCII variable names plus `\mathrm{}` or `\operatorname{}` for readable labels; avoid custom macros or LaTeX packages that Obsidian may not support.
    - Pseudocode annotations must explain inputs, outputs, each major loop/branch/state update, and the algorithmic purpose of the block.
    - Flowchart annotations must explain each node, each labeled edge/branch, the start/end condition, and how the path maps to the concept being explained.
    - Scheduling timeline annotations must explain time axis or ordering, resources/lanes, kernel or task duration, dependencies, overlap, stalls, synchronization points, and how the schedule affects latency/throughput/goodput.
    If no formula, pseudocode, flowchart, or scheduling timeline can be verified from notes or web sources, explicitly state the evidence gap and give the closest safe structured example plus its annotation.
13. If the input was a paragraph, finish with an integrated paragraph-level explanation:
    - Explain the paragraph's overall thesis or mechanism.
    - Show how the semantic units connect, preferably with a Mermaid flowchart.
    - Identify which parts are directly supported by notes, supplemented by web search, or inferred.

## Answer Format

Answer in Chinese. Use this structure unless the user requests another format. For multiple keywords or paragraph semantic units, repeat the full section for each unit. Do not compress away examples, evidence, or uncertainty notes for token-saving reasons.

```md
## 论文标题上下文
<仅当用户提供或可明确识别论文标题时使用。列出论文标题、标题检索命中的上下文文档，以及这些文档如何帮助解释关键词/语义单元。说明标题只作为上下文查询，不作为待解释关键词。>

## 语义切分
<仅当用户输入段落时使用。列出切分出的关键词/语义单元，并说明每个单元承载的概念、机制、原因、步骤或场景。>

## <关键词>

### 是什么？
<基于命中笔记解释概念本身。必要时区分笔记证据和合理推断。>

具体例子：
<用 Obsidian/MathJax 兼容公式、伪代码、Mermaid flowchart 或调度时序图表达至少一个具体例子，说明这个概念在系统、论文、实验、代码、模型或实际场景中是什么样。随后用“注释解释”详细说明公式变量/伪代码步骤/流程图节点与边/时序图资源泳道、时间顺序、依赖与重叠关系。>

### 为什么需要？
<解释它解决的痛点、引入原因、对系统/实验/想法的价值。>

具体例子：
<用 Obsidian/MathJax 兼容公式、伪代码、Mermaid flowchart 或调度时序图表达至少一个具体例子，说明没有它会遇到什么问题，或者引入它后解决了什么问题。随后用“注释解释”详细说明公式变量/伪代码步骤/流程图节点与边/时序图资源泳道、时间顺序、依赖与重叠关系。>

### 如何使用？什么场景使用？
<说明典型使用方式、落地流程、适用场景、相关限制。>

具体例子：
<用 Obsidian/MathJax 兼容公式、伪代码、Mermaid flowchart 或调度时序图表达至少一个具体例子，说明实际怎么用、在哪类任务/系统/实验中使用。优先给出可执行步骤、调度逻辑、计算关系、端到端流程或 kernel/pipeline 时序安排。随后用“注释解释”详细说明公式变量/伪代码步骤/流程图节点与边/时序图资源泳道、时间顺序、依赖与重叠关系。>

### 笔记依据
- `<vault-relative-path>`: <该笔记提供的关键信息>
- `<title-hit-context-path>`: <如果该笔记来自论文标题检索，说明它作为上下文如何支持本关键词/语义单元的解释。>

### 联网补充依据
- <source link>: <当使用联网搜索时，简述该来源补充了什么；如果未使用联网搜索，写“未使用，Obsidian 笔记证据足够”。>

### 不确定处
<列出笔记未明确说明、命中不足、别名不确定或需要进一步确认的点；没有就写“暂无明显不确定处”。>

## 整段综合解释
<仅当用户输入段落时使用。综合解释原段落的整体含义、逻辑链条、各语义单元之间的因果/流程/层次关系。优先使用 Mermaid flowchart 展示整体逻辑，再用文字解释。>
```

## Evidence Rules

- Cite vault-relative note paths for claims derived from notes.
- Cite web links for claims derived from web search.
- When a paper title is available, use the title as an additional Obsidian search query and read the resulting documents as context. The title is not an explanation target unless explicitly requested.
- Keep title-hit documents separate from direct keyword hits when reporting evidence. Use wording like `标题检索命中的上下文文档显示` for title-derived context.
- When segmenting a paragraph, preserve the user's original meaning. Do not over-split short phrases into generic words that will produce noisy searches.
- Separate note-backed facts from inference. Use wording like `笔记显示` for sourced claims and `可推断` for synthesis.
- Separate web-backed facts from note-backed facts. Use wording like `联网资料显示` when relying on web sources.
- Do not invent implementation details. If the notes do not explain something, write `笔记未明确说明`.
- Prefer detailed, example-rich technical explanations over terse summaries. Do not apply an artificial output token limit when the user asks for depth.
- Ensure each keyword or semantic unit has concrete examples under all three required questions. Express examples with formulas, pseudocode, Mermaid flowcharts, or scheduling timeline diagrams instead of plain prose whenever possible. Use scheduling timeline diagrams especially for kernel scheduling, pipeline arrangements, resource contention, overlap, or batching. Every formula, pseudocode block, flowchart, and scheduling timeline diagram must be followed by detailed annotation explaining variables, steps, nodes, edges, resources, time axis, dependencies, overlap, stalls, assumptions, and how the example answers the current question. Formula examples must render in Obsidian/MathJax: display equations use standalone `$$` delimiters, inline equations use `$...$`, and formulas are not placed inside fenced code blocks. If a question has no verified structured example, state the evidence gap directly.
- For paragraph input, include a final integrated explanation after all per-unit explanations; do not stop at isolated keyword definitions.
- Keep the search scope anchored to `knowledge_notes/`, `experiment_notes/`, `idea_notes/`, and `papers/`; mention if relevant matches appeared outside the requested scope, but do not use them as primary evidence unless the user approves.
