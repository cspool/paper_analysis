---
name: obsidian-keyword-explainer
description: Search Obsidian vault under paper_secs, knowledge_notes, experiment_notes, idea_notes for keywords using omnisearch, explaining what each is, why it is needed, and how / in what scenarios to use it. Supports paragraph semantic segmentation. Supplements uncertain terms with web search. Every question gets concrete examples as Obsidian/MathJax-compatible formulas, pseudocode, Mermaid flowcharts, or scheduling timeline diagrams with detailed annotations. No output length limit.
---

# Obsidian Keyword Explainer

## Overview

Use this skill to answer keyword or paragraph explanation requests from the Obsidian vault, especially prompts like:

`使用obsidian的API在paper_secs、knowledge_notes、experiment_notes、idea_notes下搜寻<关键词>，具体解释其是什么？为什么需要？如何使用或什么场景使用？`

Default to using the Obsidian MCP/API tools (omnisearch mode), not filesystem search. The task is read-only unless the user explicitly asks to create, edit, or update notes. When the user provides a paragraph, split it by logical semantics into segments, extract keywords per segment (no cross-segment grouping), search each keyword via omnisearch across all four directories, deduplicate and read all matched files, explain each segment with the collected context, and then synthesize the whole paragraph. When the keyword itself is ambiguous, note evidence is insufficient, or aliases/definitions are uncertain, use web search to supplement and disambiguate. Do not artificially shorten the answer; provide enough detail and concrete examples to make each question useful. Express concrete examples as formulas, pseudocode, flowcharts, or scheduling timeline diagrams whenever possible, and add detailed annotation explaining how to read each formula, pseudocode block, flowchart, or timeline. The path below is related or based on '/data3/paper_analysis/'. 

## Workflow

### Step 1: Semantic Segmentation

Split the input paragraph into **semantic segments** — each segment is a semantically coherent unit (cause/effect, contrast, parallel, conditional, progressive, comparative, or a single sentence/clause).

**1.1 Segment** by logical boundaries: cause/effect, contrast, parallel, conditional, progressive, comparative, and sentence/clause breaks.

**1.2 Extract core terms** from each segment: acronyms (KV Cache, MoE, SLO), domain terms (speculative decoding, quantization), method/framework names (vLLM, Triton), hardware/system concepts (PIM, chiplet, SM).

**1.3 Do NOT cluster/group** keywords across segments. Each keyword is processed independently in subsequent steps. Keywords belong to their originating segment only.

**1.4 Expand** each keyword with aliases, Chinese/English variants, acronym expansions, separator variants (for search coverage).

**1.5 Report segments** in final answer. Example:

| Segment | Keywords | Semantic Role |
|---------|----------|---------------|
| S1 | KV Cache, key-value cache, memory footprint | Current approach: pros & bottleneck |
| S2 | PagedAttention, Page, block management, fragmentation | Solution & its effects |

---

### Step 2: Paper Title Context

Skip Step 2.

---

### Step 3: Per-Keyword Search via Omnisearch

For **each keyword** extracted in Step 1, run **four** separate omnisearch queries — one per target directory. Each query returns scored results; record up to 5 paths per directory per keyword, ordered by score descending.

**API**: `obsidian_search_notes` with `mode="omnisearch"`. Results capped at 50 upstream; paginate via `cursor` / `nextCursor` if needed.

**Query format**: `path:<dir> <keyword>`

- Single-word keyword: `path:paper_secs KV-Cache`
- Multi-word keyword: use quoted phrases — `path:paper_secs "KV Cache"`
- Path filter is embedded in the query string. Do NOT use `pathPrefix` (text-mode only).
- Use both English and Chinese variants of the keyword in separate searches if applicable. Combine results from variant searches for the same keyword.

#### 3.1 Search `paper_secs/`

```
obsidian_search_notes(mode="omnisearch", query="path:paper_secs <keyword>")
```

Record up to 5 paths with the highest scores. Score = API-returned score directly.

#### 3.2 Search `knowledge_notes/`

```
obsidian_search_notes(mode="omnisearch", query="path:knowledge_notes <keyword>")
```

Record up to 5 paths with the highest scores.

#### 3.3 Search `experiment_notes/`

```
obsidian_search_notes(mode="omnisearch", query="path:experiment_notes <keyword>")
```

Record up to 5 paths with the highest scores.

#### 3.4 Search `idea_notes/`

```
obsidian_search_notes(mode="omnisearch", query="path:idea_notes <keyword>")
```

Record up to 5 paths with the highest scores.

#### 3.5 Relevance Scoring

**Directly use each match's score** from the omnisearch API response. No custom scoring formula needed. The omnisearch BM25 score is the relevance score.

#### 3.6 Tracking

Maintain a running table during search:

| Keyword | Segment | Dir | Path | Score |
|---------|---------|-----|------|-------|
| KV Cache | S1 | paper_secs | paper_secs/2025/xxx.md | 12.5 |
| KV Cache | S1 | knowledge_notes | knowledge_notes/kv_cache.md | 8.3 |
| ... | ... | ... | ... | ... |

---

### Step 4: Deduplicate, Read & Build Context

After ALL keywords have been searched:

**4.1 Deduplicate**: collect all unique paths from the search results table. De-duplicate by vault-relative path.

**4.2 Read**: call `obsidian_get_note` for each unique path to read the full content.

**4.3 Build context**: assemble all read notes into a unified context. For each file, annotate which keyword(s) and segment(s) it was matched for, along with the omnisearch score(s).

**4.4 Supplement**: for keywords with zero matches across all four directories, mark as "no note evidence" — these will be supplemented via web search in Step 5.

---

### Step 5: Segment Explanation

For **each semantic segment** from Step 1, output a complete explanation section using the context built in Step 4 + Step 2.

First explain each keyword in the segment, then explain the segment as a whole.

1. **What is it?** — Explain the concepts based on matched notes. Distinguish note evidence from inference.
2. **Why is it needed?** — Explain the pain points solved, motivation, value.
3. **How to use? In what scenarios?** — Typical usage, implementation flow, applicable scenarios, limitations.

4. **Concrete examples**: For each of the three questions, provide at least one structured example from:
   - **Formula**: Obsidian/MathJax-compatible. For metrics, cost models, memory/latency analysis.
   - **Pseudocode**: For algorithms, schedulers, kernel execution, decision logic.
   - **Flowchart**: Mermaid `flowchart`. For system workflows, pipeline stages, request paths.
   - **Scheduling Timeline**: Mermaid `gantt` or `sequenceDiagram`. For kernel scheduling, pipeline overlap, communication-computation overlap.

   After each structured example, add an **"Annotations"** section explaining variables, steps, nodes, edges, resource lanes, time axis, dependencies, overlap, stalls, assumptions, and how the example answers the current question.

5. **Note evidence**: List vault paths with omnisearch scores and key information provided.

6. **Web supplement**: Use `WebSearch` for keywords with insufficient note evidence. Cite authoritative sources with links.

---

### Step 6: Paragraph Synthesis

After all segments are explained, produce an integrated paragraph-level explanation:

1. **Overall summary**: 2-4 sentences capturing the core thesis and logical chain.
2. **Segment relationships**: Show how segments connect (causal, parallel, progressive, contrast, conditional) with a Mermaid flowchart.
3. **Integrated understanding**: Restore the paragraph's complete logic. Cover: core claim, how each segment supports it, implicit premises/assumptions, position in the broader field.
4. **Technical comparison table** (if applicable): When the paragraph compares multiple techniques.

---

## Answer Format

Answer in Chinese. Use this structure unless the user requests another format. For each segment, repeat the full section. Do not compress away examples, evidence, or uncertainty notes for token-saving reasons.

```md
## Paper Context
<Only when a paper title is available (Step 2). List the paper title, matched documents, and how they help as context.>

## Semantic Segments
<Table: Segment | Keywords | Semantic Role>

## Context Summary
<Table of all unique paths read, with matching keywords and omnisearch scores>

## S<n>: <one-line segment summary>

**Segment Keywords**: `<kw1>`, `<kw2>`, ...

**Relevant Context**:
- `<vault-path>` (omnisearch score: <score>): <key context>

### What is it?
<Explain the concepts based on matched notes. Distinguish note evidence from inference.>

Concrete example:
<Structured example (formula/pseudocode/flowchart/timeline) + Annotations.>

### Why is it needed?
<Explain pain points solved, motivation, value.>

Concrete example:
<Structured example + annotations.>

### How to use? In what scenarios?
<Typical usage, implementation flow, applicable scenarios, limitations.>

Concrete example:
<Structured example + annotations. Prefer executable steps, scheduling logic, end-to-end flow, or kernel/pipeline timeline.>

### Note Evidence
- `<vault-path>` (omnisearch score: <score>): <key information>

### Web Supplement
- <source link>: <What this source supplements; or "Not used — Obsidian note evidence sufficient.">

### Uncertainties
<Uncertainties, insufficient hits, ambiguous aliases; or "No significant uncertainties.">

## Paragraph Synthesis
<Only for paragraph input: Overall Summary → Segment Relationships (Mermaid) → Integrated Understanding → Technical Comparison (if any) → Note Evidence>
```

---

## Mermaid Syntax Safety Rules

When writing Mermaid `flowchart` diagrams, obey these rules strictly. Violating them causes `Parse error`:

1. **Always double-quote node text**: `A["text"]` not `A[text]`. Mandatory for nodes with Chinese, spaces, parentheses, or non-ASCII.
2. **Always double-quote edge labels**: `-->|"label"|` not `-->|label|`.
3. **Forbidden characters — replace before writing**:

| Forbidden | Replacement | Reason |
|-----------|-------------|--------|
| `^` | `#Hat;` or rephrase (e.g. `qK^T` → `qK-transpose`) | Arrow marker |
| `<br>` | `<br/>` | Unclosed HTML tag |
| `×` | `x` | Non-ASCII math symbol |
| `&` | `&amp;` | HTML entity start |
| `<` `>` (outside HTML) | `&lt;` `&gt;` | HTML tag boundaries |
| `"` (inside quoted text) | `#quot;` or rephrase | Quote delimiter |

4. **Node IDs**: Keep short and alphanumeric only (`A`, `B1`, `N2`). No Chinese or special chars.
5. **Multi-line nodes**: Use `<br/>`: `A["Line 1<br/>Line 2"]`. Never `<br>`.
6. **Subgraph titles**: always quoted: `subgraph "Title"`

---

## Formula Guidelines

- Block: `$$...$$` on separate lines, blank lines around. Inline: `$...$`. **Never in code fences.**
- ASCII variable names; use `\mathrm{Label}` or `\operatorname{name}` for labels.
- No custom macros, extra packages, or Chinese `\text{...}`. Multi-line: `aligned` environment.
- Underscore variables in prose: wrap in `$...$` or backticks.

---

## Evidence Rules

- Cite vault-relative note paths for claims derived from notes.
- Cite web links for claims derived from web search.
- When a paper title is available (Step 2), search it ONLY in `papers/` by filename match via omnisearch and read the matched `.md` as context. The title is not an explanation target unless explicitly requested.
- Separate note-backed facts from inference. Use wording like "Notes show" for sourced claims and "Can be inferred" for synthesis.
- Separate web-backed facts from note-backed facts. Use wording like "Web sources show" when relying on web sources.
- Do not invent implementation details. If the notes do not explain something, write "Notes do not explicitly clarify."
- Prefer detailed, example-rich technical explanations over terse summaries. Do not apply an artificial output token limit when the user asks for depth.
- Ensure each segment has concrete examples under all three required questions. Express examples with formulas, pseudocode, Mermaid flowcharts, or scheduling timeline diagrams instead of plain prose whenever possible. Use scheduling timeline diagrams especially for kernel scheduling, pipeline arrangements, resource contention, overlap, or batching. Every formula, pseudocode block, flowchart, and scheduling timeline diagram must be followed by detailed annotation explaining variables, steps, nodes, edges, resources, time axis, dependencies, overlap, stalls, assumptions, and how the example answers the current question. Formula examples must render in Obsidian/MathJax: display equations use standalone `$$` delimiters, inline equations use `$...$`, and formulas are not placed inside fenced code blocks. If a question has no verified structured example, state the evidence gap directly.
- For paragraph input, include a final integrated explanation after all per-segment explanations; do not stop at isolated keyword definitions.
- Search scope: `paper_secs/`, `knowledge_notes/`, `experiment_notes/`, `idea_notes/`. `papers/` is only used in Step 2 for paper title matching (filename match → read original .md as context).
- Relevance = omnisearch BM25 score directly. No custom formula.
- Each keyword searches all four directories; up to 5 paths per directory per keyword.
- When segmenting a paragraph, preserve the user's original meaning. Do not over-split short phrases into generic words that will produce noisy searches.

## Quality Checklist

- [ ] Step 1: Paragraph semantically segmented; keywords extracted per segment (no cross-segment grouping)
- [ ] Step 2: Paper context searched by title in `papers/` via omnisearch, matched .md read if available
- [ ] Step 3: Per-keyword omnisearch across all four directories (paper_secs, knowledge_notes, experiment_notes, idea_notes); up to 5 paths per directory per keyword; scores recorded directly
- [ ] Step 4: All paths deduplicated; unique files read via `obsidian_get_note`; unified context built
- [ ] Step 5: Per-segment explanation with all three dimensions; each with ≥1 structured example + annotations
- [ ] Kernel/pipeline/overlap/batching questions use timeline diagrams
- [ ] Note evidence: vault paths + omnisearch scores. Web evidence: links. Uncertainties declared.
- [ ] Not artificially shortened
- [ ] Step 6: Paragraph synthesis with Mermaid flowchart + integrated understanding
