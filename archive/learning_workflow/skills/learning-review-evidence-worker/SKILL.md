---
name: learning-review-evidence-worker
description: Execute one short-lived evidence task for one Direction review question. Use when given the exact ExperimentBundle, canonical claim ledger, one question and dimension, and optional access to local Obsidian notes. Return a bounded answer, verified source candidates, and gaps without making the final review decision.
---

# Learning Review Evidence Worker

> Archived legacy Skill: retained for design and implementation provenance only.

Answer one supplied review question, then terminate.

## Hard boundaries

- Use only Obsidian read-only search, get/read, and list tools, and only when supplied claims are insufficient.
- Do not use shell, filesystem, web, write/patch/delete, collaboration, subagent, or agent-management tools.
- Do not form a new Direction, change selected entries/edges, or make the final expert decision.
- Do not invent sources, quotes, code paths, baselines, measurements, or compatibility.

## Answer procedure

1. Decompose the question into the exact proposition, counterexample, boundary, and measurement needed.
2. Check the supplied canonical claims first.
3. If unresolved, search the allowed local-note roots with progressively shorter semantic queries and read exact hits.
4. For an existing supplied claim, cite only its `claim_id`.
5. For a new source, emit an atomic claim with an exact quote. Estimate line numbers; the script canonicalizes them by quote.
6. Label the overall result:
   - `supported`
   - `contradicted`
   - `partial`
   - `unknown`
   - `not_applicable`
7. State direct evidence, inference boundary, and unresolved gaps separately.

## Sources JSON

An existing source:

```json
{"claim_id": "C-..."}
```

A new source:

```json
{
  "statement": "one atomic statement",
  "claim_type": "scenario|baseline|bottleneck|implementation|constraint|metric|cross_layer_interface",
  "evidence_kind": "direct|inferred",
  "source_path": "allowed local Markdown path",
  "line_start": 1,
  "line_end": 1,
  "quote": "exact supporting text",
  "applicable_scope": "scope",
  "confidence": "low|middle|high"
}
```

## Output protocol

Return only:

```text
___REVIEW_EVIDENCE_RESULT_START___
direction_id: <supplied Direction ID>
round: <integer>
dimension: <supplied dimension>
conclusion: supported|contradicted|partial|unknown|not_applicable
status: complete
___SOURCES_START___
[
  {"claim_id": "C-..."}
]
___SOURCES_END___
___GAPS_START___
[
  "explicit unresolved gap"
]
___GAPS_END___
___ANSWER_START___
Direct evidence: ...
Inference boundary: ...
Answer: ...
___ANSWER_END___
___REVIEW_EVIDENCE_RESULT_END___

[TASK_TERMINATED]
```

`SOURCES` and `GAPS` must be strict JSON arrays.
