---
name: layered-exploration-workflow
description: Build or operate the standalone evidence-preserving research workflow that turns a topic and a local paper knowledge base into Anchor-specific L1-L6 intervention maps, compatible cross-layer Directions, baseline/reference registries, and expert-reviewed experiment bundles. Use when creating, running, resuming, validating, or explaining scripts/layered_exploration_orchestrator.py or its output; do not use it to modify the legacy learning_scheduler.ts, idea_review_orchestrator.ts, or their skills.
---

# Layered Exploration Workflow

## Purpose

Use this skill to construct experiment directions from a broad local knowledge space without lossy horizon/vertical summaries. Treat legacy scripts and skills only as provenance: the standalone Python orchestrator and this skill own the new protocol.

The output priority is:

```text
explorable scenario / acceleration opportunity
  > reusable implementation, code, tool, or software asset
  > paper method reference
```

Valid baselines are a mandatory parallel lane. Keep them in final output and use them in later comparison even when their `exploration_value` is low.

## Non-negotiable Boundaries

- Never patch, import, or invoke `scripts/learning_scheduler.ts` or `scripts/idea_review_orchestrator.ts`.
- Never patch their legacy `.claude/skills/*` dependencies.
- Never use `idea_brainstorm` or ideastorm as expert knowledge.
- Never replace atomic evidence with a prose summary.
- Never force a Direction to cover all L1-L6 layers.
- Never treat all entries in one Anchor as one compatible Direction.
- Never accept an AI-generated citation until path, line range, and quote pass deterministic validation.

## Route by Task

### Explain or inspect the design

Read:

1. [domain-model.md](references/domain-model.md)
2. [layer-value-rubric.md](references/layer-value-rubric.md)
3. [source-provenance.md](references/source-provenance.md)

### Initialize, run, or resume a workflow

Read:

1. [orchestration-protocol.md](references/orchestration-protocol.md)
2. [output-contract.md](references/output-contract.md)
3. [deepseek-provider.md](references/deepseek-provider.md) for the default provider, or [codex-provider.md](references/codex-provider.md) for the optional Codex CLI adapter

Then use `scripts/layered_exploration_orchestrator.py`. Start with `doctor`, then `init` or `run`. Use `--resume` only against an existing compatible state.

### Change a model role or prompt

Load only the role reference being changed:

- discovery/evidence extraction: [discovery-worker.md](references/discovery-worker.md)
- Anchor/entry/edge curation: [curator-loop.md](references/curator-loop.md)
- Direction expert review: [direction-review-loop.md](references/direction-review-loop.md)

Keep output semantics synchronized with the deterministic schemas in the Python script.

### Validate or render an existing run

Read [output-contract.md](references/output-contract.md), then run `validate` before `render`. Rendering must be deterministic and must not call a model.

## Workflow

### 1. Diagnose

Run:

```bash
python3 scripts/layered_exploration_orchestrator.py doctor
```

Interpret provider readiness separately from local workflow readiness. The default provider uses the existing `claude` CLI plus the configured Anthropic-compatible DeepSeek endpoint. The optional Codex CLI provider uses the local Codex login. Never print credentials. Direct OpenAI API readiness is informational only and does not affect this workflow.

### 2. Initialize the task graph

Create one run directory for one topic. Initialization writes a versioned config and expands:

```text
L1-L6 × exploration|implementation_reuse|method_reference
```

into independent discovery tasks. Baseline questions are cross-cutting in every task.

### 3. Discover atomic evidence

Run independent workers over bounded, line-numbered evidence packets. Workers may return atomic candidates and narrower follow-up queries. The script:

- validates every quote against its local source;
- assigns stable claim IDs;
- appends valid claims without overwriting earlier claims;
- isolates invalid claims for audit;
- checkpoints after every task transition.

### 4. Curate Anchor maps

The curator incrementally converts validated claims into:

- `GlobalEntity`;
- exact `Anchor`;
- `BaselineSet`;
- atomic `LayerEntry`;
- entry-level `CrossLayerEdge`.

The curator may request targeted evidence. The outer script executes bounded retrieval and re-enters the curator loop. Deterministic validation, not model confidence, decides whether an action can mutate the run state.

### 5. Form Directions

For each Anchor, select compatible entry subgraphs. A Direction:

- belongs to exactly one Anchor;
- names selected entries and entry-level edges;
- carries the applicable BaselineSet;
- separates substitutes/conflicts from complements/dependencies;
- states hypotheses, expected effects, preconditions, ablations, and gaps.

Baseline-only or tool/reference bundles remain visible even when they are not experiment candidates.

### 6. Review each Direction

Use an isolated Judge/Evidence loop:

- Judge sees the normalized Experiment Bundle and accumulated Q&A, but cannot retrieve.
- Evidence role sees only the bundle plus its evidence ledger and answers the current question.
- Judge can `ask`, `request_evidence`, or `complete`.
- One round focuses on one unresolved dimension.
- A complete result must distinguish `experiment_candidate`, `needs_evidence`, `baseline_reference`, and `rejected`.

### 7. Validate and render

Validation checks referential integrity, evidence quotes, Anchor membership, graph edges, baseline coverage, decision fields, and completion state. Rendering creates the Global Catalog, one L1-L6 table per Anchor, Direction Bundles, baseline/reference registry, reviews, and unresolved gaps.

## Provider Policy

Use the same runtime path as the validated legacy scripts:

```text
Python orchestrator
  → claude CLI
  → --model deepseek-v4-flash[1m]
  → ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
  → stream-json or JSON result
```

Use CLI `--json-schema` plus local semantic validation. Discovery uses isolated
sessions; bounded curator and Direction-build loops can save session IDs and
use `--resume`. Disable model tools because retrieval and writes belong to the
outer script.
Curator/direction turns may resume within their bounded batch/Anchor loop.
Direction review defaults to stateless provider turns rebuilt from the canonical
Bundle and full local Q&A ledger; `--review-session-mode resume` is available
only when provider-history continuity is worth its cumulative context cost.

The optional `codex-cli` adapter uses `codex exec --json --output-schema` and `codex exec resume`. It must run in an isolated provider directory with a read-only sandbox and reject model turns that emit tool-call events. It is a sibling provider, not a subprocess launched by a Claude Agent.

Do not silently switch providers or fall back to fabricated output. The fixture provider is for tests only and every fixture artifact must be labeled synthetic.

## Completion Standard

A workflow is complete only when:

- all non-skipped discovery tasks are terminal;
- all retained claims pass source validation;
- every entry and edge resolves to existing IDs;
- every Direction is a valid subgraph within one Anchor;
- every retained Direction has a review or explicit pending reason;
- all BaselineSets and baseline/reference-only results are rendered;
- deterministic `validate` succeeds.
