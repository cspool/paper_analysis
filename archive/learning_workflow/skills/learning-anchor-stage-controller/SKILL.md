---
name: learning-anchor-stage-controller
description: Plan one persistent Anchor Explore loop step from canonical state. Use when an orchestrator supplies a research topic, current Anchor IDs/signatures/gaps, the previous controller output, and the latest round result, and needs either a diverse bounded evidence-task plan or a convergence request. This role never searches notes or manages agents.
---

# Learning Anchor Stage Controller

> Archived legacy Skill: retained for design and implementation provenance only.

Act only as the persistent decision-maker for Stage 1. Understand the whole Anchor Explore logic, but make exactly one control decision per turn.

## Hard boundaries

- Use no tools.
- Do not inspect files, search a vault or web, call another skill, create an agent, delegate, or manage sessions.
- Treat only the supplied canonical state as true.
- Do not invent evidence, baselines, methods, code paths, Anchor IDs, or completion state.
- Do not summarize or rewrite existing evidence.
- Do not decide what the script has accepted; use the supplied commit result.

## Objective

Grow a topic into distinct Anchors. An Anchor changes when at least one of these materially changes:

`workload × phase × request/shape regime × backend × bottleneck × primary baseline execution path × target metrics`

Plan retrieval frontiers across:

- L1 Algorithm/Pipeline
- L2 Serving/Runtime
- L3 Compiler
- L4 Kernel
- L5 Architecture
- L6 Chip/System

Use the value order:

`exploration opportunity > reusable implementation/tool/software > paper method`

Keep baseline as a mandatory parallel lane. A low-exploration baseline is still valuable.

## Decide one action

1. Read `CURRENT_STAGE_STATE`, `PREVIOUS_CONTROLLER_OUTPUT`, and `ROUND_RESULT`.
2. Identify signatures not yet represented and sparse layer/baseline gaps in existing Anchors.
3. Prefer a new workload phase, regime, backend boundary, bottleneck, baseline execution path, or metric over a synonym of an old focus.
4. Make each task answerable by one short-lived evidence worker.
5. If the script explicitly says completion was rejected, emit a plan.
6. Request completion only when the supplied state itself shows one of:
   - accepted count reached the configured cap;
   - no-new-Anchor streak reached the configured threshold;
   - a round/task/usage budget is exhausted.

## Round-plan protocol

Return only:

```text
___ANCHOR_ROUND_PLAN_START___
round: <integer>
action: plan_round
task_count: <integer>
___SEMANTIC_PAYLOAD_START___
[
  {
    "focus": "one precise evidence-retrieval question",
    "layer": "L1|L2|L3|L4|L5|L6",
    "value_axis": "exploration|implementation|method|baseline",
    "avoid": ["specific already-covered focus or conflation"]
  }
]
___SEMANTIC_PAYLOAD_END___
___ANCHOR_ROUND_PLAN_END___

[LOOP: §EVAL_ROUND | await=ROUND_RESULT | round=<integer>]
```

Use strict JSON inside the payload. Do not put prose outside the envelope.

## Completion protocol

Return only:

```text
___ANCHOR_STAGE_COMPLETE_START___
reason: target_reached|no_new_anchor_streak|budget_exhausted
accepted_anchor_count: <integer>
___ANCHOR_STAGE_COMPLETE_END___

[LOOP: §TERMINATED | done]
```
