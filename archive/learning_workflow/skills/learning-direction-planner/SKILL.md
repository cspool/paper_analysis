---
name: learning-direction-planner
description: Persistently plan one Anchor at a time by selecting a connected, compatible subset of its canonical L1-L6 entries and entry-level edges as a falsifiable experimental Direction. Use when supplied an exact Anchor graph, baselines, accepted Direction signatures, prior output, and the script's latest commit result.
---

# Learning Direction Planner

> Archived legacy Skill: retained for design and implementation provenance only.

Advance one Anchor by exactly one proposal-or-completion decision per turn.

## Hard boundaries

- Use no tools and no outside knowledge.
- Do not search, inspect files, delegate, create agents, or manage sessions.
- Use only canonical entry, edge, baseline, and evidence IDs from the input.
- Never treat a rejected proposal as accepted; obey `DIRECTION_COMMIT_RESULT`.
- Do not rewrite the Anchor or invent a missing edge.

## Build a Direction

1. Find one modifiable object with a falsifiable opportunity.
2. Select only entries that belong to the supplied Anchor.
3. A single-entry Direction is valid.
4. For multiple entries, select enough canonical edges to form a connected subgraph.
5. Reject combinations with a declared `conflicts`, `substitutes`, `incompatible`, or `compatibility=conflict` relation.
6. Carry conditional-edge conditions into the hypothesis and ablation plan.
7. Select explicit current/strong/tool/implementation baselines available in the Anchor.
8. State how single-entry baselines, the combined direction, and ablations separate synergy from independent gains.
9. Make the hypothesis measurable with the Anchor's metrics and include degradation/failure boundaries.
10. Differ materially from accepted Directions in selected subgraph or hypothesis.

Do not force L1→L6 coverage. Include a layer only when the causal path needs it.

## Proposal protocol

Return only:

```text
___DIRECTION_PROPOSAL_START___
anchor_id: <supplied Anchor ID>
proposal_index: <integer>
___SEMANTIC_PAYLOAD_START___
{
  "selected_entry_ids": ["E-..."],
  "selected_edge_ids": ["X-..."],
  "baseline_ids": ["B-..."],
  "hypothesis": "falsifiable hypothesis with condition and expected metric effect",
  "ablation_plan": ["primary baseline", "entry alone", "combined direction"],
  "implementation_plan": ["smallest concrete modification"]
}
___SEMANTIC_PAYLOAD_END___
___DIRECTION_PROPOSAL_END___

[LOOP: §EVAL_DIRECTION | await=DIRECTION_COMMIT_RESULT | anchor_id=<Anchor ID>]
```

## Completion protocol

Use only when no significantly different compatible subgraph remains, or the supplied budget is exhausted:

```text
___DIRECTION_PLANNING_COMPLETE_START___
anchor_id: <supplied Anchor ID>
reason: exhausted_distinct_subgraphs|budget_exhausted
___DIRECTION_PLANNING_COMPLETE_END___

[LOOP: §TERMINATED | done]
```
