---
name: learning-anchor-curator-worker
description: Convert one bounded packet of already-verified EvidenceClaims into a short-lived semantic delta containing precise Anchor signatures, retained baselines, atomic L1-L6 entries, global entities, and entry-level edges. Use when no knowledge-base access is needed and the orchestrator will assign canonical IDs and validate every object.
---

# Learning Anchor Curator Worker

> Archived legacy Skill: retained for design and implementation provenance only.

Curate one verified claim packet, then terminate. Use only the supplied claims and canonical IDs.

## Hard boundaries

- Use no tools.
- Do not inspect files, search, delegate, launch agents, manage state, or assign canonical IDs.
- Do not alter quotes or create facts absent from the verified claim packet.
- Do not discard a valid baseline because exploration value is low.
- Do not combine incompatible scenarios or methods to make an apparently richer object.

## Object rules

### Anchor

Split Anchors by:

`workload × phase × regime × backend × bottleneck × primary baseline execution path × target metrics`

If one of these changes enough to make comparison unfair, create a different Anchor candidate.

### Baseline

Attempt to preserve independently:

- `current_practice`
- `strong_comparison`
- `tool_evaluation`
- `reusable_implementation`

Missing kinds become gaps. Never fabricate them.

### LayerEntry

- One main claim per entry.
- A layer can contain zero to many entries.
- Valid roles: `baseline_behavior`, `opportunity`, `method`, `implementation`, `constraint`, `evaluation`.
- State the concrete modifiable object when evidence provides one.
- Link only verified claim IDs.

### CrossLayerEdge

- Connect exact entries, never only `L2 → L4`.
- Valid relations: `controls`, `depends_on`, `enables`, `complements`, `conflicts`, `substitutes`, `incompatible`.
- Mark a relation `conditional` only with an explicit condition.
- Preserve conflict/substitution; do not relabel it synergy.

## Payload shape

Use strict JSON:

```json
{
  "entities": [
    {
      "kind": "method|implementation|tool|software|hardware|other",
      "name": "global entity name",
      "description": "bounded description",
      "evidence_refs": ["C-..."]
    }
  ],
  "anchors": [
    {
      "local_id": "a1",
      "title": "short discriminative title",
      "scenario": "specific scenario",
      "signature": {
        "workload": "...",
        "phase": "...",
        "regime": "...",
        "backend": "...",
        "bottleneck": "...",
        "primary_baseline_execution_path": "...",
        "target_metrics": ["..."]
      },
      "evidence_refs": ["C-..."],
      "baselines": [
        {
          "local_id": "b0",
          "kind": "current_practice|strong_comparison|tool_evaluation|reusable_implementation",
          "name": "...",
          "execution_path": "...",
          "implementation": "...",
          "comparison_scope": "...",
          "evidence_refs": ["C-..."],
          "exploration_value": "low|middle|high|unknown"
        }
      ],
      "entries": [
        {
          "local_id": "e1",
          "entity_id": "",
          "entity_name": "",
          "layer": "L1|L2|L3|L4|L5|L6",
          "role": "baseline_behavior|opportunity|method|implementation|constraint|evaluation",
          "claim": "one atomic claim",
          "modifiable_object": "...",
          "applicable_baselines": ["b0"],
          "preconditions": ["..."],
          "expected_effect": "...",
          "evidence_refs": ["C-..."],
          "confidence": "low|middle|high"
        }
      ],
      "edges": [
        {
          "from_entry": "e1",
          "to_entry": "e2",
          "relation": "controls|depends_on|enables|complements|conflicts|substitutes|incompatible",
          "interface": "specific data/control/resource interface",
          "compatibility": "compatible|conditional|conflict",
          "condition": "...",
          "evidence_refs": ["C-..."],
          "confidence": "low|middle|high"
        }
      ],
      "gaps": ["..."]
    }
  ],
  "dispositions": []
}
```

## Output protocol

```text
___ANCHOR_DELTA_START___
task_id: <supplied task id>
status: complete
___SEMANTIC_PAYLOAD_START___
{ "...": "payload above" }
___SEMANTIC_PAYLOAD_END___
___ANCHOR_DELTA_END___

[TASK_TERMINATED]
```

Return only the protocol.
