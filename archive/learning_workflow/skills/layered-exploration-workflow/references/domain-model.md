# Domain Model and Invariants

## Object hierarchy

```text
Topic
├── EvidenceClaim (append-only fact ledger)
├── GlobalEntity (deduplicated method/tool/code/mechanism)
└── Anchor
    ├── BaselineSet
    ├── L1..L6 -> zero or more LayerEntry
    ├── CrossLayerEdge
    └── Direction
        └── ExperimentBundle
            └── ExpertReview
```

The Global Layer Catalog, Anchor Layer Map, and Direction Bundle are projections of these objects. They are not separate sources of truth.

## EvidenceClaim

Required semantic fields:

```json
{
  "claim_id": "C-<stable hash>",
  "statement": "one atomic factual statement",
  "claim_type": "scenario|baseline|method|implementation|constraint|metric|relation|evaluation",
  "layer": "L1|L2|L3|L4|L5|L6|GLOBAL",
  "entity_name": "canonical or provisional name",
  "source_path": "vault-relative or absolute local path",
  "line_start": 1,
  "line_end": 1,
  "quote": "exact supporting text",
  "evidence_mode": "direct|inferred",
  "scope": "conditions under which the statement applies",
  "confidence": "high|middle|low"
}
```

Invariants:

- One claim contains one fact.
- `direct` requires the quote to entail the statement.
- `inferred` requires quoted premises and an explicit inference boundary.
- Path, line range, and normalized quote must match the local source.
- Later stages reference `claim_id`; they never replace its statement with a summary.

## GlobalEntity

Represents a method, system, codebase, tool, measurement technique, or hardware mechanism only once.

```json
{
  "entity_id": "G-<stable hash>",
  "name": "canonical name",
  "entity_type": "method|system|code|tool|hardware|dataset|metric",
  "aliases": [],
  "evidence_refs": []
}
```

An entity can appear in multiple Anchors through different LayerEntries.

## Anchor

An Anchor fixes the comparison context:

```text
workload
× workload phase
× request/shape regime
× backend context
× target bottleneck
× primary baseline
× target metric set
```

Split Anchors when any of those changes enough to make execution paths or metric comparisons materially different.

Required fields:

```json
{
  "anchor_id": "A-<stable hash>",
  "workload": "...",
  "phase": "...",
  "regime": "...",
  "backend": "...",
  "bottleneck": "...",
  "primary_baseline_id": "B-...",
  "target_metrics": [],
  "evidence_refs": [],
  "status": "candidate|active|needs_evidence|rejected"
}
```

Do not use `unknown` as a shared bucket that merges unrelated scenarios. Keep an incomplete candidate Anchor and record its exact gap.

## BaselineSet

Roles:

- `current_practice`: current/default execution path.
- `strong`: strongest fair comparison known for the Anchor.
- `tool_evaluation`: measurement or evaluation reference.
- `reusable_implementation`: code/software asset used as an implementation starting point.

A role may be missing, but the gap must be explicit. A valid baseline remains in output even when it has low exploration value.

## LayerEntry

```json
{
  "entry_id": "E-<stable hash>",
  "entity_id": "G-...",
  "anchor_id": "A-...",
  "layer": "L1|L2|L3|L4|L5|L6",
  "role": "baseline_behavior|opportunity|method|implementation|constraint|evaluation",
  "claim": "one Anchor-specific claim",
  "modifiable_object": "...",
  "applicable_baseline_ids": [],
  "preconditions": [],
  "expected_effect": "...",
  "evidence_refs": [],
  "confidence": "high|middle|low",
  "status": "candidate|accepted|needs_evidence|rejected"
}
```

Invariants:

- One main claim per entry.
- One display row per entry.
- Entry facts resolve to evidence.
- An entry belongs to one Anchor, even if its entity is global.
- Baseline behavior, opportunity, method, implementation, constraint, and evaluation are separate entries when they make separate claims.

## CrossLayerEdge

Allowed relations:

```text
depends_on | enables | controls | consumes | produces
complements | substitutes | conflicts_with | measures
```

Required semantics:

```json
{
  "edge_id": "X-<stable hash>",
  "anchor_id": "A-...",
  "from_entry_id": "E-...",
  "to_entry_id": "E-...",
  "relation": "controls",
  "interface": "concrete data/control/resource interface",
  "compatibility": "compatible|conditional|incompatible|unknown",
  "condition": "...",
  "evidence_refs": [],
  "confidence": "high|middle|low"
}
```

Both endpoints must belong to the same Anchor. `substitutes`,
`conflicts_with`, `incompatible`, or `unknown`-compatibility edges cannot be
selected as a synergistic path. A selected `conditional` edge must state its
condition explicitly.

## Direction

A Direction is a selected, compatible subgraph, not a summary of the Anchor.

```json
{
  "direction_id": "D-<stable hash>",
  "anchor_id": "A-...",
  "title": "...",
  "selected_entry_ids": [],
  "selected_edge_ids": [],
  "baseline_ids": [],
  "hypothesis": "...",
  "expected_effects": [],
  "preconditions": [],
  "ablation_plan": [],
  "evidence_refs": [],
  "gaps": [],
  "kind": "experiment|baseline_reference|implementation_reference|method_reference",
  "status": "candidate|needs_evidence|reviewed|rejected"
}
```

Direction rules:

- Exactly one Anchor.
- Every selected edge endpoint is selected.
- No incompatible edge is used as synergy.
- A one-layer Direction is valid.
- Missing layers are gaps only if its hypothesis depends on them.
- Baselines are explicit and participate in review.

## ExpertReview

Required decision fields:

```text
exploration_value: high|middle|low
implementation_reuse: high|middle|low
method_reference: high|middle|low
baseline_quality: high|middle|low|not_applicable
cross_layer_validity: high|middle|low|not_applicable
experiment_readiness: ready|partial|not_ready
decision: experiment_candidate|needs_evidence|baseline_reference|rejected
```

Trace fields distinguish selected and unselected Anchor objects:

```text
entry_refs / edge_refs
alternative_entry_refs / alternative_edge_refs
baseline_refs / evidence_refs
```

`decision` is not a scalar average. Apply lexicographic priority:

1. exploration value;
2. implementation reuse;
3. method reference.

Baseline validity is checked independently and can yield `baseline_reference` even when exploration is low.
