# Output Contract

## Run layout

```text
<work-dir>/
├── config.json
├── state.json
├── events.jsonl
├── tasks/
│   └── discovery/
├── retrieval/
├── evidence/
│   ├── claims.jsonl
│   └── rejected_claims.jsonl
├── catalog/
│   └── entities.json
├── anchors/
│   └── anchors.json
├── directions/
│   └── directions.json
├── reviews/
│   ├── <direction-id>.json
│   └── <direction-id>.md
├── logs/
│   └── model_calls/
├── validation.json
└── final.md
```

Directories may be empty before their phase runs, but canonical file names are stable.

## Canonical versus rendered artifacts

Canonical:

- JSON/JSONL facts and state;
- source quote coordinates;
- stable IDs and references;
- structured review decisions.

Rendered:

- `reviews/*.md`;
- `final.md`.

Rendering is deterministic, read-only with respect to canonical objects, and makes no provider calls.

## final.md sections

1. Run metadata and scope.
2. Outcome index ordered by:
   - experiment candidates;
   - needs-evidence candidates;
   - implementation/baseline references;
   - method references;
   - rejected items in an audit appendix.
3. Baseline/Reference Registry.
4. Global L1-L6 Catalog.
5. One Anchor section per Anchor:
   - exact Anchor context;
   - BaselineSet;
   - L1-L6 table, one row per entry;
   - unselected alternatives and conflicts.
6. Direction Bundles:
   - selected entry/edge subgraph;
   - hypothesis and conditions;
   - baseline/ablation matrix;
   - implementation assets;
   - review decision and evidence refs.
7. Evidence gaps and next retrieval/experiment actions.
8. Provenance and validation summary.

## Required traceability

Every final table row contains IDs. Readers must be able to traverse:

```text
final statement
→ review/direction/entry/baseline
→ EvidenceClaim
→ source path + line range + quote
```

Do not paste all source quotes into `final.md`; link IDs and keep exact excerpts in the evidence ledger.

## Validation

Validation fails when:

- a canonical JSON file cannot parse;
- a stable ID is duplicated with different content;
- a claim quote does not match its source;
- an evidence ref is missing;
- an entry references another Anchor's baseline;
- an edge crosses Anchors or has a missing endpoint;
- a Direction uses an edge without selecting both endpoints;
- a synergistic Direction selects an incompatible/conflict edge;
- a review uses an invalid enum or points to a missing Direction;
- a reviewed experiment candidate has no baseline or hypothesis;
- final rendering omits valid baseline/reference objects.

Warnings, not hard failures:

- a BaselineSet role is absent but recorded as a gap;
- a layer is empty;
- a Direction is single-layer;
- a promising Direction ends as `needs_evidence`.

