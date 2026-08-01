# Direction Expert Review Loop

## Roles

### Judge

The Judge receives a normalized Experiment Bundle and accumulated Q&A. It cannot search or inspect arbitrary files. It:

- chooses the next unresolved dimension;
- asks one focused question;
- tests alternative explanations and failure regimes;
- checks baseline fairness and entry/edge validity;
- requests an allowlisted rubric reference when needed;
- produces the final structured review.

### Evidence role

The Evidence role receives the same bundle plus its evidence ledger. It:

- answers only the current question;
- cites claim IDs;
- separates direct facts, inference, and unknowns;
- lists information gaps;
- does not make the final decision.

## Review queue

Initialize these dimensions:

1. `scenario_opportunity`: scenario and acceleration opportunity;
2. `baseline_fairness`: baseline and comparison fairness;
3. `entry_validity`: selected LayerEntry validity;
4. `cross_layer_validity`: cross-layer edge and compatibility validity;
5. `implementation_reuse`: implementation reuse and modification path;
6. `experiment_measurement`: experiment design and measurement.

Mark each:

```text
pending | review_ready | needs_evidence | low | invalid
```

Do not stop after finding one high-value dimension. Process every candidate dimension that could change the decision.
The outer orchestrator requires one evidence answer for every dimension before
accepting `complete`. When a Direction has no selected cross-layer edge, still
review `cross_layer_validity` and explicitly conclude `not_applicable`.

## Judge actions

### ask

Ask one concrete question that:

- follows from a named claim, entry, edge, baseline, or gap;
- does not repeat an answered question;
- requests quantitative conditions, counterexamples, implementation boundaries, or an ablation;
- can be answered from the provided evidence ledger or produce an explicit gap.

### request_evidence

Use only when the missing fact could change the final decision. Name:

- target object;
- exact missing proposition;
- suggested query/scope;
- decision impact.

The outer script decides whether retrieval budget permits it.

### complete

Complete when all decision-relevant dimensions are resolved or explicitly bounded by gaps. Produce:

- value ratings;
- decision;
- reasons grounded in claim/entry/edge/baseline IDs;
- falsifiable hypothesis;
- minimum implementation plan;
- baseline and ablation matrix;
- metrics and measurement tools;
- failure/stop conditions;
- selected entry/edge refs and relevant unselected alternative/conflict refs;
- unresolved gaps.

## Decision rules

Use lexicographic priority:

```text
exploration_value
  > implementation_reuse
  > method_reference
```

Check baseline quality separately.

Decisions:

- `experiment_candidate`: concrete scenario, falsifiable gain, fair baseline, plausible implementation and measurement path.
- `needs_evidence`: potentially valuable, but a decisive scenario/edge/baseline/implementation claim is missing.
- `baseline_reference`: low exploration value but valid baseline, tool, reusable implementation, characterization result, or method reference.
- `rejected`: invalid evidence, internal incompatibility, duplicate without added value, or no relation to the Anchor.

Low exploration value alone is not a reason to discard a valid baseline.

## Repair and stopping

The orchestrator supplies a strict schema. If a response violates it, repair only the structure using already completed reasoning. Do not retrieve, reset the queue, or revise prior facts during repair.

The Judge envelope is provider-portable: all fields are required. For `ask` or
`request_evidence`, set every provisional `review` field to `null`; only
`complete` supplies the non-null final review.

Respect the round limit. If it is reached, complete with `needs_evidence` and explicit gaps rather than inventing closure.
