# Direction Experiment Atomic Contract and Evidence Policy

Policy version: `ATOMIC_DECISION_CONTRACT_V7`.

## Authority and loop

The frozen Direction and Parent Anchor define research identity. The deterministic Script owns
state, immutable pins, contract revisions/hashes, event order, result/checkpoint indexing, and legal
transitions. Experiment Decision owns workflow semantics and freezes one atomic contract. Direction
Lab executes that contract. Evidence Judge independently audits its evidence.

```text
Decision -> one atomic Lab contract -> Judge -> Decision
```

Script does not decide whether a scientific stop condition is true or whether evidence supports the
Direction. It only checks protocol literals, identity, refs, non-empty files, runtime envelope, and
state transitions.

## Atomic Decision contracts

Each `RUN_LAB` asks one uncertainty whose answer can change the next workflow decision. Split
calibration, independent confirmation, performance, new environment deployment, and external
validity across top-level cycles whenever a later stage depends on an earlier result.

A contract defines:

- one objective;
- baseline, strongest simple comparator, unique variant change, and necessary ablations;
- carrier, data, workload, metrics, statistics, stage entry, and guards;
- priority-ordered explicit stop conditions;
- a conservative duration estimate within the Script runtime envelope;
- allowed and forbidden weakening;
- minimum evidence for every exit path.

Use this priority:

```text
terminal stop
  > downstream phase entry
  > entered-phase required artifacts
  > generic completeness
```

Never require downstream confirmation or performance after a valid terminal calibration stop.
Whether to open a later stage belongs to the next Decision after Judge review.

## Experiment Decision

Prefer independent review of unreviewed evidence. Avoid experiments already answered by the
trajectory and avoid regime hunting after fair negative evidence. Before `RUN_LAB`, read the runtime
envelope, use prior rates or a conservative pilot, reserve time for failure/checkpoint/result
packaging, and shrink oversized work to an earlier evidence gate.

A later contract may alter the experimental expression of the same causal lever inside explicit
weakening boundaries. It must not silently change the optimization object, Parent Anchor, core
lever, or immutable guards. Such a scientific change returns to Learning Flow.

## Direction Lab

Lab executes one atomic contract, establishes a correct competitive baseline, implements the unique
variant, records causal-interface traces, and performs only the bounded comparison required by the
objective. Before every expensive arm, sweep, or long command it evaluates the frozen stop
conditions. On a stop, conflict, forbidden weakening, or insufficient packaging time, it preserves
the narrowest reviewable result and exits; it does not decide global support/rejection.

Long work is sharded. Complete shards have raw files, completion markers, and hashes; partial shards
are retained but excluded. Lab atomically maintains a contract-bound checkpoint and atomically
commits `result.md` only when independently reviewable.

Every cycle has isolated mutable source. Shared model/data/environment caches are immutable and
content-addressed; cycle results record revision/hash manifests and local adapters or patches.

## Evidence Judge

Judge first audits any claimed stop condition. When the contract validly requires stopping, omitted
downstream phases are not incompleteness. The resulting claim remains limited to the exact policy,
candidate domain, data, model, thresholds, carrier, and statistics.

Judge reports only:

- `VALID_POSITIVE`;
- `VALID_NEGATIVE`;
- `INCONCLUSIVE`;
- `INVALID`.

Installation failure, unavailable original hardware, broken implementation, or no-trigger invalid
proxy is not scientific negative evidence. A valid local negative does not automatically reject the
whole method family. Decision owns global completion.

## Fidelity and weakening

Use the highest causal fidelity that can progress. Complete paper reproduction is not mandatory.
Explicit weakening can use a smaller real model, real data subset, fewer requests, shorter trace,
modifiable open-source substitute, minimal local component, weak proxy, or simulator. Never weaken
a dimension carrying the core claim. Synthetic evidence cannot establish real task heterogeneity;
simulation cannot establish actual GPU performance.

Evidence scopes are:

- `DESIGN_AUDIT_ONLY`;
- `WEAKENED_PROXY_MECHANISM`;
- `LOCAL_SINGLE_GPU_PERFORMANCE`;
- `SIMULATED_HARDWARE_MECHANISM`;
- `PAPER_EXTERNAL_VALIDITY`.

## Measurement language and uncertainty

Latency excluding media preprocessing is model-path latency, not complete request E2E. Requests/s
from concurrency=1 closed-loop is serial service rate, not production throughput. Serving claims
require an appropriate open-loop or concurrent/continuous-batching carrier.

Small-sample zero differences do not imply zero population uncertainty. Non-inferiority needs a
sample-size or power rationale, a suitable finite-sample bound, calibration/confirmation separation,
and conservative handling of multiple tasks or candidate selection.

## Runtime and recovery

Lab hard timeout is a single provider-invocation watchdog, not a scientific budget. Every invocation
should leave a checkpoint or final result. A valid final result is indexed and sent to Judge
regardless of provider status. A valid checkpoint without a result pauses the run and resumes the
same persistent Goal in a new independently audited invocation window.

Operator pause and signals use a lock-free control request. The Controller interrupts the active
Goal, then adopts a final result or indexes a checkpoint. Provider raw events are stored separately
from compact normative events.
