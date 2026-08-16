# Experiment Decision method

## 1. Reconstruct the current decision

Read the original Direction, Parent Anchor, source review, evidence manifest, experiment policy,
current state, runtime envelope, and complete indexed trajectory. Identify:

- the performance gap and competitive baseline that motivate the Direction;
- the unique causal change and immutable correctness, quality, throughput, fairness, and resource
  guards;
- every prior contract, completed result, checkpoint, independent judgment, negative result, and
  external obstacle;
- the actual evidence scope rather than the intended scope;
- whether the newest result has received independent review.

Follow referenced artifacts when summaries conflict. Inspect only enough evidence to understand the
workflow state; Evidence Judge owns scientific audit.

## 2. Apply branch priority

Choose in this order:

1. Use `RUN_JUDGE` for a new unreviewed result or a specifically justified re-review.
2. If the latest Judgment resolves the decisive question, complete, reject, or return to Learning.
3. Use `RUN_LAB` only when a new measurement can resolve a remaining decisive uncertainty.
4. Avoid local regime hunting after fair negative evidence has removed the incremental claim.

Judge assessments are local evidence. Decide separately whether their scope covers the Direction's
final need.

## 3. Select one decisive uncertainty

Ask one question whose answer changes the next branch. Prefer the earliest evidence gate that can:

- establish a correct parent, closest-method, or strongest-simple baseline;
- verify a non-degenerate calibrated policy;
- confirm correctness or quality on independent samples;
- isolate the unique change on the same carrier;
- measure target performance with required guards.

Do not combine calibration, confirmation, performance, new environment deployment, and external
validity into one contract when later inputs depend on earlier results. Split them across top-level
Lab → Judge → Decision cycles.

## 4. Choose fidelity and controlled weakening

Evaluate fidelity separately for baseline, optimization interface, model, data/task distribution,
software behavior, hardware/topology, workload/trace, metrics, quality, and statistics. Select the
highest-causal-fidelity path that can progress with current resources.

Record any weakening and narrow its conclusion. A smaller real model, real subset, shorter trace,
source-installable substitute, minimal local component, weak proxy, or simulator can be valid when
it preserves the tested mechanism. Never weaken a dimension that carries the core claim. Synthetic
task profiles cannot establish real task heterogeneity; simulation cannot establish real GPU
performance.

## 5. Check execution feasibility

Before `RUN_LAB`:

1. Read the Script-provided Lab hard/idle timeout, result reserve, and maximum contract minutes.
2. Use prior measured rates, a pilot, or a conservative estimate for setup, runs, statistics, and
   result packaging.
3. Reserve time for environment failure, checkpointing, and atomic result commit.
4. If no rate is known, request a minimal pilot or calibration contract.
5. If the question cannot finish within one invocation envelope, choose an earlier evidence gate or
   shardable unit instead of an oversized contract.

Set `estimatedMinutes` to a positive estimate no greater than the injected maximum.

## 6. Form one atomic RUN_LAB contract

Write a complete contract with:

- `objective`: one decision-changing uncertainty;
- `comparison`: required baseline, strongest simple comparator, unique variant change, and only the
  necessary ablations;
- `conditions`: carrier, data, workload, metrics, stage entry, statistics, and guards;
- `stopConditions`: non-empty, priority-ordered conditions observable from this contract's output;
- `estimatedMinutes`: conservative total invocation time including result packaging;
- `allowedWeakening`: dimensions Lab may choose within and their boundaries;
- `forbiddenWeakening`: causal dimensions and guards Lab must preserve;
- `completionEvidence`: minimum artifacts for each exit path.

Use this priority:

```text
terminal stop
  > downstream phase entry condition
  > entered-phase required artifacts
  > generic completeness
```

Condition the evidence requirements. For example, if calibration yields a predeclared degenerate
policy, require calibrated raw data, mechanism audit, statistics, frozen policy, exclusions, and the
narrow observation; do not require confirmation or performance. If calibration is non-degenerate,
still end the atomic contract and let the next Decision decide whether confirmation is warranted.

Never define one fact as both a terminal stop and an unconditional reason to continue downstream.

## 7. Select a decision

- `RUN_LAB`: a new atomic measurement is needed; supply the full contract.
- `RUN_JUDGE`: existing evidence needs independent first review or focused re-review; supply only a
  concise focus.
- `COMPLETE_SUPPORT`: competitive baseline, triggered unique behavior, paired attribution, guards,
  and decisive causal dimensions are covered.
- `COMPLETE_REJECT`: valid equivalence, dominance, core failure, or converged fair negative evidence
  removes incremental value.
- `RETURN_TO_LEARNING`: progress requires a different optimization object, Parent Anchor, core
  lever, evidence source, or research claim.
- `BLOCKED`: a specific external impasse prevents any honest in-scope experiment or judgment.

## 8. Output

Return one JSON object using the Script template. Populate only the conditional field required by
the selected decision.
