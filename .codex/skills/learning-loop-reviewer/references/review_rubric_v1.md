# Anchor and Direction Review Rubric

This rubric defines the technical and evidence checks. The Reviewer Skill
defines the ordered review method and the mutually exclusive verdict rule.

## Shared checks

- The content stays within `workflow_goal.json` and satisfies the current
  Turn Task requirements and constraints.
- Every material factual claim is traceable to evidence under the stated
  conditions; hypotheses are identified as hypotheses.
- Fields are mutually consistent and do not claim readiness while naming an
  unresolved blocker.
- Comparisons control the variables that materially affect the Goal-defined
  primary metrics and guardrails.

## Anchor checks

- The scenario states workload, phase, and regime rather than only a broad
  object family.
- The baseline describes an actual execution path or controlled comparison.
- The performance tension is observable and tied to the Goal-defined
  optimization objective and guardrails.
- The non-empty 6L region identifies concrete performance objects without
  silently narrowing the Topic.
- The evidence supports the scenario, baseline, or performance tension under
  its stated conditions.

## Direction checks

- The Direction remains inside the bound Anchor.
- The primary change is the smallest intervention that can be tested and
  interpreted, and is distinguishable from setup, instrumentation, and other
  frozen enablers.
- Independently toggleable changes are not bundled into one unattributable
  claim. A joint package is acceptable only when its components are
  technically inseparable and the Direction claims package-level effects
  without pretending to attribute each component.
- The baseline change and causal mechanism are explicit and falsifiable.
- Expected effects cover the Goal-defined primary metrics and material
  guardrails under controlled conditions.
- Tradeoffs and material failure or degradation conditions are present,
  including the strongest counterexample supported by the available evidence.
- The measurement plan reproduces the baseline before testing the primary
  change and fixes or stratifies every configuration, workload,
  environment, quality, and service variable that materially affects fairness.
- Implementation and measurement are bounded; a request for new evidence is
  not presented as an established result.
- The expression is minimally sufficient for reproduction and falsification.
  It uses compact deterministic generation or sampling rules where possible;
  exhaustive future manifests and per-sample records do not obscure the single
  claim or make the plan operationally unbounded.

## Blocking threshold and revision ratchet

- A specification gap is blocking only when two reasonable implementations
  could change the principal comparison, pass/fail result, baseline
  reproducibility, primary-change/enabler boundary, material guardrail, or
  causal attribution.
- A detail that can be selected and frozen before experiment execution without
  changing those meanings belongs to a non-blocking future handoff, not a new
  admission condition.
- Do not demand complete manifests, every sample/configuration ID, literal
  hash encodings, unrelated binary64 rounding rules, per-draw bootstrap
  formulas, window tables, traces, or execution scripts by default.
- When `previousReview` is available, use it to check the prior correction
  boundary. Retire resolved blockers. Add a new blocker only for a new
  conclusion-level defect or one genuinely exposed by closing the prior
  defect; added prose detail alone does not justify a stricter next boundary.
