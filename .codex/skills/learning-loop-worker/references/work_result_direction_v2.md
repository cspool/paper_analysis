# `work-result-direction-v2`

Return one JSON object and no surrounding text:

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {
    "name": "short distinctive Direction name",
    "mechanism": "causal mechanism and modifiable object",
    "baselineChange": "competitive simple baseline, its fair selection boundary, and the single substantive primary difference with an expected behavioral discriminator",
    "expectedEffects": [
      {
        "metric": "goal-defined metric or guardrail",
        "effect": "expected measurable effect",
        "conditions": "conditions under which the effect should hold"
      }
    ],
    "tradeoffs": ["cost, degradation, or competing objective"],
    "failureConditions": ["condition under which the expected gain disappears"],
    "measurementPlan": ["controlled comparison or falsification step"]
  },
  "evidence": [
    {
      "sourceRef": "vault/path.md#heading",
      "supports": "one bounded statement supported by that source"
    }
  ],
  "unresolved": []
}
```

Rules:

- `workOutcome` is `READY_FOR_REVIEW`, `PARTIAL_RESULT`, or
  `BLOCKED_NO_RESULT`.
- `content` is the complete Direction object for `READY_FOR_REVIEW` and
  `PARTIAL_RESULT`.
- `content` is `null` for `BLOCKED_NO_RESULT`.
- `BLOCKED_NO_RESULT` is appropriate when reviewed negative EXP history closes
  the available causal lever and a bounded search finds no independently
  supported, materially different Direction inside the bound Anchor. Summarize
  the failed boundary and alternatives checked in `unresolved`; do not rename
  a threshold, signal, feature, or small head as a new mechanism.
- `READY_FOR_REVIEW` has `unresolved=[]`; the other two outcomes have at least
  one unresolved item.
- `expectedEffects` contains at least one effect and covers every
  goal-defined metric or guardrail material to the claim.
- `baselineChange`, `mechanism`, and evidence together identify the parent
  execution baseline, closest method baseline, and strongest simple baseline
  needed to exclude the most credible simpler explanation. They state how the
  decisive baseline receives a fair bounded calibration over its complete
  legal domain, the single substantive primary change, and an event class in
  which that change should produce an observable action, state, or execution
  path difference. Do not add duplicate baseline fields.
- Required setup, instrumentation, compatibility work, and implementation
  support are enablers, not additional claimed changes. A renamed method, an
  artificially weak default, or a complex policy behaviorally equivalent to a
  simpler calibrated policy is not a new Direction.
- A joint package is permitted only when its components are technically
  inseparable for the intervention. Explain that boundary in `mechanism` or
  `baselineChange` and claim only package-level effects. Independently
  toggleable changes must not be bundled for an unattributable claim.
- `failureConditions` treats behavioral equivalence or dominance by the
  strongest simple baseline as failure whenever the claim depends on added
  information or policy complexity. If a different trigger regime can resolve
  apparent equivalence, it must be bounded before confirmatory results rather
  than selected afterward.
- `measurementPlan` first reproduces and validates the baseline; fairly
  calibrates the strongest simple baseline on calibration-only data; verifies
  trigger coverage and relevant action/state/path divergence; then compares
  the frozen baseline and variant in one same-carrier paired ablation. Request
  broader sensitivity, native performance, simulator envelopes, or paper
  external validity only after the unique change is exercised. State compact
  deterministic rules instead of enumerating future manifests and per-sample
  details.
- Once the Direction is technically valid, `measurementPlan`, `tradeoffs`, and
  evidence identify the closest reusable baseline/reference experiment,
  relevant code/framework/simulator/profiler/benchmark/trace/hardware
  environment, its coverage limits, and the minimum adaptation for an EXP
  Goal. A bounded, explicit "no direct reusable environment found" handoff is
  preferable to fabricated detail.
- The JSON is a minimally sufficient research Direction, not a complete future
  experiment log. Preserve information needed to reproduce or falsify the
  claim while leaving generated artifacts to a later handoff.
- Complete request manifests, enumerated sample/configuration IDs, literal
  hash encodings, unrelated binary64 rounding rules, per-draw bootstrap
  algorithms, window tables, traces, and execution scripts stay in a future
  experiment handoff unless the particular detail changes the claim,
  comparison, attribution, guardrail, or pass/fail semantics.
- The complete shape and cross-field rules are the Worker content contract.
  Reviewer checks them and Decision decides whether a material violation
  requires semantic retry.
- The Script only requires a parseable JSON object and a legal
  `workOutcome`. It records deviations from this template as non-blocking
  audit advisories.
- All arrays are present. Do not include IDs, revisions, scheduling fields, or
  workflow-control fields.
