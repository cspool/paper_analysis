# `work-result-direction-v2`

Return one JSON object and no surrounding text:

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {
    "name": "short distinctive Direction name",
    "mechanism": "causal mechanism and modifiable object",
    "baselineChange": "the single primary change from the bound baseline",
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
- `READY_FOR_REVIEW` has `unresolved=[]`; the other two outcomes have at least
  one unresolved item.
- `expectedEffects` contains at least one effect and covers every
  goal-defined metric or guardrail material to the claim.
- `baselineChange` names one minimal testable primary change. Required setup,
  instrumentation, and implementation support are enablers, not additional
  claimed changes.
- A joint package is permitted only when its components are technically
  inseparable for the intervention. Explain that boundary in `mechanism` or
  `baselineChange` and claim only package-level effects. Independently
  toggleable changes must not be bundled for an unattributable claim.
- `measurementPlan` first reproduces the baseline, then compares the frozen
  baseline with the primary change under material controls and falsifiers.
  State compact deterministic generation or sampling rules instead of
  enumerating future manifests and per-sample details.
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
