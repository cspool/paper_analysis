# `work-result-anchor-v2`

Return one JSON object and no surrounding text:

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {
    "name": "short distinctive Anchor name",
    "scenario": "concrete workload, execution phase, and operating regime",
    "baseline": "current execution path or controlled comparison",
    "performanceTension": "observable optimization tension",
    "scope6L": {
      "L1": "algorithm or pipeline region involved in this Anchor",
      "L2": "serving or runtime region involved in this Anchor",
      "L3": null,
      "L4": null,
      "L5": null,
      "L6": null
    },
    "constraints": ["applicability condition or goal-defined guardrail"]
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
- `content` is the complete Anchor object for `READY_FOR_REVIEW` and
  `PARTIAL_RESULT`.
- `content` is `null` for `BLOCKED_NO_RESULT`.
- For a bounded `CREATE_ANCHOR` exploration, `BLOCKED_NO_RESULT` may report
  that no non-duplicate, source-supported Anchor can honestly be formed.
  `unresolved` then summarizes the bounded search coverage, duplicate or
  unsupported candidate routes, and the reason no object is returned. Tool
  failure or avoidably narrow coverage must be reported as such rather than as
  Topic saturation.
- `READY_FOR_REVIEW` has `unresolved=[]`; the other two outcomes have at least
  one unresolved item.
- `scope6L` always contains the six named keys `L1` through `L6`. Use one
  concise region/object description for each involved layer and `null` for
  each uninvolved layer. At least one layer is non-null.
- The complete shape and cross-field rules are the Worker content contract.
  Reviewer checks them and Decision decides whether a material violation
  requires semantic retry.
- The Script only requires a parseable JSON object and a legal
  `workOutcome`. It records deviations from this template as non-blocking
  audit advisories.
- All arrays are present. Do not include IDs, revisions, scheduling fields, or
  workflow-control fields.
