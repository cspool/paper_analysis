# `review-result-v2`

Return one JSON object and no surrounding text:

```json
{
  "reviewVerdict": "REVISE",
  "summary": "short independent judgment",
  "findings": [
    {
      "severity": "BLOCKING",
      "issue": "specific defect",
      "basis": "evidence source or cross-field basis",
      "expected": "bounded correction or disposition"
    }
  ],
  "queryGaps": [
    {
      "question": "one object-local unanswered question",
      "dimension": "experiment",
      "reason": "why answering it may change a finding or verdict"
    }
  ]
}
```

## Literal values

`reviewVerdict` is one of:

```text
PASS
REVISE
REJECT
```

Use the mutually exclusive verdict rule in `SKILL.md`.

`severity` is one of:

```text
BLOCKING
NON_BLOCKING
```

`dimension` is exactly one resolution channel:

- `experiment`: the question requires missing measurements, traces, benchmark
  results, or other empirical observations.
- `idea`: the question requires candidate methods, baselines, alternatives, or
  performance tensions from idea-oriented sources.
- `knowledge`: the question requires established concepts, mechanisms,
  implementation facts, or technical relations.
- `human`: the question requires user scope, authorization, preference, or
  expert judgment that available evidence cannot determine.

If one broader uncertainty requires multiple channels, split it into multiple
bounded query-gap objects. Do not return a channel array.

## Semantic constraints

- `PASS` has no `BLOCKING` finding.
- `REVISE` and `REJECT` each have at least one `BLOCKING` finding.
- Reviewer owns these constraints. Decision independently checks their
  consistency with the Work Result, Task, Goal, and actual findings.
- The Script only requires a parseable JSON object and a legal
  `reviewVerdict`. It records deviations from this template as non-blocking
  audit advisories.
- A query gap that could affect admission to the final result has a
  corresponding `BLOCKING` finding.
- `queryGaps` is scoped to the one reviewed object. An empty list is not a
  claim that the Topic is saturated or that the workflow should finish.
- All arrays are present, including empty arrays.
- Do not include scheduling or workflow-control fields.
