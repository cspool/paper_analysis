# Domain proposal rules

## TopicFrame

Preserve the user topic. Keep unknown fields unresolved. A narrowed scope
requires explicit user authorization. Commit only through
`SCRIPT_APPLY_TOPIC_FRAME`.

## SearchNeed

Ask one bounded question with explicit success criteria. Choose one primary and
at most one registered auxiliary knowledge dimension. `targetDimensions`
equals that ordered set exactly. Freeze technical objects, scenario terms,
performance relations, evidence-intent terms, known terms, synonyms, and
excluded source units. The Evidence Reader, not this Turn, forms Q1–Q3.

Success criteria describe evidence facts whose presence would answer the
question. Do not put protocol behavior or fallback instructions in that list:
no “if nothing is found, return not_found/no-delta,” output-format rule, query
accounting rule, or retry rule. `not_found` and its audit requirements already
belong to the Evidence Reader contract; mixing them into success criteria
makes the unanswered-criterion ledger self-contradictory.

The route is a closed registry, not a semantic suggestion:

| SearchIntent | required primary | admitted auxiliary |
|---|---|---|
| `discover_anchor` | `idea` | `human` |
| `define_baseline` | `idea` | `experiment` |
| `find_modification` | `idea` | `knowledge` |
| `explain_mechanism` | `knowledge` | `idea` |
| `find_implementation` | `experiment` | `knowledge` |
| `design_measurement` | `experiment` | `knowledge` |
| `challenge_direction` | `knowledge` | `experiment` or `human` |
| `verify_primary_source` | `paper` | none |

An auxiliary is optional and there can be at most one. `paper` is legal only
as the primary dimension of `verify_primary_source`; it is never an auxiliary.
Do not route by which directory sounds most authoritative.

## SemanticDelta

Target one current object revision. Cite committed result refs. Use semantic
field paths in `changedFields`. Same-source repetition or paraphrase is
`no_semantic_delta`. Commit only through
`SCRIPT_APPLY_SEMANTIC_DELTA`; the Controller consumes results in the same
transaction.

## Direction review request

Bind one Direction revision, one registered purpose, and a registered rubric
binding. Do not pre-fill the Reviewer decision. Use `DIRECTION_REVIEW` through
`REQUEST_EVALUATION`.

## StopCandidate

Bind the current run/topic/canonical revision and all ten fixed StopProof
claims. A candidate is only a proposal for Controller preflight and an
independent Closure Reviewer. It is never completion.
