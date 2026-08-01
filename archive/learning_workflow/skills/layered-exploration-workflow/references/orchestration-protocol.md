# Orchestration Protocol

## Separation of responsibilities

The Python orchestrator owns:

- task graph and phase transitions;
- retrieval execution;
- provider calls and budgets;
- JSON Schema validation;
- stable ID assignment;
- source quote verification;
- state, events, raw call logs, retries, resume;
- referential and graph integrity;
- deterministic rendering.

Model roles own:

- semantic extraction;
- entity/Anchor hypotheses;
- entry and edge interpretation;
- targeted evidence requests;
- counterexample-driven questions;
- value judgment within the supplied rubric.

Models never write canonical artifacts directly.

Prompt-size control operates only at semantic boundaries: DiscoveryTask,
claim batch, Anchor, and Direction. Never truncate a JSON object, evidence
snippet, claim list, Anchor map, or Experiment Bundle in the middle. If a
complete object exceeds provider context, checkpoint and split an earlier
collection boundary or stop with an explicit size gap; do not substitute a
lossy prose summary.

## Phases

```text
initialized
  → discovery
  → curation
  → direction_build
  → direction_review
  → validated
  → rendered
```

Each phase is restartable from canonical artifacts produced by earlier phases.

## Task lifecycle

```text
pending → running → done
                  ↘ failed_retriable
                  ↘ failed_terminal
                  ↘ skipped
```

On resume:

- reset stale `running` tasks to `pending`;
- accept `done` only when its artifact exists and validates;
- never infer completion from a log message alone;
- preserve attempt count and last error.

## Discovery loop

For one `Layer × ValueAxis` task:

1. Build a deterministic seed query from topic, layer terms, value terms, and baseline terms.
2. Retrieve bounded, line-numbered local snippets.
3. Ask the discovery role for atomic candidates and narrower `next_queries`.
4. Validate candidates against sources; append valid claims and quarantine invalid claims.
5. If the evidence threshold is not met and query/round budget remains, retrieve a new packet and repeat.
6. Mark the task terminal with explicit gaps.

Independent discovery tasks may run in a worker pool. Writes to state and shared ledgers must be serialized or atomic.

## Curator loop

Curator actions:

```text
integrate
request_evidence
complete
```

For `integrate`, the output is a proposed set of Anchor batches containing baselines, entries, and edges. The script assigns canonical IDs, validates references, merges exact duplicates, and records rejected mutations.

Use one resumable model session per bounded claim batch, while including the
current canonical catalog/Anchor checkpoint in every turn. Starting a new batch
starts a new session so earlier conversational history cannot grow without
bound; canonical artifacts provide cross-batch continuity.

For `request_evidence`, each request includes:

```text
target object
missing claim
query
preferred source scope
why the answer changes a decision
```

The script enforces a bounded query/round budget and sends validated new claims back to the curator. A request that repeats a normalized query is rejected as a no-progress loop.

`complete` requires explicit unresolved gaps. Completion does not imply that all six layers are populated.

## Direction build loop

Run per Anchor, never across mixed Anchor contexts.

1. Supply all accepted/candidate entries and edges for that Anchor.
2. Ask for one or more compatible subgraphs.
3. Validate edge endpoints and compatibility.
4. Separate experiment candidates from baseline/implementation/method reference bundles.
5. Request targeted evidence only for a named missing edge, baseline, or precondition.

Do not auto-create edges merely because two entries occur in adjacent layers.

## Direction review loop

Use two isolated histories:

- Judge history contains normalized bundle, questions, answers, and requested reference keys.
- Evidence history contains bundle evidence ledger and the current question.

The broker validates round numbers and allowed action types before routing. One retry may repair structure but may not request new evidence or revise earlier answers.

By default, those histories are local JSON state, and each provider turn is
stateless: the script supplies the full Bundle plus accumulated Q&A. This bounds
provider-side context growth while preserving every turn. Provider `resume` is
an optional transport optimization, never the only copy of history.

## Checkpoint and event log

`state.json` is the latest compact state. `events.jsonl` is append-only and records:

```text
timestamp
phase
task/direction ID
event type
attempt
input/output artifact paths
provider/model/response ID where applicable
validation result
error summary
```

Write JSON artifacts atomically using a temporary file in the same directory followed by rename.

## Budgets and stopping

Bound at least:

- discovery retrieval rounds per task;
- snippets and characters per evidence packet;
- model attempts per action;
- curator evidence requests;
- review rounds per Direction;
- output tokens per call;
- optional API cost/usage policy.

Stop a loop when:

- its completion contract passes;
- the next action repeats without new evidence;
- budget is exhausted, in which case preserve exact gaps;
- deterministic validation fails repeatedly.
