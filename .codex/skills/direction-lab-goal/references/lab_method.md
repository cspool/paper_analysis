# Direction Lab execution method

## 1. Bind and restore

Read the cycle binding and verify the contract revision, path, hash, source Decision, output path,
checkpoint path, isolated source path, and frozen Direction. If a checkpoint exists, verify its
binding and resume only its incomplete units. Do not reconstruct progress from conversation history.

Translate the atomic contract into a checklist with separate exits for:

- each ordered stop condition;
- ordinary atomic completion without a stop;
- invalid implementation or measurement;
- contract conflict or forbidden weakening;
- insufficient time or external obstruction.

## 2. Prepare isolated, reusable execution state

Modify code only under the cycle-specific source directory. Pin the base repository revision and
store the cycle patch, build, and configuration. Do not mutate source used by a prior cycle.

Reuse models, datasets, and base environments from the shared cache only through revision/content
hash manifests. Keep cache objects immutable. Store cycle-local adapters, patches, selected-file
manifests, and configurations in the cycle workspace.

Read Direction evidence and relevant local `experiment_notes/` or `human_notes/` when they reduce
deployment or measurement uncertainty. Prefer an official or source-installable implementation;
otherwise construct only the controller, queue, cache path, model path, simulator event, or
measurement interface required by the contract.

## 3. Establish baseline and variant

For every decisive baseline:

- pin repository, environment, model/data, configuration, and command;
- verify outputs, invariants, legal parameter domain, and guards;
- give the strongest simple comparator the frozen fair calibration opportunity;
- distinguish reproduction, local equivalent, weak proxy, and simulator;
- preserve setup failure rather than silently substituting a new claim.

Implement only the contract's causal lever in the variant. Pair non-causal code paths and inputs.
Instrument the nearest interface: trigger counts, actions, state transitions, execution paths,
queue/cache behavior, resource allocation, communication, or another contract observable.

## 4. Execute one atomic stage

Run only the stage needed to answer the contract objective. Use identical calibration data and
resource constraints for compared methods. Freeze selected parameters before held-out evidence.
For paired tests, hold source/build, resources, model, precision, input, arrivals, seeds, warmup,
metrics, and guards fixed except for the unique change.

Do not begin confirmation when the contract asks only whether calibration is non-degenerate. Do not
begin performance when the contract asks only whether held-out correctness passes. The top-level
Loop defines later stages after Judge review.

## 5. Run the Stop Gate

Evaluate the ordered `stopConditions`:

1. after baseline/variant validation;
2. after every arm or fixed shard group;
3. before every full sweep, high-cost command, or downstream arm;
4. after a substantive failure or contract conflict;
5. before the result-reserve portion of the invocation.

When a condition is observed, stop expensive work, preserve the exact supporting artifacts, exclude
partial units, and commit the narrowest result. This is contract execution, not a global scientific
verdict.

If stop conditions conflict with required artifacts, forbidden weakening would be necessary, or
remaining time cannot produce a minimum review package, preserve the conflict and completed evidence
as a reviewable result. Do not choose the more expensive or favorable interpretation.

## 6. Shard and checkpoint long work

Split long work by fixed task, arm, or sample shard. For every shard, write an independent raw file,
completion marker, and hash. Aggregate only complete validated shards. Preserve interrupted shards
and list them under `partialExcludedRefs`; never include them in statistics.

Atomically update `checkpoint.json` at stage completion, Stop Gate, material error, and before a
long-running command. Use:

```json
{
  "cycle": 1,
  "contractRevision": 1,
  "contractHash": "...",
  "phase": "CALIBRATION",
  "completedUnits": ["arm-B0-shard-01"],
  "validatedArtifacts": ["workspace/cycles/1/raw/B0-01.json"],
  "lastProgressAt": "ISO-8601",
  "activeCommand": null,
  "resumeAction": "run remaining calibration shards",
  "partialExcludedRefs": []
}
```

Write a temporary file and atomically rename it to the required checkpoint path. Use a mechanical
runner for heartbeat, counts, and errors; do not spend LLM turns polling sample progress. On resume,
validate hashes and run only missing units.

## 7. Commit a reviewable result

Write `result.md` when the current atomic question has a valid positive, valid negative,
inconclusive, invalid, early-stop, or contract-conflict observation. Include:

1. contract revision/path/hash, cycle binding, carrier, and actual executed scope;
2. baseline correctness and competitiveness;
3. exact variant change and causal-interface trace;
4. completed arms, samples, and shards;
5. the exact stop condition observed, or the fact that none was observed;
6. paired/component results, statistics, guards, failures, and excluded partial artifacts;
7. the narrowest local observation and statements that cannot be inferred;
8. code, patch, commands, configs, raw, analysis, freeze, and cache-manifest refs.

Write to a temporary file in the same directory, fsync when practical, and atomically rename to the
Script-required `result.md`. Never expose a draft at the final path.

## 8. Name evidence honestly

Call latency that excludes media preprocessing “model-path latency,” not full request end-to-end
latency. Call concurrency=1 closed-loop requests/s “serial service rate,” not production throughput.
Use open-loop or adequate concurrent/continuous-batching execution when the claim depends on serving
throughput.

For non-inferiority, record sample-size or power reasoning and a finite-sample upper bound for
disagreement/error. Distinguish “no failure observed” from “the population non-inferiority bound is
satisfied,” especially after multi-task or multi-candidate selection.

Do not declare the entire Direction supported or rejected. Evidence Judge audits the result;
Experiment Decision chooses the workflow outcome.
