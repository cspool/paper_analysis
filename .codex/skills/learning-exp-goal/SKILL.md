---
name: learning-exp-goal
description: Run one persistent, bounded experiment Goal selected by the Learning Loop to resolve a performance-baseline or mechanism uncertainty through iterative environment setup, implementation, measurement, and diagnosis.
---

# Learning EXP Goal

Execute one Script-frozen experimental objective to an evidence-backed terminal
result. This is a persistent Goal, so continue across Turns while useful work
remains. You may adapt the environment, code, and next measurement to observed
failures and data. You do not own the parent Learning state machine.

## Load the frozen task

1. Read the absolute `experiment_goal_task.json` path in the Prompt.
2. Read [experiment_goal_task_v1.md](references/experiment_goal_task_v1.md).
3. Starting at the task directory, walk upward to the nearest directory
   containing its `goalRef`; that is the parent Learning run directory.
4. Read the Workflow Goal, `anchorWork`, optional `directionWork`, and prior
   experiment results named by the parent context when present.
5. Treat `experimentObjective` as the only empirical question to resolve and
   `workspaceRef` as this Goal's durable working directory.

The parent Anchor and Direction are immutable context. Never edit Learning
state, results, reviews, observations, rounds, or final reports. Do not create
new Anchors, Directions, Reviewer verdicts, or Learning scheduling decisions.

## Iterate from the smallest discriminating experiment

1. Restate the competing possibilities and the observation that distinguishes
   them. Prefer profiling, trace analysis, workload sweep, microbenchmark,
   minimal ablation, simulator sensitivity, or a minimal prototype before a
   full system implementation.
2. Inspect available hardware, repositories, dependencies, data, and prior
   artifacts. Search `experiment_notes` and `human_notes` when they can reveal
   a reusable environment or known constraint; use other authorized local or
   online sources only as needed by the frozen objective.
3. Establish the closest reproducible baseline before implementing the primary
   change. Record exact commands, code revision or source, configuration,
   workload, seeds where relevant, environment, and raw outputs in the Goal
   workspace.
4. Implement only what is needed for the next discriminating observation.
   When a real error or measurement changes the diagnosis, revise the setup or
   experiment and continue; do not pretend the environment could have been
   fully predetermined.
5. Preserve failed attempts and negative results. Distinguish source facts,
   observed measurements, and inference. Check material correctness and
   guardrails before interpreting speed or resource results.
6. Stop when the objective has a credible supported, unsupported, narrower,
   or inconclusive answer, or when progress genuinely requires new authority
   or unavailable resources.

All experiment code, environment manifests, logs, raw measurements, analysis,
and derived artifacts belong under `workspaceRef` unless operating an exact
existing project in place is essential. If external state must be used, record
the path and avoid modifying unrelated user work.

## Maintain the Goal

Use the active Goal as the persistence mechanism. Keep working while its status
is active and a safe, bounded next experiment can add information. Mark it
complete only after the experimental objective has a terminal evidence-backed
answer; a credible negative or inconclusive result can be complete. Mark it
blocked only under the Goal protocol's genuine repeated-impasse rule. If new
permission, hardware, credentials, or a material user choice is required,
preserve the exact blocker and do not fabricate a result.

The parent runtime uses a progress-sensitive idle timeout: fifteen minutes
without meaningful Agent, tool, Goal-status, or usage activity pauses this EXP
Goal, while continued useful activity may run until the separate hard cap or
Goal budget. Before a potentially long operation, persist the exact command and
checkpoint in `workspaceRef`. Split opaque long-running work into observable
bounded steps, or make real progress visible in its logs; do not use empty
heartbeats to conceal a stalled experiment.

## Final answer

Return concise natural-language findings, not a controller JSON message. State:

- the objective and terminal conclusion (`supported`, `not supported`,
  `supported only under narrower conditions`, or `inconclusive`);
- the baseline, environment, and experiment actually run;
- key observed measurements or failure evidence;
- material validity limits and guardrails;
- paths to the code, logs, raw data, and analysis in `workspaceRef`;
- the consequence the next Learning Decision should consider.

The Script records this output as an immutable EXP Goal Result. It—not this
Goal—returns control to Learning Decision.
