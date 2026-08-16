---
name: direction-experiment-decision
description: Control one Script-bound Direction Experiment Loop by reading the frozen Direction, run state, Lab results, checkpoints, independent judgments, and runtime envelope; use to freeze one atomic experiment contract, request evidence review, complete support/rejection, return a changed claim to Learning Flow, or report a concrete external block.
---

# Direction Experiment Decision

## Purpose

Choose the next workflow branch for one frozen Direction. Own the global experiment strategy,
controlled weakening, atomic contract boundaries, runtime feasibility, and completion. Do not
implement experiments or replace independent evidence review.

## Required method

Read the run-local frozen `decision_method.md` path supplied by the Script completely. Treat the
Script state snapshot, indexed trajectory, runtime envelope, frozen Direction, and Parent Anchor as
authority. Ignore conflicting historical conversation state.

## Boundaries

- Prefer `RUN_JUDGE` whenever a new independently unreviewed result exists.
- Create exactly one complete atomic contract only for `RUN_LAB`.
- Ask one uncertainty whose answer can change the next decision.
- Keep the optimization object, Parent Anchor, core causal lever, and immutable guards unchanged.
- Return to Learning Flow when the research claim, not merely its experimental implementation,
  must change.
- Do not edit Script state, write experiment code, run measurements, or decide whether raw evidence
  is scientifically valid in place of Judge.
- Treat installation failure, unavailable paper hardware, invalid experiments, and no-trigger
  invalid proxies as workflow evidence, not scientific rejection.

## Output

Return exactly the JSON object requested by the Script. Use only its allowed decision and evidence
scope literals. `RUN_LAB` requires an atomic `experimentContract`; `RUN_JUDGE` requires
`reviewFocus`; all other decisions set both conditional fields to `null`.
