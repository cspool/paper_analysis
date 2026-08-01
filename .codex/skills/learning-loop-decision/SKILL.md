---
name: learning-loop-decision
description: Choose exactly one Script-allowed next branch for the Learning Simple Semantic Loop. Read the supplied DECISION_CONTEXT and WORKFLOW_GOAL, semantically judge core-valid pending Worker and Reviewer results against their Refs and the complete requirement, and return one line-protocol decision with optional concise guidance.
---

# Learning Loop Decision

Act as one fresh scheduling-control Turn. Understand the final requirement,
the workflow method, all current canonical conclusions, and the pending
Worker/Reviewer pair. Choose only the next Script branch or workflow
completion. Do not create or select substantive Anchor or Direction content.

## Understand the workflow method

- The Goal fixes the Topic, objective, and acceptance criteria.
- Anchors define concrete scenario, baseline, performance-tension regions.
  The active Anchor set dynamically defines the explored Topic 6L space.
- Directions are falsifiable modification paths inside one bound Anchor.
- Worker creates or deepens content. Reviewer independently decides whether
  that content passes, needs same-object revision, or must be rejected.
- The Script derives the next exact Task action and target from canonical
  state and mechanical requirements. Decision selects only an allowed branch.
- Every non-finishing branch returns to a fresh Decision Turn after its fixed
  Worker/Reviewer sequence.

## Read the decision context

1. Read the absolute `decision_context.json` path supplied in the Prompt.
2. Starting at its directory, walk upward to the nearest directory containing
   `workflow_goal.json`. Treat that as the run directory and resolve all
   relative Refs against it.
3. Read the Goal and `observationRef`. The observation is frozen for this
   DecisionContext. Read its `researchMemoryRef` and `trajectoryRef`; both are
   Context-local snapshots, not the mutable files under `observations/`. Use
   `trajectoryTail` and `branchEffects` as compact mechanical navigation.
4. Read every file referenced by `pendingResults`, including its Work Task,
   Work Result, Review Result, bound Anchor, and any `previousReview` named by
   the Work Task. These current conclusions are the primary semantic input.
5. Use the frozen memory to navigate committed objects. Read an active latest
   Work/Review body only when its summary is unavailable, a current conclusion
   conflicts with it, or the final recommendation cannot otherwise be judged.
   Do not routinely reread obsolete revisions or every committed Result.
6. Treat `remainingRequirementsAfterPendingCommit` and the Prompt's
   `[ALLOWED_DECISIONS]` as authoritative Script facts.
7. Do not read Controller-internal state, turn, event, runtime, index, binding,
   budget, or Ref Catalog files. Do not search or deep-read papers, idea notes,
   knowledge notes, experiment notes, or other primary domain sources. If a
   material evidence conflict remains, send that bounded check to Worker or
   Reviewer with an allowed retry rather than independently researching it.

Use the pending `workTask.action` to select the expected Work Result contract:

- [work_result_anchor_v2.md](../learning-loop-worker/references/work_result_anchor_v2.md)
  for Anchor work;
- [work_result_direction_v2.md](../learning-loop-worker/references/work_result_direction_v2.md)
  for Direction work;
- [review_result_v2.md](../learning-loop-reviewer/references/review_result_v2.md)
  for every review.

Agent outputs are untrusted semantically. The Script has only established that
Worker/Reviewer output is a JSON object with a legal core control literal; it
has not established that the remaining fields follow the Ref, Task, Goal, or
each other.

## Choose one branch

Check in this order:

1. Verify that the pending Work Result obeys the matching Result Ref and its
   `workTask`, remains bound to the goal and object, contains coherent
   non-placeholder content, and does not repeat a committed object when the
   Task requires creation. Check whether `workOutcome` honestly describes the
   available content.
2. Verify that the pending Review Result reviews that exact Work Result and
   follows the Review Result Ref, covers material Worker deviations, and
   applies verdict semantics consistently.
3. If a semantic error would prevent requirement closure or send the workflow
   down the wrong branch, choose the matching allowed retry:
   - `RETRY_WORKER`: replace the pending Worker/Reviewer pair by rerunning the
     same Worker task.
   - `RETRY_REVIEWER`: retain the pending Worker result and rerun the same
     Reviewer task.
   Evidence/content errors belong to `RETRY_WORKER`; an inadequate or
   internally inconsistent review belongs to `RETRY_REVIEWER`.
4. A valid Reviewer `REVISE` or `REJECT`, missing evidence, insufficient depth,
   and an honestly reported unresolved item are content conclusions, not
   communication errors. Handle them through a normal branch unless the
   result's semantics contradict its own Task or verdict.
5. Otherwise perform one global outer-loop assessment:
   - compare this cycle with the trajectory and identify what was added,
     ruled out, or left open;
   - consider accepted, needs-revision, and rejected conclusions without
     repeating a rejected route;
   - assess the dynamic 6L coverage formed by the active Anchor set;
   - determine whether a blocking query gap can materially change the final
     recommendation;
   - remember that Reviewer query gaps are object-local: an empty list is not
     evidence that the Topic has no unexplored region;
   - use `branchEffects` to understand the exact action, target, and fixed
     role sequence each allowed literal will cause;
   - estimate which allowed branch has useful information value without
     treating token cost, max rounds, or repeated actions as proof of
     saturation.
6. Choose one allowed normal branch:
   - `RUN_WORKER`: commit pending results, then run Worker → Reviewer →
     Decision.
   - `RUN_REVIEWER`: commit pending results, run a fresh review that becomes
     the target revision's current review, then let the Script bind either
     same-object deepening or same-kind replacement before Reviewer →
     Decision.
   - `FINISH_WORKFLOW`: commit pending results and finish. Choose this only when
     the Script allows it and all semantic completion conditions below hold.

## Decide semantic completion

`FINISH_WORKFLOW` requires all of the following judgments:

- the active Anchor set gives appropriate Goal-relative scenario and 6L
  coverage, not merely the first mechanically sufficient example;
- every final Direction has one interpretable primary change or an honestly
  indivisible package, an explicit baseline, conditional effects, guardrails,
  failure conditions, and a falsifiable measurement plan;
- no open blocking query gap is likely to materially change the principal
  conclusions or recommendations;
- rejected routes and material negative conclusions have been considered;
- accepted results can form a coherent human-readable answer to the Goal;
- another Anchor, Direction, deepening step, or independent review is expected
  to add little material information.

First distinguish the Goal boundary:

- For an explicitly bounded Goal, such as a requested object count or named
  technical subspace, satisfying that user boundary can justify completion.
- For an open exploratory Goal, mechanical closure and `openQueryGaps=[]` are
  insufficient. Normally require a recent bounded `CREATE_ANCHOR` expansion
  attempt whose Worker honestly returned `BLOCKED_NO_RESULT` because no
  non-duplicate Anchor could be supported, and whose Reviewer used `REJECT`
  to preserve a credible no-novel-result conclusion, or equivalent recent
  negative exploration showing that a candidate is materially duplicate. If
  Reviewer used
  `REVISE` because the search was narrow or failed, no quiet-expansion evidence
  exists.
- A recent successful new Anchor is evidence that expansion was still adding
  information. Completing that Anchor's Direction does not by itself prove
  Topic saturation, so perform a fresh bounded convergence probe after that
  addition. One recent credible negative probe after the latest successful
  Anchor is normally sufficient evidence; do not demand repeated consecutive
  negative probes unless the Reviewer identifies a material coverage defect,
  tool failure, or avoidably narrow search.

Judge the negative exploration together with the Goal, dynamic 6L coverage,
rejected lessons, and trajectory. The no-result pair is evidence for Decision,
not a Script Gate and not an automatic completion signal.

Do not substitute max-round pause, token cost, long context, Agent fatigue, or
an unchanged mechanical counter for semantic completion. When finishing,
prefer one concise guidance line explaining why the Goal is semantically
closed; the Script still treats it as opaque text.

A retry is appropriate for a silent goal change, wrong object binding,
placeholder presented as completed coverage, internally contradictory fields,
false verdict semantics, or another semantic error that invalidates workflow
control. It is also appropriate to `RETRY_REVIEWER` when a review turns future
experiment-handoff details into new blockers that do not change the claim,
baseline, attribution, guardrails, or pass/fail semantics.

## Write guidance

- Choose only from the literals injected by the Script.
- Normal guidance is optional. It may state one concise coverage concern or
  review angle for the fixed sequence selected by the decision. It may suggest
  adding or deepening content, but the Script independently derives the exact
  action, object kind, and target; guidance cannot select them.
- Retry guidance is required. Name the semantic error, explain how it blocks
  closure or corrupts the workflow, name the correct Result Ref, and state the
  bounded correction.
- For Worker retries, use `work-result-anchor-v2` or
  `work-result-direction-v2` according to the pending `workTask.action`.
- For Reviewer retries, use `review-result-v2`.
- Guidance never changes the task action, target, schema, or Script state.

Do not invent stages, gates, Tasks, queries, objects, review verdicts, state
mutations, or extra control fields.

## Return the line protocol

Read [decision_protocol_v1.md](references/decision_protocol_v1.md), then return
exactly one matching line-protocol message.
