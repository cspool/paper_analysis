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
- An allowed `RUN_EXP_GOAL` branch is a bounded empirical detour:
  Decision → persistent EXP Goal → fresh Decision. The EXP Goal may iterate
  environment, code, measurement, and diagnosis, but it never commits a
  Learning object or chooses a Learning branch.

## Read the decision context

1. Read the absolute `decision_context.json` path supplied in the Prompt.
2. Starting at its directory, walk upward to the nearest directory containing
   `workflow_goal.json`. Treat that as the run directory and resolve all
   relative Refs against it.
3. Read the Goal and `observationRef`. The observation is frozen for this
   DecisionContext. Read its `researchMemoryRef` and `trajectoryRef`; both are
   Context-local snapshots, not the mutable files under `observations/`. Use
   `trajectoryTail` and `branchEffects` as compact mechanical navigation. When
   present, also read its frozen `negativeExperimentHistoryRef`; it is a
   Script-generated Ref index, not a semantic family classification.
4. When `pendingResults` is non-null, read every file it references, including its Work Task,
   Work Result, Review Result, bound Anchor, and any `previousReview` named by
   the Work Task. These current conclusions are the primary semantic input.
5. Read `experimentContext` when present. It identifies the frozen current
   Anchor, optional Direction, and prior EXP Goal result refs under that
   Anchor—not only the current Direction. Read those results and their
   conclusions when they bear on the next branch. A completed, negative,
   narrow, inconclusive, paused, or failed experiment is evidence—not an
   automatic verdict.
6. Use the frozen memory to navigate committed objects. Read an active latest
   Work/Review body only when its summary is unavailable, a current conclusion
   conflicts with it, or the final recommendation cannot otherwise be judged.
   Do not routinely reread obsolete revisions or every committed Result.
7. Treat `remainingRequirementsAfterPendingCommit` and the Prompt's
   `[ALLOWED_DECISIONS]` as authoritative Script facts.
8. Do not read Controller-internal state, turn, event, runtime, index, binding,
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

1. When a pending pair exists, verify that its Work Result obeys the matching Result Ref and its
   `workTask`, remains bound to the goal and object, contains coherent
   non-placeholder content, and does not repeat a committed object when the
   Task requires creation. Check whether `workOutcome` honestly describes the
   available content.
2. When a pending pair exists, verify that its Review Result reviews that exact Work Result and
   follows the Review Result Ref, covers material Worker deviations, and
   applies verdict semantics consistently.
3. When a pending pair exists and a semantic error would prevent requirement closure or send the workflow
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
5. Otherwise perform one global outer-loop assessment. This includes the
   post-EXP case where no pending Worker/Reviewer pair exists:
   - compare this cycle with the trajectory and identify what was added,
     ruled out, or left open;
   - compare reviewed negative experiments by baseline change, causal lever,
     and preserved model/workload/execution boundary rather than by Direction
     name;
   - consider accepted, needs-revision, and rejected conclusions without
     repeating a rejected route;
   - assess the dynamic 6L coverage formed by the active Anchor set;
   - determine whether a blocking query gap can materially change the final
     recommendation;
   - remember that Reviewer query gaps are object-local: an empty list is not
     evidence that the Topic has no unexplored region;
   - use `branchEffects` to understand the exact action, target, and fixed
     role sequence each allowed literal will cause;
   - distinguish source-reported performance gaps from observations produced
     by an integrated EXP Goal. A polished paper-only Direction is still a
     hypothesis when its principal value depends on local applicability;
   - when Reviewer records a verdict-changing `experiment` query gap, compare
     the smallest discriminating experiment with another literature/deepening
     cycle rather than treating EXP as an exceptional fallback;
   - estimate which allowed branch has useful information value without
     treating token cost, max rounds, or repeated actions as proof of
     saturation.
6. Choose one allowed normal branch:
   - `RUN_WORKER`: commit pending results, then run Worker → Reviewer →
     Decision.
   - `RUN_REVIEWER`: follow the exact `branchEffects` sequence. In ordinary
     work this may be Reviewer → Worker → Reviewer → Decision. Immediately
     after an unreviewed EXP it is the atomic POST_EXP_REVIEWER → Decision
     path; after reviewed negative Directions leave an Anchor without a viable
     Direction it may be an Anchor reassessment → Decision path. Do not assume
     an automatic Worker suffix.
   - `RUN_EXP_GOAL`: commit any pending pair, freeze the current Anchor and
     optional Direction, run one persistent experimental Goal, and return
     directly to a fresh Decision. Prefer it when one empirical fact can
     materially change the Anchor, Direction, or workflow choice, saved
     evidence cannot settle it, and a bounded trace, profile, microbenchmark,
     minimal reproduction, or ablation can discriminate the possibilities.
     A Reviewer `REVISE` with a matching `experiment` query gap is the clearest
     signal, provided this literal is allowed. For a Direction mechanism test,
     require that the closest method baseline has already been checked. Prefer
     one cheap go/no-go observation before a full Direction implementation.
   - `FINISH_WORKFLOW`: commit pending results and finish. Choose this only when
     the Script allows it and all semantic completion conditions below hold.

## Converge reviewed negative experiments

Treat Reviewer conclusions as the semantic entry point for negative EXP
evidence. `goalStatus=complete` alone says only that the EXP Goal terminated
normally. `budgetLimited`, credential or download failure, runtime failure,
and an experiment without a completed discriminating measurement are not
negative mechanism evidence unless a Reviewer establishes otherwise.

Use this default semantic policy within the experiment's stated boundary:

1. The first credible negative closes the tested Direction, not automatically
   the whole mechanism family.
2. A second credible negative with substantially the same baseline change,
   causal lever, and preserved boundary normally closes that mechanism family.
   Prefer another causal lever, another concrete 6L object, another Anchor, or
   parent-Anchor reassessment instead of merely changing a score, threshold,
   feature, or small head.
3. Permit at most one evidence-backed reopening when the candidate changes a
   failed causal assumption and independent evidence supports that difference.
   Guidance must name the changed assumption and the smallest observation that
   changes the Learning choice.
4. If that reopening is also credibly negative, treat the broader mechanism
   family as converged for this Run and boundary. Do not spend remaining EXP
   authorization on a fourth adjacent variant.

These are semantic defaults, not Script counters. Read the EXP and Review Refs
before grouping them, preserve applicability boundaries, and do not extrapolate
a local negative result to every model, workload, or hardware regime.

When a reviewed mechanism-family failure leaves an active Anchor without a
credible Direction, consider `RUN_REVIEWER` for the Script-previewed parent
Anchor reassessment. If the original performance tension or remaining
optimization space is no longer supportable, the Reviewer may reject that
Anchor so mechanical closure does not manufacture replacements. If a genuinely
different causal lever remains, choose `RUN_WORKER` and use guidance only to
state the failed assumption to avoid; Worker owns the substantive candidate.

## Decide semantic completion

`FINISH_WORKFLOW` requires all of the following judgments. Do not run an EXP
Goal merely because completion is near or because budget remains. Before
finishing, however, explicitly audit whether an accepted high-value Direction
still rests on a source-reported performance gap whose local existence or
causal attribution can be cheaply discriminated. Experiment is an
information-gain branch, not a ceremonial final confirmation stage:

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
- `RUN_EXP_GOAL` guidance is required and becomes the experiment objective.
  State one concise empirical question, the baseline or mechanism uncertainty
  it resolves, and how its observation will change the next Learning choice.
  Do not prescribe the complete simulator, repository, commands, or experiment
  implementation; the EXP Goal adapts those from real results.
- Guidance never changes the task action, target, schema, or Script state.

Do not invent stages, gates, Tasks, queries, objects, review verdicts, state
mutations, or extra control fields.

## Return the line protocol

Read [decision_protocol_v1.md](references/decision_protocol_v1.md), then return
exactly one matching line-protocol message.
