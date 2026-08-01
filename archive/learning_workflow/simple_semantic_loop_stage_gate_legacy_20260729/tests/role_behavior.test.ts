import assert from "node:assert/strict";
import { test } from "node:test";
import {
  CLOSURE_CHECK_NAMES,
  CLOSURE_FINDING_REGISTRY,
  canonicalJson,
  type ClosureChecks,
  type ClosureReviewEnvelope,
  type ClosureReviewTaskEnvelope,
  type EvidencePacketEnvelope,
  type EvidenceReaderTaskEnvelope,
  type KnowledgeDimension,
  type ReviewDeltaEnvelope,
  type SearchIntent,
} from "../contracts/index.ts";
import {
  validateEvidenceRuntimeTrace,
  validateRuntimeToolEvents,
  type RuntimeToolEvent,
} from "../security/no_experiment_guard.ts";
import {
  INTENT_DIMENSION_ROUTES,
  validateSearchNeed,
} from "../validators/domain_validator.ts";
import {
  deriveClosureFacts,
  validateClosureReview,
} from "../validators/closure_review_validator.ts";
import { validateEvidencePacket } from "../validators/evidence_packet_validator.ts";
import { validateReviewDelta } from "../validators/review_delta_validator.ts";
import {
  makeClosureResult,
  makeClosureTask,
  makeDirectionTask,
  makeEvidenceResult,
  makeEvidenceTask,
  makeReviewResult,
  makeSearchNeed,
} from "./fixtures/factory.ts";

test("all eight SearchIntent routes admit only their frozen dimensions", () => {
  const intents = Object.keys(INTENT_DIMENSION_ROUTES) as SearchIntent[];
  assert.equal(intents.length, 8);
  for (const intent of intents) {
    const route = INTENT_DIMENSION_ROUTES[intent];
    for (const auxiliary of [null, ...route.auxiliary]) {
      const need = makeSearchNeed();
      need.intent = intent;
      need.primaryDimension = route.primary;
      need.auxiliaryDimension = auxiliary;
      need.targetDimensions = [
        route.primary,
        ...(auxiliary ? [auxiliary] : []),
      ];
      const report = validateSearchNeed(need);
      assert.equal(
        report.valid,
        true,
        `${intent}/${auxiliary}: ${canonicalJson(report.errors)}`,
      );
    }

    const invalid = makeSearchNeed();
    invalid.intent = intent;
    invalid.primaryDimension = otherDimension(route.primary);
    invalid.auxiliaryDimension = null;
    invalid.targetDimensions = [invalid.primaryDimension];
    const report = validateSearchNeed(invalid);
    assert.equal(report.valid, false, `${intent} rejects wrong primary`);
    assert.ok(
      report.errors.some(
        (error) => error.code === "domain.search_route_primary",
      ),
    );
  }
});

test("Evidence Q1 → Q2 → deep read → context-derived Q3 is valid and trace-bound", () => {
  const task = makeEvidenceTask();
  const { result, events } = makeThreeLevelEvidence(task);
  const semantic = validateEvidencePacket(result, task);
  assert.equal(semantic.valid, true, canonicalJson(semantic.errors));
  const trace = validateEvidenceRuntimeTrace(task, result, events);
  assert.equal(trace.valid, true, canonicalJson(trace.errors));
});

test("Evidence runtime admission rejects reordering, failed tools, and unobserved sources", () => {
  const task = makeEvidenceTask();
  const { result, events } = makeThreeLevelEvidence(task);

  const reordered = structuredClone(events);
  [reordered[2], reordered[4]] = [reordered[4]!, reordered[2]!];
  const orderReport = validateEvidenceRuntimeTrace(task, result, reordered);
  assert.ok(
    orderReport.errors.some(
      (error) => error.code === "security.evidence_tool_ledger_order",
    ),
  );

  const failed = structuredClone(events);
  failed[0]!.status = "failed";
  failed[0]!.error = "tool unavailable";
  const failureReport = validateEvidenceRuntimeTrace(task, result, failed);
  assert.ok(
    failureReport.errors.some(
      (error) => error.code === "security.tool_call_failed",
    ),
  );

  const hiddenHit = structuredClone(events);
  hiddenHit[1]!.resultText = "no matching path";
  const hitReport = validateEvidenceRuntimeTrace(task, result, hiddenHit);
  assert.ok(
    hitReport.errors.some(
      (error) => error.code === "security.evidence_hit_not_observed",
    ),
  );

  const legacyFlatRead = structuredClone(events);
  const readIndex = legacyFlatRead.findIndex(
    (event) =>
      event.toolName === "mcp__obsidian__obsidian_get_note",
  );
  const read = legacyFlatRead[readIndex]!;
  const nestedTarget = read.arguments.target as {
    type: "path";
    path: string;
  };
  read.arguments = {
    path: nestedTarget.path,
    format: read.arguments.format,
  };
  const flatReadReport = validateEvidenceRuntimeTrace(
    task,
    result,
    legacyFlatRead,
  );
  assert.ok(
    flatReadReport.errors.some(
      (error) => error.code === "security.obsidian_read_arguments",
    ),
  );

  const traversalRead = structuredClone(events);
  const traversalIndex = traversalRead.findIndex(
    (event) =>
      event.toolName === "mcp__obsidian__obsidian_get_note",
  );
  traversalRead[traversalIndex]!.arguments.target = {
    type: "path",
    path: "knowledge_notes/../human_notes/private.md",
  };
  const traversalReport = validateEvidenceRuntimeTrace(
    task,
    result,
    traversalRead,
  );
  assert.ok(
    traversalReport.errors.some(
      (error) => error.code === "security.obsidian_read_arguments",
    ),
  );

  const widenedSearch = structuredClone(events);
  const searchIndex = widenedSearch.findIndex(
    (event) =>
      event.toolName === "mcp__obsidian__obsidian_search_notes",
  );
  widenedSearch[searchIndex]!.arguments.query =
    "path:knowledge_notes/../human_notes/ launch fusion";
  const widenedSearchReport = validateEvidenceRuntimeTrace(
    task,
    result,
    widenedSearch,
  );
  assert.ok(
    widenedSearchReport.errors.some(
      (error) => error.code === "security.obsidian_search_arguments",
    ),
  );

  const writeEvent: RuntimeToolEvent = {
    toolName: "mcp__obsidian__obsidian_patch_note",
    arguments: { path: "knowledge_notes/fusion.md" },
    status: "completed",
    resultText: "patched",
    error: null,
  };
  const writeReport = validateRuntimeToolEvents(
    "evidence_reader",
    [writeEvent],
    task,
  );
  assert.ok(
    writeReport.errors.some(
      (error) =>
        error.code === "security.unregistered_tool" ||
        error.code === "security.execution_event",
    ),
  );
});

test("Evidence rejects repeated logical queries and consumed source units", () => {
  const task = makeEvidenceTask();
  const result = makeEvidenceResult();
  task.payload.previousQueries.push({
    query: result.payload.searches[0]!.query.toUpperCase(),
    dimension: "knowledge",
    logicalQueryLevel: 1,
    outcome: "no_hits",
  });
  task.payload.consumedSourceUnitIds.push(
    result.payload.findings[0]!.sourceUnitId,
  );
  const report = validateEvidencePacket(result, task);
  assert.ok(
    report.errors.some(
      (error) => error.code === "evidence.duplicate_logical_query",
    ),
  );
  assert.ok(
    report.errors.some(
      (error) => error.code === "evidence.consumed_source_unit",
    ),
  );
});

test("Direction decision matrix admits all four decisions and exact duplicate rejection", () => {
  const task = makeDirectionTask();

  const testable = makeReviewResult();
  assertValidReview(testable, task, "testable");

  const continueSearch = makeReviewResult();
  continueSearch.payload.decision = "continue_search";
  continueSearch.payload.nextQuestion =
    "Which supplied local note resolves the weakest causal link?";
  continueSearch.payload.nextQuestionAnswerableFromKnowledgeBase = true;
  continueSearch.payload.readinessChecks.causalChainFalsifiable = false;
  continueSearch.payload.readinessChecks.knowledgeAnswerableCriticalGapRemaining =
    true;
  continueSearch.payload.weakestCausalLink =
    "The launch-count to latency relation is not yet bounded.";
  assertValidReview(continueSearch, task, "continue_search");

  const experiment = makeReviewResult();
  experiment.payload.decision = "experiment_required";
  experiment.payload.readinessChecks.newExperimentRequired = true;
  experiment.payload.experimentHandoff = {
    handoffId: "handoff-1",
    directionId: "direction-1",
    tag: "EXPERIMENT_REQUIRED",
    reason: "Only a new controlled measurement can determine the effect.",
    requiredArtifact: "benchmark",
    hypothesisToTest: "Fusion lowers latency under equal outputs.",
    suggestedEntryPoints: ["runtime scheduler decode dispatch"],
    controlledVariables: ["model", "input tokens", "output tokens"],
    metrics: ["latency"],
    acceptanceCriteria: ["Compare the same model and token sequence."],
    failureStopConditions: ["Stop if output equivalence is not established."],
    executionAuthorized: false,
  };
  assertValidReview(experiment, task, "experiment_required");

  const rejected = makeReviewResult();
  rejected.payload.decision = "rejected";
  rejected.payload.rejectionCategory = "out_of_scope";
  rejected.payload.readinessChecks.inTopicAndAnchorScope = false;
  assertValidReview(rejected, task, "rejected/out_of_scope");

  const duplicateTask = makeDirectionTask();
  duplicateTask.payload.siblingDirections = [
    {
      directionRef: {
        objectType: "direction",
        objectId: "direction-sibling",
        revision: 2,
      },
      baseline: "separate launches",
      comparisonScope: ["same model"],
      controlledVariables: ["model", "tokens"],
      primaryChanges: [
        {
          layer: "L4",
          object: "decode operation sequence",
          fromState: "separate launches",
          toState: "one fused launch",
          conditions: ["batch=1"],
        },
      ],
      causalTargets: ["lower launch overhead"],
      hypothesis: "Equivalent fusion hypothesis.",
    },
  ];
  const duplicate = makeReviewResult();
  duplicate.payload.decision = "rejected";
  duplicate.payload.rejectionCategory = "duplicate";
  duplicate.payload.duplicateOfDirectionRef =
    duplicateTask.payload.siblingDirections[0]!.directionRef;
  duplicate.payload.duplicateComparison = {
    baselineScopeEquivalent: true,
    primaryChangeEquivalent: true,
    causalTargetEquivalent: true,
    materialDifference: null,
  };
  assertValidReview(duplicate, duplicateTask, "rejected/duplicate");
});

test("Direction Reviewer fails closed on executable handoffs and fabricated counterexamples", () => {
  const task = makeDirectionTask();
  const experiment = makeReviewResult();
  experiment.payload.decision = "experiment_required";
  experiment.payload.readinessChecks.newExperimentRequired = true;
  experiment.payload.experimentHandoff = {
    handoffId: "handoff-1",
    directionId: "direction-1",
    tag: "EXPERIMENT_REQUIRED",
    reason: "External evidence is required.",
    requiredArtifact: "benchmark",
    hypothesisToTest: "Fusion lowers latency.",
    suggestedEntryPoints: ["python run_benchmark.py"],
    controlledVariables: ["model"],
    metrics: ["latency"],
    acceptanceCriteria: ["Same outputs."],
    failureStopConditions: ["Stop on mismatch."],
    executionAuthorized: false,
  };
  const executable = validateReviewDelta(experiment, task);
  assert.ok(
    executable.errors.some(
      (error) =>
        error.code === "no_experiment.executable_handoff_instruction",
    ),
  );

  const counterexampleTask = makeDirectionTask();
  counterexampleTask.payload.counterexamples = [
    {
      counterexampleId: "counter-1",
      statement: "Fusion can increase register pressure.",
      evidenceRefs: ["ev-1"],
      degradationCondition: "large fused kernel",
    },
  ];
  const fabricated = makeReviewResult();
  fabricated.payload.strongestCounterexample =
    "A counterexample that was never supplied.";
  fabricated.payload.counterexampleResolution = "It is resolved.";
  const counterexample = validateReviewDelta(fabricated, counterexampleTask);
  assert.ok(
    counterexample.errors.some(
      (error) => error.code === "review.counterexample_not_supplied",
    ),
  );
});

test("all thirteen Closure checks independently prevent a fabricated accept", () => {
  const cases: Record<
    keyof ClosureChecks,
    (task: ClosureReviewTaskEnvelope) => void
  > = {
    stopProofRevisionCurrent(task) {
      task.payload.currentCanonicalRevision += 1;
    },
    stopProofMatchesCanonical(task) {
      task.payload.stopCandidateBundle.candidate.topicId = "topic-other";
    },
    mechanicalPreflightPassed(task) {
      task.payload.mechanicalPreflight.passed = false;
    },
    topicScopePreserved(task) {
      task.payload.topic.scopeAudit.currentFingerprint = "narrowed";
      task.payload.topic.scopeAudit.changes.push({
        changeId: "scope-change-1",
        field: "workloads",
        changeType: "narrow",
        userAuthorized: false,
        reason: "Silent narrowing.",
      });
    },
    noKnowledgeAnswerableCriticalNeed(task) {
      task.payload.needs.push({
        needId: "need-open",
        revision: 1,
        owner: {
          topicId: "topic-1",
          anchorId: null,
          directionId: null,
        },
        critical: true,
        answerability: "knowledge_base",
        status: "pending",
      });
      task.payload.stopCandidateBundle.proof.openNeedIds.push("need-open");
    },
    allAnchorsClosed(task) {
      task.payload.anchors[0]!.status = "active";
      task.payload.anchors[0]!.saturationReason = null;
    },
    allDirectionsTerminal(task) {
      task.payload.directions[0]!.status = "exploring";
    },
    lastTopicExpansionNoDelta(task) {
      task.payload.lastTopicExpansion = null;
      task.payload.stopCandidateBundle.proof.lastTopicExpansionNeedId = null;
    },
    noUnconsumedOrUncommittedWork(task) {
      task.payload.taskIndex.push({
        id: "task-pending",
        status: "pending",
        resolvedById: null,
        objectRef: null,
      });
      task.payload.stopCandidateBundle.proof.pendingTaskIds.push(
        "task-pending",
      );
    },
    criticalContradictionsReviewed(task) {
      task.payload.contradictions.push({
        contradictionId: "contradiction-open",
        critical: true,
        dispositionReviewId: null,
        objectRef: {
          objectType: "contradiction",
          objectId: "contradiction-open",
          revision: 1,
        },
      });
      task.payload.stopCandidateBundle.proof
        .unreviewedCriticalContradictionIds.push("contradiction-open");
    },
    experimentHandoffsComplete(task) {
      task.payload.directions[0]!.status = "experiment_required";
      task.payload.directions[0]!.experimentHandoffId = "handoff-missing";
    },
    runtimeEligibleForCompletion(task) {
      task.payload.runtimeEligibility.budgetExhausted = true;
      task.payload.runtimeEligibility.reason = "turn budget exhausted";
    },
    finalOutputTraceable(task) {
      task.payload.outputCoverage.fields.topic_scope = [];
    },
  };

  assert.deepEqual(
    Object.keys(cases).sort(),
    [...CLOSURE_CHECK_NAMES].sort(),
  );
  for (const check of CLOSURE_CHECK_NAMES) {
    const task = makeClosureTask();
    cases[check](task);
    const derived = deriveClosureFacts(task);
    assert.equal(derived.checks[check], false, `${check} is derived false`);
    const report = validateClosureReview(makeClosureResult(), task);
    assert.equal(report.valid, false, `${check} cannot be accepted`);
  }
});

test("Closure treats reasoned rejected Anchors as closed, but rejects missing reasons", () => {
  const valid = makeClosureTask();
  valid.payload.anchors[0]!.status = "rejected";
  valid.payload.anchors[0]!.statusReason =
    "The candidate violates the immutable Topic scope.";
  valid.payload.anchors[0]!.saturationReason = null;
  assert.equal(deriveClosureFacts(valid).checks.allAnchorsClosed, true);

  const invalid = structuredClone(valid);
  invalid.payload.anchors[0]!.statusReason = "";
  const facts = deriveClosureFacts(invalid);
  assert.equal(facts.checks.allAnchorsClosed, false);
  assert.ok(
    facts.issues.some(
      (issue) => issue.code === "anchor_missing_status_reason",
    ),
  );
});

test("Closure accepts an exact reject report and rejects fabricated basis refs", () => {
  const task = makeClosureTask();
  task.payload.anchors[0]!.status = "active";
  task.payload.anchors[0]!.saturationReason = null;
  task.payload.stopCandidateBundle.proof.claims.allAnchorsClosed = false;
  const reject = makeExactClosureReject(task);
  const valid = validateClosureReview(reject, task);
  assert.equal(valid.valid, true, canonicalJson(valid.errors));

  const acceptTask = makeClosureTask();
  const fabricated = makeClosureResult();
  fabricated.payload.verifiedClosureBasis[0]!.objectRefs = [
    {
      objectType: "direction",
      objectId: "fabricated",
      revision: 99,
    },
  ];
  const report = validateClosureReview(fabricated, acceptTask);
  assert.ok(
    report.errors.some(
      (error) => error.code === "closure.fabricated_basis_ref",
    ),
  );
});

function makeThreeLevelEvidence(
  task: EvidenceReaderTaskEnvelope,
): { result: EvidencePacketEnvelope; events: RuntimeToolEvent[] } {
  const result = structuredClone(makeEvidenceResult());
  const path = "knowledge_notes/fusion.md";
  const exactContext =
    "For batch-1 decode, fusing adjacent launches removes launch overhead and reduces per-token latency.";
  result.payload.searches = [
    {
      searchId: "search-q1",
      sequence: 1,
      logicalQueryLevel: 1,
      dimension: "knowledge",
      query:
        "path:knowledge_notes/ launch fusion batch-1 decode mechanism",
      pathFilter: "path:knowledge_notes/",
      terms: [
        {
          term: "launch fusion",
          source: "task",
          sourceRef: "searchNeed.technicalObjects",
          introducedAtSequence: 0,
        },
        {
          term: "batch-1 decode",
          source: "task",
          sourceRef: "searchNeed.scenarioTerms",
          introducedAtSequence: 0,
        },
        {
          term: "mechanism",
          source: "task",
          sourceRef: "searchNeed.evidenceIntentTerms",
          introducedAtSequence: 0,
        },
      ],
      page: 1,
      cursorUsed: null,
      nextCursor: null,
      toolCallIndex: 1,
      pageHitCount: 0,
      cumulativeHitCount: 0,
      outcome: "no_hits",
      stopReason: "next_level_required",
    },
    {
      searchId: "search-q2",
      sequence: 2,
      logicalQueryLevel: 2,
      dimension: "knowledge",
      query: "path:knowledge_notes/ batch-1 decode lower latency",
      pathFilter: "path:knowledge_notes/",
      terms: [
        {
          term: "batch-1 decode",
          source: "task",
          sourceRef: "searchNeed.scenarioTerms",
          introducedAtSequence: 0,
        },
        {
          term: "lower latency",
          source: "task",
          sourceRef: "searchNeed.performanceRelations",
          introducedAtSequence: 0,
        },
      ],
      page: 1,
      cursorUsed: null,
      nextCursor: null,
      toolCallIndex: 2,
      pageHitCount: 1,
      cumulativeHitCount: 1,
      outcome: "hits",
      stopReason: "next_level_required",
    },
    {
      searchId: "search-q3",
      sequence: 6,
      logicalQueryLevel: 3,
      dimension: "idea",
      query: "path:idea_notes/ fusing",
      pathFilter: "path:idea_notes/",
      terms: [
        {
          term: "fusing",
          source: "context",
          sourceRef: "context-1",
          introducedAtSequence: 5,
        },
      ],
      page: 1,
      cursorUsed: null,
      nextCursor: null,
      toolCallIndex: 3,
      pageHitCount: 0,
      cumulativeHitCount: 0,
      outcome: "no_hits",
      stopReason: "success_criteria_met",
    },
  ];
  result.payload.hitsConsidered = [
    {
      hitId: "hit-1",
      searchId: "search-q2",
      sequence: 3,
      path,
      score: 8.2,
      sourceFamily: "fusion-note",
      selected: true,
      selectionReason: "Directly addresses the success criterion.",
    },
  ];
  result.payload.contextsRead = [
    {
      contextId: "context-map-1",
      sequence: 4,
      hitId: "hit-1",
      path,
      format: "document-map",
      heading: "",
      sectionTarget: null,
      sourceUnitId: "map",
      sourceFamily: "fusion-note",
      exactContext: "## Decode mechanism",
      summary: "The note contains a decode mechanism section.",
    },
    {
      contextId: "context-1",
      sequence: 5,
      hitId: "hit-1",
      path,
      format: "section",
      heading: "Decode mechanism",
      sectionTarget: "Decode mechanism",
      sourceUnitId: "decode-mechanism",
      sourceFamily: "fusion-note",
      exactContext,
      summary: "Fusion removes launch overhead in batch-1 decode.",
    },
  ];
  result.payload.findings = [
    {
      evidenceId: "ev-1",
      claimKey: "fusion-removes-launch-overhead",
      claim: "Launch fusion removes launch overhead in batch-1 decode.",
      evidenceRole: "mechanism",
      directness: "direct",
      attribution: "source_report",
      applicableConditions: ["batch-1 decode"],
      comparisonBaseline: "separate launches",
      sourcePath: path,
      sourceUnitId: "decode-mechanism",
      sourceFamily: "fusion-note",
      heading: "Decode mechanism",
      quoteOrExactContext:
        "fusing adjacent launches removes launch overhead",
    },
  ];
  result.payload.unanswered = [];
  result.payload.conclusion = "answered";

  const searches = result.payload.searches;
  const contexts = result.payload.contextsRead;
  const events: RuntimeToolEvent[] = [
    searchEvent(searches[0]!, "no hits"),
    searchEvent(searches[1]!, `result: ${path}`),
    readEvent(contexts[0]!),
    readEvent(contexts[1]!),
    searchEvent(searches[2]!, "no additional hits"),
  ];
  return { result, events };
}

function searchEvent(
  search: EvidencePacketEnvelope["payload"]["searches"][number],
  resultText: string,
): RuntimeToolEvent {
  return {
    toolName: "mcp__obsidian__obsidian_search_notes",
    arguments: {
      mode: "omnisearch",
      query: search.query,
      ...(search.cursorUsed ? { cursor: search.cursorUsed } : {}),
    },
    status: "completed",
    resultText,
    error: null,
  };
}

function readEvent(
  context: EvidencePacketEnvelope["payload"]["contextsRead"][number],
): RuntimeToolEvent {
  return {
    toolName: "mcp__obsidian__obsidian_get_note",
    arguments: {
      target: { type: "path", path: context.path },
      format: context.format,
      ...(context.sectionTarget
        ? {
            section: {
              type: "heading",
              target: context.sectionTarget,
            },
          }
        : {}),
    },
    status: "completed",
    resultText: context.exactContext,
    error: null,
  };
}

function assertValidReview(
  review: ReviewDeltaEnvelope,
  task: ReturnType<typeof makeDirectionTask>,
  label: string,
): void {
  const report = validateReviewDelta(review, task);
  assert.equal(report.valid, true, `${label}: ${canonicalJson(report.errors)}`);
}

function makeExactClosureReject(
  task: ClosureReviewTaskEnvelope,
): ClosureReviewEnvelope {
  const derived = deriveClosureFacts(task);
  const runRef = {
    objectType: "run",
    objectId: task.runId,
    revision: null,
  };
  const reopenScopes = [
    ...new Map(
      derived.issues.map((issue) => [
        canonicalJson(issue.scope),
        issue.scope,
      ]),
    ).values(),
  ];
  const result = structuredClone(makeClosureResult());
  result.payload.decision = "reject";
  result.payload.closureChecks = derived.checks;
  result.payload.verifiedClosureBasis = CLOSURE_CHECK_NAMES.filter(
    (check) => derived.checks[check],
  ).map((check) => ({
    check,
    statement: `${check} is true in the supplied canonical projection.`,
    objectRefs: [runRef],
  }));
  result.payload.blockingFindings = derived.issues.map((issue, index) => {
    const rule = CLOSURE_FINDING_REGISTRY[issue.code];
    return {
      findingId: `finding-${index + 1}`,
      check: issue.check,
      type: rule.type,
      code: issue.code,
      summary: `${issue.code} blocks completion.`,
      objectRefs: [issue.scope],
      reopenScope: issue.scope,
      recoveryAction: rule.recoveryAction,
    };
  });
  result.payload.reopenScopes = reopenScopes;
  result.payload.allowsFinalization = false;
  result.payload.finalizationRequirements = [];
  result.payload.rationale =
    "The exact supplied blockers require reopening before completion.";
  return result;
}

function otherDimension(
  dimension: KnowledgeDimension,
): KnowledgeDimension {
  return (
    ["idea", "knowledge", "experiment", "human", "paper"] as const
  ).find((candidate) => candidate !== dimension)!;
}
