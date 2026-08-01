import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ROLE_REASONING_EFFORT,
  TRIGGER_ALLOWED_ACTIONS,
  canonicalJson,
  type WorkflowDecisionAction,
  type WorkflowState,
  type WorkflowTrigger,
} from "../contracts/index.ts";
import {
  ZERO_BUDGET,
  EVIDENCE_BUDGET,
  makeClosureTask,
  makeEvidenceTask,
  makeTopic,
  makeWorkflowProposal,
  makeWorkflowTask,
} from "./fixtures/factory.ts";
import {
  buildRegisteredTrigger,
  type TriggerSignals,
} from "../workflow/trigger_engine.ts";
import { defaultWorkflowPermission } from "../turns/task_factory.ts";
import { validateWorkflowDecisionProposal } from "../validators/workflow_proposal_validator.ts";

test("all eleven semantic trigger points are deterministic and permission-bound", () => {
  const fixtures: Array<{
    expected: WorkflowTrigger;
    lifecycle?: WorkflowState["lifecycle"];
    signals: Partial<TriggerSignals>;
  }> = [
    {
      expected: "INITIALIZE_TOPIC",
      lifecycle: "initialized",
      signals: { hasTopicFrame: false },
    },
    {
      expected: "USER_DECISION_REQUIRED",
      signals: { userDecisionRequired: true },
    },
    {
      expected: "CLOSURE_REJECTED",
      signals: { closureRejected: true },
    },
    {
      expected: "GATE_FAILED_WITHOUT_RECOVERY_RULE",
      signals: { gateFailedWithoutRecovery: true },
    },
    {
      expected: "EVIDENCE_CONTRADICTION",
      signals: { criticalEvidenceContradiction: true },
    },
    {
      expected: "COMMITTED_RESULT_REQUIRES_INTEGRATION",
      signals: { committedResultRequiresIntegration: true },
    },
    {
      expected: "NO_PROGRESS_THRESHOLD_REACHED",
      signals: { noProgressThresholdReached: true },
    },
    {
      expected: "MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE",
      signals: { nonEquivalentRunnableStageCount: 2 },
    },
    {
      expected: "FRONTIER_SELECTION_REQUIRED",
      signals: { nonEquivalentFrontierCount: 2 },
    },
    {
      expected: "PLAN_EXHAUSTED_OBJECTIVE_OPEN",
      signals: { planExhaustedObjectiveOpen: true },
    },
    {
      expected: "NO_RUNNABLE_STAGE",
      signals: { hasRunnableStage: false },
    },
  ];
  assert.equal(fixtures.length, 11);

  for (const fixture of fixtures) {
    const trigger = buildRegisteredTrigger(
      state(fixture.lifecycle ?? "running"),
      signals(fixture.signals),
    );
    assert.equal(trigger?.trigger, fixture.expected);
    const permission = defaultWorkflowPermission(trigger!, {
      workflow: ZERO_BUDGET,
      evidence: EVIDENCE_BUDGET,
      direction: ZERO_BUDGET,
    });
    assert.deepEqual(
      permission.allowedActions,
      [...TRIGGER_ALLOWED_ACTIONS[fixture.expected]],
    );
    assert.ok(!permission.allowedRoles.includes("workflow_decision"));
    assert.ok(!permission.allowedRoles.includes("closure_reviewer"));
    assert.ok(!permission.allowedStageTypes.includes("WORKFLOW_DECISION"));
    assert.ok(!permission.allowedStageTypes.includes("CLOSURE_REVIEW"));
    assert.ok(!permission.allowedStageTypes.includes("RENDER_FINAL"));
  }
});

test("trigger precedence is fixed when several semantic signals coexist", () => {
  const trigger = buildRegisteredTrigger(
    state("running"),
    signals({
      userDecisionRequired: true,
      closureRejected: true,
      gateFailedWithoutRecovery: true,
      committedResultRequiresIntegration: true,
      noProgressThresholdReached: true,
    }),
  );
  assert.equal(trigger?.trigger, "USER_DECISION_REQUIRED");
});

test("every trigger rejects an action outside both trigger and task allowlists", () => {
  const allActions: WorkflowDecisionAction[] = [
    "RUN_STAGE",
    "RETRY_STAGE",
    "REPLAN",
    "REQUEST_EVALUATION",
    "ASK_USER",
    "REPORT_BLOCKED",
    "PROPOSE_PAUSE",
    "PROPOSE_COMPLETE",
  ];
  for (const trigger of Object.keys(
    TRIGGER_ALLOWED_ACTIONS,
  ) as WorkflowTrigger[]) {
    const task = makeWorkflowTask();
    task.trigger = trigger;
    task.triggerReport.trigger = trigger;
    task.permission.allowedActions = [
      ...TRIGGER_ALLOWED_ACTIONS[trigger],
    ];
    const forbidden = allActions.find(
      (action) => !task.permission.allowedActions.includes(action),
    );
    if (!forbidden) continue;
    const proposal = makeWorkflowProposal();
    proposal.decision = forbidden;
    const report = validateWorkflowDecisionProposal(proposal, task);
    assert.ok(
      report.errors.some(
        (error) => error.code === "workflow.action_not_allowed",
      ),
      `${trigger} rejects ${forbidden}: ${canonicalJson(report.errors)}`,
    );
  }
});

test("PROPOSE_COMPLETE must copy the exact Controller completion projection", () => {
  const task = makeWorkflowTask();
  const closure = makeClosureTask();
  const proof = closure.payload.stopCandidateBundle.proof;
  const {
    proofId: _proofId,
    stopCandidateId: _stopCandidateId,
    ...facts
  } = proof;
  task.trigger = "FRONTIER_SELECTION_REQUIRED";
  task.triggerReport.trigger = task.trigger;
  task.permission.allowedActions = [
    ...TRIGGER_ALLOWED_ACTIONS[task.trigger],
  ];
  task.domainProjection.topic = makeTopic();
  task.domainProjection.completionProjection = {
    ...structuredClone(facts),
    eligibleForProposal: true,
    blockingClaims: [],
  };
  const consumedBasis = {
    objectType: "turn_result",
    objectId: "evidence-consumed-1",
    revision: 1,
  };
  task.resultIndex.consumedResultRefs = [consumedBasis];
  task.permission.suppliedResultRefs = [consumedBasis];
  const proposal = makeWorkflowProposal();
  proposal.decision = "PROPOSE_COMPLETE";
  proposal.proposedStageContract = null;
  proposal.proposedGateDefinition = null;
  proposal.domainProposal = {
    kind: "stop_candidate",
    value: structuredClone(closure.payload.stopCandidateBundle),
  };
  proposal.basisResultRefs = [consumedBasis];

  const valid = validateWorkflowDecisionProposal(proposal, task);
  assert.equal(valid.valid, true, canonicalJson(valid.errors));

  const tampered = structuredClone(proposal);
  if (tampered.domainProposal?.kind !== "stop_candidate") {
    throw new Error("fixture lost stop candidate");
  }
  tampered.domainProposal.value.proof.outputCoverageProjectionId =
    "fabricated-coverage";
  const invalid = validateWorkflowDecisionProposal(tampered, task);
  assert.equal(invalid.valid, false);
  assert.ok(
    invalid.errors.some(
      (error) => error.code === "workflow.domain_proposal_binding",
    ),
  );
});

test("EVIDENCE_READ can bind one supplied pending SearchNeed without recreating it", () => {
  const task = makeWorkflowTask();
  const need = makeEvidenceTask().payload.searchNeed;
  const topic = makeTopic();
  const needRef = {
    objectType: "search_need" as const,
    objectId: need.needId,
    revision: need.revision,
  };
  const topicRef = {
    objectType: "topic" as const,
    objectId: topic.topicId,
    revision: topic.revision,
  };
  task.trigger = "PLAN_EXHAUSTED_OBJECTIVE_OPEN";
  task.triggerReport.trigger = task.trigger;
  task.lifecycle = "running";
  task.domainProjection.topic = topic;
  task.domainProjection.searchNeeds = [need];
  task.permission.allowedActions = ["RUN_STAGE"];
  task.permission.allowedStageTypes = ["EVIDENCE_READ"];
  task.permission.allowedRoles = ["evidence_reader"];
  task.permission.allowedTools = [
    "mcp__obsidian__obsidian_search_notes",
    "mcp__obsidian__obsidian_get_note",
  ];
  task.permission.allowedPathPrefixes = ["knowledge_notes/"];
  task.permission.maxBudgetByRole.evidence_reader =
    structuredClone(EVIDENCE_BUDGET);
  task.permission.suppliedObjectRefs = [topicRef, needRef];

  const proposal = makeWorkflowProposal();
  proposal.decision = "RUN_STAGE";
  proposal.proposedStageContract = {
    proposalLocalStageKey: "reuse-need",
    stageType: "EVIDENCE_READ",
    objective: "Execute the supplied pending SearchNeed.",
    scope: [needRef, topicRef],
    executionKind: "WORKER_TURN",
    role: "evidence_reader",
    requiredInputs: [],
    expectedOutputMessageType: "EVIDENCE_PACKET",
    requestedTools: [...task.permission.allowedTools],
    requestedPaths: ["knowledge_notes/"],
    prohibitedActions: [
      "state write",
      "experiment execution",
    ],
    budget: structuredClone(EVIDENCE_BUDGET),
  };
  proposal.proposedGateDefinition = {
    proposalLocalStageKey: "reuse-need",
    mechanicalChecks: [],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  };
  proposal.domainProposal = null;

  const valid = validateWorkflowDecisionProposal(proposal, task);
  assert.equal(valid.valid, true, canonicalJson(valid.errors));

  const unboundGate = structuredClone(proposal);
  unboundGate.proposedGateDefinition!.mechanicalChecks.push({
    checkId: "missing-result-pointer",
    predicate: "equals",
    actual: {
      source: "result",
      pointer: "/whateverTheWorkerSays",
      valueType: "boolean",
    },
    expected: true,
  });
  const unboundGateReport = validateWorkflowDecisionProposal(
    unboundGate,
    task,
  );
  assert.ok(
    unboundGateReport.errors.some(
      (error) =>
        error.code === "gate.pointer_not_found",
    ),
  );

  const contradictoryGate = structuredClone(proposal);
  contradictoryGate.proposedGateDefinition!.mechanicalChecks.push({
    checkId: "allow-one-experiment",
    predicate: "equals",
    actual: {
      source: "runtime",
      fact: "experiment_execution_count",
      valueType: "number",
    },
    expected: 1,
  });
  const contradictoryGateReport =
    validateWorkflowDecisionProposal(contradictoryGate, task);
  assert.ok(
    contradictoryGateReport.errors.some(
      (error) =>
        error.code ===
        "gate.contradicts_controller_mandatory_check",
    ),
  );

  const missingNewNeedScope = structuredClone(proposal);
  const proposedNeed = {
    ...structuredClone(need),
    needId: "need-new-proposal",
  };
  missingNewNeedScope.domainProposal = {
    kind: "search_need",
    value: proposedNeed,
  };
  missingNewNeedScope.proposedStageContract!.scope = [];
  const missingNewNeedScopeReport =
    validateWorkflowDecisionProposal(missingNewNeedScope, task);
  assert.ok(
    missingNewNeedScopeReport.errors.some(
      (error) =>
        error.code === "workflow.evidence_stage_need_scope",
    ),
  );

  const fabricated = structuredClone(proposal);
  fabricated.proposedStageContract!.scope[0]!.objectId =
    "need-fabricated";
  const invalid = validateWorkflowDecisionProposal(fabricated, task);
  assert.ok(
    invalid.errors.some(
      (error) =>
        error.code === "workflow.fabricated_stage_scope_ref" ||
        error.code === "workflow.evidence_stage_need_scope",
    ),
  );
});

test("role effort policy is immutable: Workflow max, all other Turns high", () => {
  assert.deepEqual(ROLE_REASONING_EFFORT, {
    workflow_decision: "max",
    evidence_reader: "high",
    direction_reviewer: "high",
    closure_reviewer: "high",
  });
  assert.equal(Object.isFrozen(ROLE_REASONING_EFFORT), true);
});

function state(
  lifecycle: WorkflowState["lifecycle"],
): WorkflowState {
  return {
    workflowId: "workflow-1",
    runId: "run-1",
    snapshotVersion: 1,
    canonicalRevision: 1,
    eventCursor: 1,
    workflowPlanRevision: 1,
    lifecycle,
    currentStageId: null,
    activeFocusRef: null,
    runnableStageIds: [],
    pendingTaskIds: [],
    inFlightTaskIds: [],
    committedUnconsumedResultIds: [],
    pendingProposalIds: [],
    retryCounters: {},
    noProgressCounters: {},
    budgetState: {
      turnsUsed: 0,
      maxTurns: 10,
      inputTokensUsed: 0,
      maxInputTokens: 10_000,
      outputTokensUsed: 0,
      maxOutputTokens: 10_000,
      toolCallsUsed: 0,
      maxToolCalls: 10,
      elapsedMs: 0,
      maxElapsedMs: 60_000,
      exhaustedDimensions: [],
    },
    pauseOrBlockReason: null,
  };
}

function signals(overrides: Partial<TriggerSignals>): TriggerSignals {
  return {
    hasTopicFrame: true,
    committedResultRequiresIntegration: false,
    nonEquivalentFrontierCount: 0,
    nonEquivalentRunnableStageCount: 0,
    gateFailedWithoutRecovery: false,
    planExhaustedObjectiveOpen: false,
    criticalEvidenceContradiction: false,
    noProgressThresholdReached: false,
    closureRejected: false,
    userDecisionRequired: false,
    hasPendingOrInFlight: false,
    hasRunnableStage: true,
    ...overrides,
  };
}
