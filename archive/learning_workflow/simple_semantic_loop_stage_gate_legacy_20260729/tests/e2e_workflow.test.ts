import assert from "node:assert/strict";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { test } from "node:test";
import {
  CLOSURE_CHECK_NAMES,
  FINALIZATION_REQUIREMENTS,
  ROLE_REASONING_EFFORT,
  RUBRIC_REGISTRY,
  canonicalSha256,
  sha256Bytes,
  type Anchor,
  type ClosureReviewTaskEnvelope,
  type Direction,
  type DirectionReviewTaskEnvelope,
  type EvidenceReaderTaskEnvelope,
  type ObjectRef,
  type ReviewDeltaEnvelope,
  type SearchNeed,
  type SemanticDelta,
  type StageContractDraft,
  type TopicFrame,
  type WorkflowDecisionProposal,
  type WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  SemanticLoopController,
  type ControllerConfig,
} from "../controller.ts";
import { WorkflowStore } from "../db/workflow_store.ts";
import {
  ScriptedFreshTurnRuntime,
  type FrozenTurnDispatch,
} from "../turns/runtime.ts";
import {
  ALLOWED_EVIDENCE_TOOLS,
  type RuntimeToolEvent,
} from "../security/no_experiment_guard.ts";
import {
  EVIDENCE_BUDGET,
  ZERO_BUDGET,
  makeAnchor,
  makeDirection,
  makeTopic,
} from "./fixtures/factory.ts";

type TerminalDecision = "testable" | "experiment_required";

test("E2E: Topic → dimension-routed Evidence → reviewed Direction → accepted closure", async () => {
  const outcome = await runCompleteScenario("testable");
  try {
    assert.equal(outcome.result.completed, true);
    assert.equal(outcome.store.getRun("run-e2e").lifecycle, "completed");
    assertFinalAndAudit(outcome.workDir, outcome.store, "testable");
  } finally {
    outcome.store.close();
  }
});

test("E2E: experiment-required Direction produces only a non-executable handoff", async () => {
  const outcome = await runCompleteScenario("experiment_required");
  try {
    assert.equal(outcome.result.completed, true);
    const handoffs = outcome.store.query(
      `SELECT object_json FROM canonical_objects
       WHERE run_id = 'run-e2e' AND object_type = 'experiment_handoff'
       AND active = 1`,
    );
    assert.equal(handoffs.length, 1);
    const handoff = JSON.parse(String(handoffs[0]!.object_json));
    assert.equal(handoff.executionAuthorized, false);
    assert.equal(handoff.tag, "EXPERIMENT_REQUIRED");
    assert.equal(
      outcome.store.query(
        `SELECT stage_type FROM stage_contracts
         WHERE run_id = 'run-e2e' AND lower(stage_type) LIKE '%experiment%'`,
      ).length,
      0,
    );
    assert.ok(
      outcome.store
        .query(
          "SELECT role FROM attempts WHERE run_id = 'run-e2e'",
        )
        .every((row) =>
          [
            "workflow_decision",
            "evidence_reader",
            "direction_reviewer",
            "closure_reviewer",
          ].includes(String(row.role)),
        ),
    );
    assertFinalAndAudit(
      outcome.workDir,
      outcome.store,
      "experiment_required",
    );
  } finally {
    outcome.store.close();
  }
});

interface ScenarioOutcome {
  workDir: string;
  store: WorkflowStore;
  result: Awaited<ReturnType<SemanticLoopController["run"]>>;
}

async function runCompleteScenario(
  terminalDecision: TerminalDecision,
): Promise<ScenarioOutcome> {
  const workDir = mkdtempSync(resolve(tmpdir(), "simple-loop-e2e-"));
  const store = new WorkflowStore(resolve(workDir, "workflow.db"));
  store.createRun({
    runId: "run-e2e",
    workflowId: "workflow-e2e",
    objective:
      "Find a traceable performance optimization opportunity for the user topic.",
    acceptanceCriteria: [
      "Preserve topic scope.",
      "Route bounded knowledge gaps to the matching vault dimensions.",
      "Independently review each Direction.",
      "Complete only through closure review and deterministic rendering.",
    ],
    config: {
      maxTurns: 80,
      maxInputTokens: 1_000_000,
      maxOutputTokens: 500_000,
      maxToolCalls: 200,
      maxElapsedMs: 3_600_000,
    },
  });

  const dispatches: FrozenTurnDispatch[] = [];
  const runtime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    dispatches.push(dispatch);
    assert.ok(dispatch.outputSchema);
    const task = parseTaskPacket(dispatch.prompt);
    switch (dispatch.role) {
      case "workflow_decision":
        return JSON.stringify(
          decideWorkflow(task as WorkflowTurnTask, terminalDecision),
        );
      case "evidence_reader": {
        const built = evidenceResult(task as EvidenceReaderTaskEnvelope);
        return {
          text: JSON.stringify(built.result),
          toolEvents: built.toolEvents,
        };
      }
      case "direction_reviewer":
        return JSON.stringify(
          directionReviewResult(
            task as DirectionReviewTaskEnvelope,
            terminalDecision,
          ),
        );
      case "closure_reviewer":
        return JSON.stringify(
          closureAcceptResult(task as ClosureReviewTaskEnvelope),
        );
    }
  });
  const controller = new SemanticLoopController(
    store,
    runtime,
    controllerConfig(workDir),
  );
  try {
    const result = await controller.run("run-e2e");
    assert.equal(result.lifecycle, "completed");
    assert.ok(dispatches.length >= 12);
    assert.equal(
      new Set(dispatches.map((item) => item.attemptId)).size,
      dispatches.length,
    );
    for (const dispatch of dispatches) {
      assert.equal(
        dispatch.logicalEffort,
        ROLE_REASONING_EFFORT[dispatch.role],
      );
      assert.equal(
        dispatch.providerWireEffort,
        dispatch.role === "workflow_decision" ? "max" : "high",
      );
    }
    return { workDir, store, result };
  } catch (error) {
    store.close();
    throw error;
  } finally {
    await runtime.close?.();
  }
}

function controllerConfig(workDir: string): ControllerConfig {
  const projectRoot = resolve(import.meta.dirname, "../../..");
  return {
    projectRoot,
    workDir,
    model: "scripted-model",
    skillRoot: resolve(projectRoot, ".codex/skills"),
    schemaManifestPath: resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/schema_manifest.json",
    ),
    capabilityManifest: {
      provider: "scripted",
      model: "scripted-model",
      wireEffortByLogicalEffort: { high: "high", max: "max" },
      highestWireEffort: "max",
    },
    budgets: {
      workflow: structuredClone(ZERO_BUDGET),
      evidence: {
        ...structuredClone(EVIDENCE_BUDGET),
        maxToolCalls: 8,
        evidenceRead: {
          ...structuredClone(EVIDENCE_BUDGET.evidenceRead!),
          maxSearchToolCalls: 4,
          maxContextsRead: 4,
        },
      },
      direction: structuredClone(ZERO_BUDGET),
      closure: structuredClone(ZERO_BUDGET),
    },
    maxTransitionsPerRun: 100,
    noProgressThreshold: 5,
  };
}

function decideWorkflow(
  task: WorkflowTurnTask,
  terminalDecision: TerminalDecision,
): WorkflowDecisionProposal {
  if (task.trigger === "INITIALIZE_TOPIC") {
    return topicProposal(task);
  }
  if (task.trigger === "COMMITTED_RESULT_REQUIRES_INTEGRATION") {
    const ref = task.resultIndex.committedUnconsumedResultRefs[0];
    assert.ok(ref, "integration trigger supplies one unconsumed result");
    const packet = task.domainProjection.evidencePackets.find(
      (item) => item.packetId === ref.objectId,
    );
    if (packet?.needId === "need-anchor") {
      return semanticProposal(
        task,
        ref,
        createAnchor(packet.findings[0]!.evidenceId),
        "create",
      );
    }
    if (packet?.needId === "need-direction") {
      return semanticProposal(
        task,
        ref,
        createDirection(packet.findings[0]!.evidenceId),
        "create",
      );
    }
    if (packet?.needId === "need-expansion") {
      const topic = task.domainProjection.topic!;
      return noDeltaProposal(task, ref, "topic", topic.topicId, topic.revision);
    }
    const review = task.domainProjection.directionReviews.find(
      (item) => item.reviewId === ref.objectId,
    );
    assert.ok(review, `unconsumed result ${ref.objectId} has a projection`);
    if (review.reviewId === "review-primary") {
      const direction = structuredClone(task.domainProjection.focusDirection!);
      direction.revision += 1;
      direction.status = terminalDecision;
      direction.statusReason =
        terminalDecision === "testable"
          ? "Independent review found the bounded hypothesis testable."
          : "Only a new external measurement can resolve the remaining claim.";
      direction.experimentHandoffId =
        terminalDecision === "experiment_required" ? "handoff-primary" : null;
      return semanticProposal(task, ref, direction, "revise");
    }
    const anchor = structuredClone(task.domainProjection.focusAnchor!);
    anchor.revision += 1;
    anchor.directionIds = ["direction-1"];
    anchor.status = "saturated";
    anchor.statusReason = "";
    anchor.saturationReason =
      "The bounded Direction was independently reviewed and the final topic expansion was scheduled.";
    return semanticProposal(task, ref, anchor, "revise");
  }

  assert.ok(
    [
      "PLAN_EXHAUSTED_OBJECTIVE_OPEN",
      "NO_RUNNABLE_STAGE",
      "FRONTIER_SELECTION_REQUIRED",
    ].includes(task.trigger),
    `unexpected trigger ${task.trigger}`,
  );
  const anchor = task.domainProjection.focusAnchor;
  const direction = task.domainProjection.focusDirection;
  const needs = task.domainProjection.searchNeeds;
  const reviews = task.domainProjection.directionReviews;
  if (!anchor && !needs.some((need) => need.needId === "need-anchor")) {
    return searchNeedProposal(task, anchorNeed());
  }
  if (
    anchor &&
    !direction &&
    !needs.some((need) => need.needId === "need-direction")
  ) {
    return searchNeedProposal(task, directionNeed());
  }
  if (direction && reviews.length === 0) {
    return reviewRequestProposal(task, direction, "initial");
  }
  if (
    direction &&
    ["testable", "experiment_required"].includes(direction.status) &&
    reviews.length === 1
  ) {
    return reviewRequestProposal(task, direction, "terminal_check");
  }
  if (
    anchor?.status === "saturated" &&
    !needs.some((need) => need.needId === "need-expansion")
  ) {
    return searchNeedProposal(task, expansionNeed());
  }
  const completion = task.domainProjection.completionProjection;
  assert.ok(completion?.eligibleForProposal);
  assert.deepEqual(completion.blockingClaims, []);
  return completionProposal(task, completion);
}

function baseProposal(task: WorkflowTurnTask): WorkflowDecisionProposal {
  return {
    protocolVersion: 1,
    messageType: "WORKFLOW_DECISION_PROPOSAL",
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    expectedState: task.stateSnapshot,
    decisionInputHash: task.decisionInputHash,
    proposalId: `proposal-${task.taskId}`,
    decision: "RUN_STAGE",
    reason: `Resolve registered trigger ${task.trigger}.`,
    assumptions: [],
    proposedStageContract: null,
    proposedGateDefinition: null,
    proposedPlanPatch: null,
    targetStageId: null,
    domainProposal: null,
    basisArtifactRefs: [],
    basisResultRefs: [],
    requestedUserInput: null,
    blockedReport: null,
    pauseProposal: null,
    confidence: null,
  };
}

function topicProposal(task: WorkflowTurnTask): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const contract = scriptContract(task, "SCRIPT_APPLY_TOPIC_FRAME");
  return {
    ...proposal,
    proposedStageContract: contract,
    proposedGateDefinition: gate(contract, "TopicFrame"),
    domainProposal: { kind: "topic_frame", value: makeTopic() },
  };
}

function searchNeedProposal(
  task: WorkflowTurnTask,
  need: SearchNeed,
): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const local = `evidence-${task.taskId}`;
  const budget = task.permission.maxBudgetByRole.evidence_reader!;
  const paths = need.targetDimensions.map(
    (dimension) =>
      ({
        idea: "idea_notes/",
        knowledge: "knowledge_notes/",
        experiment: "experiment_notes/",
        human: "human_notes/",
        paper: "paper_secs/",
      })[dimension],
  );
  const contract: StageContractDraft = {
    proposalLocalStageKey: local,
    stageType: "EVIDENCE_READ",
    objective: need.question,
    scope: [
      {
        objectType: "search_need",
        objectId: need.needId,
        revision: need.revision,
      },
    ],
    executionKind: "WORKER_TURN",
    role: "evidence_reader",
    requiredInputs: [],
    expectedOutputMessageType: "EVIDENCE_PACKET",
    requestedTools: [...ALLOWED_EVIDENCE_TOOLS],
    requestedPaths: paths,
    prohibitedActions: [
      "vault write",
      "shell execution",
      "agent delegation",
      "experiment execution",
    ],
    budget,
  };
  const evidenceGate = gate(contract, "EVIDENCE_PACKET");
  evidenceGate.mechanicalChecks.push(
    {
      checkId: "need-revision",
      predicate: "equals",
      actual: {
        source: "canonical",
        objectRef: {
          objectType: "search_need",
          objectId: need.needId,
          revision: need.revision,
        },
        pointer: "/revision",
        valueType: "number",
      },
      expected: need.revision,
    },
    {
      checkId: "bound-need-revision",
      predicate: "equals",
      actual: {
        source: "result",
        pointer: "/payload/needRevision",
        valueType: "number",
      },
      expected: need.revision,
    },
    {
      checkId: "bound-need-id",
      predicate: "equals",
      actual: {
        source: "result",
        pointer: "/payload/needId",
        valueType: "string",
      },
      expected: need.needId,
    },
    {
      checkId: "stage-input-primary-dimension",
      predicate: "equals",
      actual: {
        source: "task",
        pointer: "/payload/searchNeed/primaryDimension",
        valueType: "string",
      },
      expected: need.primaryDimension,
    },
    {
      checkId: "stage-input-target-dimensions",
      predicate: "equals",
      actual: {
        source: "task",
        pointer: "/payload/searchNeed/targetDimensions",
        valueType: "string_array",
      },
      expected: need.targetDimensions,
    },
  );
  return {
    ...proposal,
    proposedStageContract: contract,
    proposedGateDefinition: evidenceGate,
    domainProposal: { kind: "search_need", value: need },
  };
}

function reviewRequestProposal(
  task: WorkflowTurnTask,
  direction: Direction,
  purpose: "initial" | "terminal_check",
): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const local = `review-${task.taskId}`;
  const contract: StageContractDraft = {
    proposalLocalStageKey: local,
    stageType: "DIRECTION_REVIEW",
    objective: `Independently review Direction ${direction.directionId}.`,
    scope: [
      {
        objectType: "direction",
        objectId: direction.directionId,
        revision: direction.revision,
      },
    ],
    executionKind: "EVALUATOR_TURN",
    role: "direction_reviewer",
    requiredInputs: [],
    expectedOutputMessageType: "REVIEW_DELTA",
    requestedTools: [],
    requestedPaths: [],
    prohibitedActions: [
      "search",
      "state write",
      "agent delegation",
      "experiment execution",
    ],
    budget: task.permission.maxBudgetByRole.direction_reviewer!,
  };
  return {
    ...proposal,
    decision: "REQUEST_EVALUATION",
    proposedStageContract: contract,
    proposedGateDefinition: gate(contract, "REVIEW_DELTA"),
    domainProposal: {
      kind: "direction_review_request",
      value: {
        directionRef: {
          objectType: "direction",
          objectId: direction.directionId,
          revision: direction.revision,
        },
        purpose,
        rubric: RUBRIC_REGISTRY.direction_readiness_v1,
      },
    },
  };
}

function semanticProposal(
  task: WorkflowTurnTask,
  basis: ObjectRef,
  object: TopicFrame | Anchor | Direction,
  action: "create" | "revise",
): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const contract = scriptContract(task, "SCRIPT_APPLY_SEMANTIC_DELTA");
  const identity =
    "directionId" in object
      ? { type: "direction" as const, id: object.directionId }
      : "anchorId" in object
        ? { type: "anchor" as const, id: object.anchorId }
        : { type: "topic" as const, id: object.topicId };
  const delta: SemanticDelta = {
    deltaId: `delta-${task.taskId}`,
    basisResultRefs: [basis],
    expectedTargetRevision: action === "create" ? 0 : object.revision - 1,
    target: identity,
    action,
    changedFields:
      action === "create"
        ? ["create"]
        : identity.type === "anchor"
          ? ["status", "saturationReason", "directionIds"]
          : ["status", "statusReason", "experimentHandoffId"],
    rationale: `Integrate ${basis.objectId} into ${identity.type}.`,
    proposedObject: object,
  };
  return {
    ...proposal,
    proposedStageContract: contract,
    proposedGateDefinition: gate(contract, "SemanticDelta"),
    domainProposal: { kind: "semantic_delta", value: delta },
    basisResultRefs: [basis],
  };
}

function noDeltaProposal(
  task: WorkflowTurnTask,
  basis: ObjectRef,
  type: "topic",
  id: string,
  revision: number,
): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const contract = scriptContract(task, "SCRIPT_APPLY_SEMANTIC_DELTA");
  return {
    ...proposal,
    proposedStageContract: contract,
    proposedGateDefinition: gate(contract, "SemanticDelta"),
    domainProposal: {
      kind: "semantic_delta",
      value: {
        deltaId: `delta-${task.taskId}`,
        basisResultRefs: [basis],
        expectedTargetRevision: revision,
        target: { type, id },
        action: "no_semantic_delta",
        changedFields: [],
        rationale:
          "The final topic expansion found no new Anchor or critical semantic delta.",
        proposedObject: null,
      },
    },
    basisResultRefs: [basis],
  };
}

function completionProposal(
  task: WorkflowTurnTask,
  completion: NonNullable<
    WorkflowTurnTask["domainProjection"]["completionProjection"]
  >,
): WorkflowDecisionProposal {
  const proposal = baseProposal(task);
  const {
    eligibleForProposal: _eligible,
    blockingClaims: _blocking,
    ...proofFacts
  } = completion;
  const stopCandidateId = `stop-${task.taskId}`;
  const proofId = `proof-${task.taskId}`;
  return {
    ...proposal,
    decision: "PROPOSE_COMPLETE",
    proposedStageContract: null,
    proposedGateDefinition: null,
    domainProposal: {
      kind: "stop_candidate",
      value: {
        candidate: {
          stopCandidateId,
          stopProofId: proofId,
          runId: task.runId,
          topicId: task.domainProjection.topic!.topicId,
          canonicalRevision: task.stateSnapshot.canonicalRevision,
          reason:
            "The exact Controller completion projection has no blocking claim.",
        },
        proof: {
          proofId,
          stopCandidateId,
          ...proofFacts,
        },
      },
    },
  };
}

function scriptContract(
  task: WorkflowTurnTask,
  stageType:
    | "SCRIPT_APPLY_TOPIC_FRAME"
    | "SCRIPT_APPLY_SEMANTIC_DELTA",
): StageContractDraft {
  return {
    proposalLocalStageKey: `script-${task.taskId}`,
    stageType,
    objective:
      stageType === "SCRIPT_APPLY_TOPIC_FRAME"
        ? "Commit the validated TopicFrame."
        : "Commit one validated SemanticDelta.",
    scope: [],
    executionKind: "SCRIPT_TRANSITION",
    role: null,
    requiredInputs: [],
    expectedOutputMessageType: null,
    requestedTools: [],
    requestedPaths: [],
    prohibitedActions: [
      "provider call",
      "state write outside Controller transaction",
      "experiment execution",
    ],
    budget: task.permission.maxBudgetByRole.workflow_decision!,
  };
}

function gate(contract: StageContractDraft, subject: string) {
  void subject;
  return {
    proposalLocalStageKey: contract.proposalLocalStageKey,
    mechanicalChecks: [],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  };
}

function anchorNeed(): SearchNeed {
  return {
    ...baseNeed("need-anchor", "define_baseline"),
    question:
      "What is the baseline execution path and bottleneck for batch-1 decode?",
    successCriteria: [
      "A deep-read source states the baseline path and performance tension.",
    ],
    primaryDimension: "idea",
    auxiliaryDimension: "experiment",
    targetDimensions: ["idea", "experiment"],
    technicalObjects: ["decode scheduler"],
    performanceRelations: ["baseline latency"],
    evidenceIntentTerms: ["baseline"],
  };
}

function directionNeed(): SearchNeed {
  return {
    ...baseNeed("need-direction", "find_modification"),
    owner: {
      topicId: "topic-1",
      anchorId: "anchor-1",
      directionId: null,
    },
    question:
      "What bounded runtime change can reduce launch overhead in batch-1 decode?",
    successCriteria: [
      "A deep-read source states a bounded modification and its mechanism.",
    ],
    primaryDimension: "idea",
    auxiliaryDimension: "knowledge",
    targetDimensions: ["idea", "knowledge"],
    technicalObjects: ["launch fusion"],
    performanceRelations: ["lower launch overhead"],
    evidenceIntentTerms: ["modification"],
  };
}

function expansionNeed(): SearchNeed {
  return {
    ...baseNeed("need-expansion", "discover_anchor"),
    question:
      "Does the bounded topic contain another distinct batch-1 decode baseline tension?",
    successCriteria: [
      "A deep-read source establishes a distinct in-scope Anchor.",
    ],
    primaryDimension: "idea",
    auxiliaryDimension: "human",
    targetDimensions: ["idea", "human"],
    technicalObjects: ["decode baseline"],
    performanceRelations: ["performance tension"],
    evidenceIntentTerms: ["scenario"],
  };
}

function baseNeed(
  needId: string,
  intent: SearchNeed["intent"],
): SearchNeed {
  return {
    needId,
    revision: 1,
    owner: {
      topicId: "topic-1",
      anchorId: null,
      directionId: null,
    },
    intent,
    question: "bounded question",
    rationale: "One explicit semantic gap remains.",
    successCriteria: ["One exact criterion."],
    primaryDimension: "idea",
    auxiliaryDimension: null,
    targetDimensions: ["idea"],
    queryVariants: [],
    technicalObjects: ["decode"],
    knownTerms: ["autoregressive"],
    synonymGroups: [["decode", "token generation"]],
    scenarioTerms: ["batch-1 decode"],
    performanceRelations: ["latency"],
    evidenceIntentTerms: ["mechanism"],
    excludedSourceUnits: [],
    previousAttemptIds: [],
    critical: true,
    answerability: "knowledge_base",
    status: "pending",
  };
}

function createAnchor(evidenceId: string): Anchor {
  const anchor = makeAnchor("active");
  anchor.evidenceRefs = [evidenceId];
  anchor.directionIds = [];
  anchor.openNeedIds = [];
  return anchor;
}

function createDirection(evidenceId: string): Direction {
  const direction = makeDirection("exploring");
  direction.supportingEvidenceRefs = [evidenceId];
  direction.changes[0]!.evidenceRefs = [evidenceId];
  direction.causalLinks[0]!.evidenceRefs = [evidenceId];
  direction.degradationConditions = [];
  return direction;
}

function evidenceResult(task: EvidenceReaderTaskEnvelope): {
  result: unknown;
  toolEvents: RuntimeToolEvent[];
} {
  const need = task.payload.searchNeed;
  const dimension = need.primaryDimension;
  const prefix = task.payload.allowedVaultRoots.find(
    (root) => root.dimension === dimension,
  )!.relativePathPrefix;
  const searchId = `search-${need.needId}`;
  const queryTerms = [
    need.technicalObjects[0]!,
    need.scenarioTerms[0]!,
    need.evidenceIntentTerms[0]!,
  ];
  const query = `path:${prefix} ${queryTerms.join(" ")}`;
  const search = {
    searchId,
    sequence: 1,
    logicalQueryLevel: 1 as const,
    dimension,
    query,
    pathFilter: `path:${prefix}`,
    terms: [
      {
        term: queryTerms[0]!,
        source: "task" as const,
        sourceRef: "searchNeed.technicalObjects",
        introducedAtSequence: 0,
      },
      {
        term: queryTerms[1]!,
        source: "task" as const,
        sourceRef: "searchNeed.scenarioTerms",
        introducedAtSequence: 0,
      },
      {
        term: queryTerms[2]!,
        source: "task" as const,
        sourceRef: "searchNeed.evidenceIntentTerms",
        introducedAtSequence: 0,
      },
    ],
    page: 1,
    cursorUsed: null,
    nextCursor: null,
    toolCallIndex: 1,
    pageHitCount: need.needId === "need-expansion" ? 0 : 1,
    cumulativeHitCount: need.needId === "need-expansion" ? 0 : 1,
    outcome:
      need.needId === "need-expansion"
        ? ("no_hits" as const)
        : ("hits" as const),
    stopReason:
      need.needId === "need-expansion"
        ? ("no_more_pages" as const)
        : ("success_criteria_met" as const),
  };
  const common = {
    protocolVersion: 1 as const,
    messageType: "EVIDENCE_PACKET" as const,
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
  };
  const searchEvent: RuntimeToolEvent = {
    toolName: "mcp__obsidian__obsidian_search_notes",
    arguments: { mode: "omnisearch", query },
    status: "completed",
    resultText:
      need.needId === "need-expansion"
        ? "[]"
        : `[{"path":"${prefix}${need.needId}.md"}]`,
    error: null,
  };
  if (need.needId === "need-expansion") {
    return {
      result: {
        ...common,
        payload: {
          packetId: `packet-${need.needId}`,
          needId: need.needId,
          needRevision: need.revision,
          status: "complete",
          searches: [search],
          hitsConsidered: [],
          contextsRead: [],
          findings: [],
          contradictions: [],
          unanswered: need.successCriteria.map((criterion, index) => ({
            unansweredId: `unanswered-${index + 1}`,
            successCriterion: criterion,
            reason: "no_matching_source",
            attemptedSearchIds: [searchId],
          })),
          conclusion: "not_found",
          conclusionRationale:
            "The bounded Q1 returned no source and no wider query can preserve the exact topic boundary.",
        },
      },
      toolEvents: [searchEvent],
    };
  }

  const path = `${prefix}${need.needId}.md`;
  const heading = "Bounded evidence";
  const exactContext =
    need.needId === "need-anchor"
      ? "The batch-1 decode scheduler launches sequential kernels; launch overhead is the baseline latency tension."
      : "Fusing adjacent compatible decode launches is a bounded runtime change that removes launch overhead.";
  const sourceUnitId = `unit-${need.needId}`;
  const findingId =
    need.needId === "need-anchor" ? "ev-baseline" : "ev-modification";
  const hit = {
    hitId: `hit-${need.needId}`,
    searchId,
    sequence: 2,
    path,
    score: 9,
    sourceFamily: `family-${need.needId}`,
    selected: true,
    selectionReason: "Directly matches the frozen success criterion.",
  };
  const contexts = [
    {
      contextId: `map-${need.needId}`,
      sequence: 3,
      hitId: hit.hitId,
      path,
      format: "document-map" as const,
      heading: "",
      sectionTarget: null,
      sourceUnitId: `map-${need.needId}`,
      sourceFamily: hit.sourceFamily,
      exactContext: `## ${heading}`,
      summary: "A relevant section is present.",
    },
    {
      contextId: `context-${need.needId}`,
      sequence: 4,
      hitId: hit.hitId,
      path,
      format: "section" as const,
      heading,
      sectionTarget: heading,
      sourceUnitId,
      sourceFamily: hit.sourceFamily,
      exactContext,
      summary: "The section directly answers the bounded question.",
    },
  ];
  const finding = {
    evidenceId: findingId,
    claimKey: `claim-${need.needId}`,
    claim: exactContext,
    evidenceRole:
      need.needId === "need-anchor"
        ? ("baseline" as const)
        : ("modification" as const),
    directness: "direct" as const,
    attribution: "source_report" as const,
    applicableConditions: ["batch-1 decode"],
    comparisonBaseline: "sequential launches",
    sourcePath: path,
    sourceUnitId,
    sourceFamily: hit.sourceFamily,
    heading,
    quoteOrExactContext: exactContext,
  };
  return {
    result: {
      ...common,
      payload: {
        packetId: `packet-${need.needId}`,
        needId: need.needId,
        needRevision: need.revision,
        status: "complete",
        searches: [search],
        hitsConsidered: [hit],
        contextsRead: contexts,
        findings: [finding],
        contradictions: [],
        unanswered: [],
        conclusion: "answered",
        conclusionRationale:
          "One selected source was mapped and deep-read and directly meets the criterion.",
      },
    },
    toolEvents: [
      searchEvent,
      {
        toolName: "mcp__obsidian__obsidian_get_note",
        arguments: {
          target: { type: "path", path },
          format: "document-map",
        },
        status: "completed",
        resultText: contexts[0]!.exactContext,
        error: null,
      },
      {
        toolName: "mcp__obsidian__obsidian_get_note",
        arguments: {
          target: { type: "path", path },
          format: "section",
          section: { type: "heading", target: heading },
        },
        status: "completed",
        resultText: exactContext,
        error: null,
      },
    ],
  };
}

function directionReviewResult(
  task: DirectionReviewTaskEnvelope,
  terminalDecision: TerminalDecision,
): ReviewDeltaEnvelope {
  const primary = task.payload.priorReviews.length === 0;
  const decision = primary ? terminalDecision : "testable";
  const handoff =
    decision === "experiment_required"
      ? {
          handoffId: "handoff-primary",
          directionId: task.payload.direction.directionId,
          tag: "EXPERIMENT_REQUIRED" as const,
          reason:
            "A new controlled latency measurement is outside this read-only workflow.",
          requiredArtifact: "benchmark" as const,
          hypothesisToTest:
            "Launch fusion lowers batch-1 per-token latency under equal outputs.",
          suggestedEntryPoints: ["runtime scheduler decode dispatch"],
          controlledVariables: ["model", "input tokens", "output tokens"],
          metrics: ["per-token latency", "launch count"],
          acceptanceCriteria: [
            "The external artifact compares the same model and token sequence.",
          ],
          failureStopConditions: [
            "Stop if semantic equivalence cannot be established externally.",
          ],
          executionAuthorized: false as const,
        }
      : null;
  return {
    protocolVersion: 1,
    messageType: "REVIEW_DELTA",
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
    payload: {
      reviewId: primary ? "review-primary" : "review-terminal",
      directionId: task.payload.direction.directionId,
      directionRevision: task.payload.direction.revision,
      supportedParts: [
        "scope",
        "baseline",
        "change",
        "causal chain",
        "implementation",
        "measurement",
      ],
      evidenceRefsUsed: [...task.payload.allowedEvidenceIds],
      weakestCausalLink: null,
      baselineProblem: null,
      implementationProblem: null,
      measurementProblem: null,
      strongestCounterexample: null,
      counterexampleResolution: null,
      nextQuestion: null,
      nextQuestionAnswerableFromKnowledgeBase: false,
      decision,
      rationale:
        decision === "experiment_required"
          ? "The semantic Direction is bounded, but only an external new measurement can settle the effect."
          : "All eleven readiness checks pass against committed evidence.",
      duplicateOfDirectionRef: null,
      duplicateComparison: null,
      rejectionCategory: null,
      readinessChecks: {
        inTopicAndAnchorScope: true,
        baselineFair: true,
        minimumChangeSetExplicit: true,
        causalChainFalsifiable: true,
        implementationPathBounded: true,
        measurementPlanComplete: true,
        falsifiersPresent: true,
        criticalCounterexampleResolved: true,
        evidenceTraceable: true,
        knowledgeAnswerableCriticalGapRemaining: false,
        newExperimentRequired: decision === "experiment_required",
      },
      experimentHandoff: handoff,
    },
  };
}

function closureAcceptResult(task: ClosureReviewTaskEnvelope) {
  const runRef = {
    objectType: "run",
    objectId: task.runId,
    revision: null,
  };
  return {
    protocolVersion: 1 as const,
    messageType: "CLOSURE_REVIEW" as const,
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
    payload: {
      reviewId: "closure-accept",
      stopCandidateId:
        task.payload.stopCandidateBundle.candidate.stopCandidateId,
      canonicalRevision: task.payload.currentCanonicalRevision,
      status: "complete" as const,
      decision: "accept" as const,
      verifiedClosureBasis: CLOSURE_CHECK_NAMES.map((check) => ({
        check,
        statement: `${check} is true in the supplied canonical projection.`,
        objectRefs: [runRef],
      })),
      closureChecks: Object.fromEntries(
        CLOSURE_CHECK_NAMES.map((check) => [check, true]),
      ),
      blockingFindings: [],
      reopenScopes: [],
      allowsFinalization: true,
      finalizationRequirements: [...FINALIZATION_REQUIREMENTS],
      rationale:
        "All thirteen checks independently pass; only the Controller may finalize.",
    },
  };
}

function parseTaskPacket(prompt: string): unknown {
  const match = prompt.match(
    /\[TASK_PACKET\]\n([\s\S]*?)\n\n\[EXPECTED_OUTPUT_SCHEMA\]/,
  );
  assert.ok(match, "prompt contains exactly one task packet section");
  return JSON.parse(match[1]!);
}

function assertFinalAndAudit(
  workDir: string,
  store: WorkflowStore,
  terminalDecision: TerminalDecision,
): void {
  const finalPath = resolve(workDir, "final.md");
  assert.equal(existsSync(finalPath), true);
  const final = readFileSync(finalPath, "utf8");
  for (const text of [
    "## Topic scope",
    "## Performance opportunity overview",
    "## Anchor summaries",
    "## Direction statuses",
    "## Evidence provenance",
    "## Contradictions and limits",
    "## Experiment handoffs",
    "## Unresolved questions",
    "anchor-1",
    "direction-1",
    "atom-1",
    "#### Causal chain and expected effects",
    "#### Fair comparison, implementation, and falsification",
    "ev-baseline",
    "ev-modification",
  ]) {
    assert.ok(final.includes(text), `final output contains ${text}`);
  }
  if (terminalDecision === "experiment_required") {
    assert.ok(final.includes("handoff-primary"));
    assert.ok(final.includes("executionAuthorized=`false`"));
  }
  for (const path of [
    "exports/workflow_state.json",
    "exports/workflow_plan.json",
    "exports/tasks.jsonl",
    "exports/attempts.jsonl",
    "exports/events.jsonl",
    "artifacts/manifest.jsonl",
  ]) {
    assert.equal(existsSync(resolve(workDir, path)), true, path);
  }
  const attempts = store.query(
    `SELECT role, logical_effort, provider_thread_id
     FROM attempts WHERE run_id = 'run-e2e' ORDER BY started_at`,
  );
  assert.ok(attempts.length > 0);
  assert.equal(
    new Set(attempts.map((row) => String(row.provider_thread_id))).size,
    attempts.length,
  );
  for (const attempt of attempts) {
    assert.equal(
      attempt.logical_effort,
      ROLE_REASONING_EFFORT[
        attempt.role as keyof typeof ROLE_REASONING_EFFORT
      ],
    );
  }
  const finalArtifact = store.query(
    `SELECT sha256, relative_path FROM artifact_manifests
     WHERE run_id = 'run-e2e' AND kind = 'final_markdown'`,
  );
  assert.equal(finalArtifact.length, 1);
  assert.equal(finalArtifact[0]!.relative_path, "final.md");
  assert.equal(finalArtifact[0]!.sha256, sha256Bytes(final));
}
