import {
  CLOSURE_CHECK_NAMES,
  FINALIZATION_REQUIREMENTS,
  canonicalSha256,
  type Anchor,
  type ClosureReviewEnvelope,
  type ClosureReviewTaskEnvelope,
  type Direction,
  type DirectionReviewTaskEnvelope,
  type EvidencePacketEnvelope,
  type EvidenceReaderTaskEnvelope,
  type SearchNeed,
  type StateBinding,
  type TopicFrame,
  type TurnBudget,
  type WorkflowDecisionProposal,
  type WorkflowTurnTask,
} from "../../contracts/index.ts";

export const HASH_A = "a".repeat(64);
export const HASH_B = "b".repeat(64);
export const HASH_C = "c".repeat(64);

export const STATE: StateBinding = {
  snapshotVersion: 4,
  canonicalRevision: 3,
  eventCursor: 9,
  workflowPlanRevision: 2,
};

export const ZERO_BUDGET: TurnBudget = {
  timeoutMs: 30_000,
  maxInputTokens: 20_000,
  maxOutputTokens: 8_000,
  maxToolCalls: 0,
  evidenceRead: null,
};

export const EVIDENCE_BUDGET: TurnBudget = {
  timeoutMs: 30_000,
  maxInputTokens: 20_000,
  maxOutputTokens: 8_000,
  maxToolCalls: 6,
  evidenceRead: {
    maxLogicalQueries: 3,
    maxSearchToolCalls: 3,
    maxHitsConsidered: 10,
    maxSelectedSources: 2,
    maxContextsRead: 3,
  },
};

export function makeTopic(): TopicFrame {
  return {
    topicId: "topic-1",
    revision: 1,
    userTopic: "LLM inference runtime optimization",
    objective: "Identify traceable, testable inference optimization directions.",
    workloads: ["decoder-only LLM"],
    phases: ["decode"],
    regimes: ["batch=1"],
    stackScope: ["runtime", "kernel"],
    layerScope: ["L3", "L4"],
    targetMetrics: ["latency"],
    invariants: ["semantic equivalence"],
    exclusions: ["training"],
    seedTerms: ["decode", "latency"],
    synonymGroups: [["decode", "autoregressive generation"]],
    unresolvedScopeQuestions: [],
    scopeAudit: {
      initialFingerprint: "scope-1",
      currentFingerprint: "scope-1",
      changes: [],
    },
  };
}

export function makeAnchor(status: Anchor["status"] = "active"): Anchor {
  return {
    anchorId: "anchor-1",
    topicId: "topic-1",
    revision: 1,
    scenario: {
      workload: "decoder-only LLM",
      phase: "decode",
      regime: "batch=1",
      stack: ["runtime", "GPU kernel"],
    },
    baseline: {
      name: "sequential decode",
      executionPath: ["scheduler", "attention kernel"],
      configuration: ["batch=1"],
      comparisonScope: ["same model", "same tokens"],
    },
    performanceTension: {
      symptom: "high per-token latency",
      suspectedMechanism: "launch and memory overhead",
      bottleneckResources: ["memory bandwidth"],
      targetMetrics: ["latency"],
    },
    constraints: ["semantic equivalence"],
    evidenceRefs: ["ev-1"],
    openNeedIds: [],
    directionIds: ["direction-1"],
    status,
    statusReason: status === "rejected" ? "Out of scope." : "",
    saturationReason:
      status === "saturated" ? "All knowledge-answerable gaps are closed." : null,
  };
}

export function makeDirection(
  status: Direction["status"] = "exploring",
): Direction {
  return {
    directionId: "direction-1",
    anchorId: "anchor-1",
    revision: 1,
    title: "Fuse bounded decode operations",
    hypothesis:
      "Fusing adjacent decode operations reduces launch overhead and latency.",
    changes: [
      {
        atomId: "atom-1",
        layer: "L4",
        object: "decode operation sequence",
        fromState: "separate launches",
        toState: "one fused launch",
        role: "primary",
        conditions: ["batch=1"],
        evidenceRefs: ["ev-1"],
      },
    ],
    causalLinks: [
      {
        from: "fused launch",
        to: "lower launch overhead",
        relation: "causes",
        condition: "operations remain compatible",
        evidenceRefs: ["ev-1"],
        directness: "direct",
      },
    ],
    comparison: {
      baseline: "separate launches",
      controlledVariables: ["model", "tokens"],
      ablations: ["fusion on/off"],
    },
    expectedEffects: [
      {
        metric: "latency",
        expectedDirection: "decrease",
        rationale: "fewer launches",
      },
    ],
    implementation: {
      targetComponents: ["runtime scheduler"],
      knownEntryPoints: ["decode dispatch"],
      unresolvedInterfaces: [],
    },
    falsifiers: ["launch count falls but latency does not"],
    degradationConditions: ["fusion increases register pressure"],
    supportingEvidenceRefs: ["ev-1"],
    contradictingEvidenceRefs: [],
    inferredClaims: [],
    unresolvedNeedIds: [],
    status,
    statusReason:
      status === "testable"
        ? "Causal, implementation, measurement, and falsifier boundaries are complete."
        : status === "rejected"
          ? "A registered rejection reason."
          : "",
    experimentHandoffId: status === "experiment_required" ? "handoff-1" : null,
  };
}

export function makeSearchNeed(): SearchNeed {
  return {
    needId: "need-1",
    revision: 1,
    owner: {
      topicId: "topic-1",
      anchorId: "anchor-1",
      directionId: "direction-1",
    },
    intent: "explain_mechanism",
    question:
      "What mechanism links decode launch fusion to lower batch-1 latency?",
    rationale: "The causal link needs local evidence.",
    successCriteria: ["A source states the mechanism under the decode scenario."],
    primaryDimension: "knowledge",
    auxiliaryDimension: "idea",
    targetDimensions: ["knowledge", "idea"],
    queryVariants: [],
    technicalObjects: ["launch fusion"],
    knownTerms: ["decode"],
    synonymGroups: [["fusion", "kernel fusion"]],
    scenarioTerms: ["batch-1 decode"],
    performanceRelations: ["lower latency"],
    evidenceIntentTerms: ["mechanism"],
    excludedSourceUnits: [],
    previousAttemptIds: [],
    critical: true,
    answerability: "knowledge_base",
    status: "pending",
  };
}

function identity(messageType: string) {
  return {
    protocolVersion: 1 as const,
    messageType,
    workflowId: "workflow-1",
    runId: "run-1",
    taskId: "task-1",
    attemptId: "attempt-1",
    stageId: "stage-1",
    stageContractHash: HASH_A,
  };
}

function skill(name: string) {
  return { name, version: "1.0.0", sha256: HASH_B };
}

function schema(expectedOutputMessageType: string) {
  return {
    manifestSha256: HASH_C,
    expectedOutputMessageType,
    expectedOutputSchemaSha256: HASH_A,
  };
}

function permission(
  role:
    | "workflow_decision"
    | "evidence_reader"
    | "direction_reviewer"
    | "closure_reviewer",
  budget: TurnBudget,
) {
  return {
    role,
    tools:
      role === "evidence_reader"
        ? [
            "mcp__obsidian__obsidian_search_notes",
            "mcp__obsidian__obsidian_get_note",
          ]
        : [],
    allowedPathPrefixes:
      role === "evidence_reader"
        ? ["knowledge_notes/", "idea_notes/"]
        : [],
    filesystem: role === "evidence_reader" ? "vault_read_only" : "none",
    network: false as const,
    delegation: false as const,
    goals: false as const,
    stateWrite: false as const,
    experimentExecution: false as const,
    maxBudget: structuredClone(budget),
  };
}

export function makeEvidenceTask(): EvidenceReaderTaskEnvelope {
  const envelope = {
    ...identity("EVIDENCE_READER_TASK"),
    messageType: "EVIDENCE_READER_TASK" as const,
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      searchNeed: makeSearchNeed(),
      focus: {
        topic: makeTopic(),
        anchor: makeAnchor(),
        direction: makeDirection(),
      },
      previousQueries: [],
      previousReads: [],
      consumedSourceUnitIds: [],
      allowedVaultRoots: [
        {
          dimension: "knowledge" as const,
          relativePathPrefix: "knowledge_notes/",
        },
        { dimension: "idea" as const, relativePathPrefix: "idea_notes/" },
      ],
      budget: structuredClone(EVIDENCE_BUDGET),
      skill: skill("learning-semantic-loop-evidence-reader"),
      schema: schema("EVIDENCE_PACKET") as never,
      permission: permission("evidence_reader", EVIDENCE_BUDGET) as never,
      correctionFeedback: null,
      terminationCondition:
        "Emit exactly one EVIDENCE_PACKET JSON value, then terminate.",
    },
  };
  return envelope;
}

export function makeEvidenceResult(): EvidencePacketEnvelope {
  const exact =
    "For batch-1 decode, fusing adjacent launches removes launch overhead and reduces per-token latency.";
  return {
    ...identity("EVIDENCE_PACKET"),
    messageType: "EVIDENCE_PACKET",
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      packetId: "packet-1",
      needId: "need-1",
      needRevision: 1,
      status: "complete",
      searches: [
        {
          searchId: "search-1",
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
          pageHitCount: 1,
          cumulativeHitCount: 1,
          outcome: "hits",
          stopReason: "success_criteria_met",
        },
      ],
      hitsConsidered: [
        {
          hitId: "hit-1",
          searchId: "search-1",
          sequence: 2,
          path: "knowledge_notes/fusion.md",
          score: 8.2,
          sourceFamily: "fusion-note",
          selected: true,
          selectionReason: "Directly addresses the success criterion.",
        },
      ],
      contextsRead: [
        {
          contextId: "context-map-1",
          sequence: 3,
          hitId: "hit-1",
          path: "knowledge_notes/fusion.md",
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
          sequence: 4,
          hitId: "hit-1",
          path: "knowledge_notes/fusion.md",
          format: "section",
          heading: "Decode mechanism",
          sectionTarget: "Decode mechanism",
          sourceUnitId: "decode-mechanism",
          sourceFamily: "fusion-note",
          exactContext: exact,
          summary: "Fusion removes launch overhead in batch-1 decode.",
        },
      ],
      findings: [
        {
          evidenceId: "ev-1",
          claimKey: "fusion-removes-launch-overhead",
          claim: "Launch fusion removes launch overhead in batch-1 decode.",
          evidenceRole: "mechanism",
          directness: "direct",
          attribution: "source_report",
          applicableConditions: ["batch-1 decode"],
          comparisonBaseline: "separate launches",
          sourcePath: "knowledge_notes/fusion.md",
          sourceUnitId: "decode-mechanism",
          sourceFamily: "fusion-note",
          heading: "Decode mechanism",
          quoteOrExactContext:
            "fusing adjacent launches removes launch overhead",
        },
      ],
      contradictions: [],
      unanswered: [],
      conclusion: "answered",
      conclusionRationale:
        "The deep-read source directly satisfies the sole success criterion.",
    },
  };
}

export function makeDirectionTask(): DirectionReviewTaskEnvelope {
  const direction = makeDirection();
  return {
    ...identity("DIRECTION_REVIEW_TASK"),
    messageType: "DIRECTION_REVIEW_TASK",
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      topic: makeTopic(),
      anchor: makeAnchor(),
      direction,
      siblingDirections: [],
      evidenceFindings: makeEvidenceResult().payload.findings,
      contradictingEvidence: [],
      unresolvedSearchNeeds: [],
      counterexamples: [],
      priorReviews: [],
      reviewPurpose: "initial",
      rubric: {
        rubricId: "direction-readiness-v1",
        version: "1.0.0",
        sha256: HASH_A,
      },
      allowedEvidenceIds: ["ev-1"],
      allowedObjectRefs: [
        { objectType: "direction", objectId: "direction-1", revision: 1 },
      ],
      inputArtifacts: [],
      budget: structuredClone(ZERO_BUDGET),
      skill: skill("learning-semantic-loop-direction-reviewer"),
      schema: schema("REVIEW_DELTA") as never,
      permission: permission("direction_reviewer", ZERO_BUDGET) as never,
      correctionFeedback: null,
      terminationCondition:
        "Emit exactly one REVIEW_DELTA JSON value, then terminate.",
    },
  };
}

export function makeReviewResult() {
  return {
    ...identity("REVIEW_DELTA"),
    messageType: "REVIEW_DELTA" as const,
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      reviewId: "review-1",
      directionId: "direction-1",
      directionRevision: 1,
      supportedParts: ["mechanism", "implementation", "measurement"],
      evidenceRefsUsed: ["ev-1"],
      weakestCausalLink: null,
      baselineProblem: null,
      implementationProblem: null,
      measurementProblem: null,
      strongestCounterexample: null,
      counterexampleResolution: null,
      nextQuestion: null,
      nextQuestionAnswerableFromKnowledgeBase: false,
      decision: "testable" as const,
      rationale:
        "All registered readiness checks are supported by the supplied projection.",
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
        newExperimentRequired: false,
      },
      experimentHandoff: null,
    },
  };
}

export function makeClosureTask(): ClosureReviewTaskEnvelope {
  const topic = makeTopic();
  const anchor = makeAnchor("saturated");
  const direction = makeDirection("testable");
  const topicRef = {
    objectType: "topic",
    objectId: topic.topicId,
    revision: topic.revision,
  };
  const anchorRef = {
    objectType: "anchor",
    objectId: anchor.anchorId,
    revision: anchor.revision,
  };
  const directionRef = {
    objectType: "direction",
    objectId: direction.directionId,
    revision: direction.revision,
  };
  const candidate = {
    stopCandidateId: "stop-1",
    stopProofId: "proof-1",
    runId: "run-1",
    topicId: "topic-1",
    canonicalRevision: 3,
    reason: "All registered closure facts appear satisfied.",
  };
  return {
    ...identity("CLOSURE_REVIEW_TASK"),
    messageType: "CLOSURE_REVIEW_TASK",
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      stopCandidateBundle: {
        candidate,
        proof: {
          proofId: "proof-1",
          stopCandidateId: "stop-1",
          canonicalRevision: 3,
          topicFrameRevision: 1,
          anchorRefs: [anchorRef],
          directionRefs: [directionRef],
          openNeedIds: [],
          pendingTaskIds: [],
          inFlightTaskIds: [],
          pendingOutputRetryTaskIds: [],
          unconsumedResultIds: [],
          uncommittedDeltaIds: [],
          unresolvedValidationFailureIds: [],
          failedTaskIds: [],
          unreviewedCriticalContradictionIds: [],
          experimentHandoffIds: [],
          lastTopicExpansionNeedId: "need-expand-1",
          outputCoverageProjectionId: "coverage-1",
          claims: {
            topicScopePreserved: true,
            noKnowledgeAnswerableCriticalNeed: true,
            allAnchorsClosed: true,
            allDirectionsTerminal: true,
            lastTopicExpansionNoDelta: true,
            noUnconsumedOrUncommittedWork: true,
            criticalContradictionsReviewed: true,
            experimentHandoffsComplete: true,
            runtimeEligibleForCompletion: true,
            finalOutputTraceable: true,
          },
        },
      },
      currentCanonicalRevision: 3,
      topic,
      anchors: [
        {
          anchorRef,
          status: anchor.status,
          statusReason: anchor.statusReason,
          saturationReason: anchor.saturationReason,
        },
      ],
      directions: [
        {
          directionRef,
          anchorId: anchor.anchorId,
          status: direction.status,
          statusReason: direction.statusReason,
          experimentHandoffId: null,
        },
      ],
      needs: [],
      taskIndex: [],
      resultIndex: [],
      deltaIndex: [],
      outputAttemptIndex: [],
      validationFailureIndex: [],
      recentSemanticRecords: [],
      recentNoDeltaRecords: [
        { objectType: "no_delta", objectId: "no-delta-1", revision: 1 },
      ],
      lastTopicExpansion: {
        needId: "need-expand-1",
        needRevision: 1,
        intent: "discover_anchor",
        ownerTopicId: topic.topicId,
        completed: true,
        outcome: "no_new_anchor_no_critical_delta",
        noDeltaRecordId: "no-delta-1",
        semanticDeltaId: null,
      },
      contradictions: [],
      experimentHandoffs: [],
      mechanicalPreflight: {
        preflightId: "preflight-1",
        stopCandidateId: "stop-1",
        canonicalRevision: 3,
        checks: [{ checkId: "all", passed: true, issueIds: [] }],
        passed: true,
      },
      rubric: {
        rubricId: "closure-rubric-v1",
        version: "1.0.0",
        sha256: HASH_A,
      },
      budgetState: {
        turnsUsed: 4,
        maxTurns: 20,
        inputTokensUsed: 1000,
        maxInputTokens: 100_000,
        outputTokensUsed: 500,
        maxOutputTokens: 50_000,
        toolCallsUsed: 3,
        maxToolCalls: 30,
        elapsedMs: 1000,
        maxElapsedMs: 60_000,
        exhaustedDimensions: [],
      },
      lifecycle: "closure_preflight",
      runtimeEligibility: {
        budgetExhausted: false,
        paused: false,
        blocked: false,
        failed: false,
        reason: null,
      },
      outputCoverage: {
        projectionId: "coverage-1",
        fields: {
          topic_scope: [topicRef],
          anchor_summaries: [anchorRef],
          direction_statuses: [directionRef],
          evidence_provenance: [directionRef],
          contradictions_and_limits: [topicRef],
          experiment_handoffs: [directionRef],
          unresolved_questions: [topicRef],
        },
      },
      freshTurn: true,
      providerHistoryIncluded: false,
      canonicalOnly: true,
      budget: structuredClone(ZERO_BUDGET),
      skill: skill("learning-semantic-loop-closure-reviewer"),
      schema: schema("CLOSURE_REVIEW") as never,
      permission: permission("closure_reviewer", ZERO_BUDGET) as never,
      correctionFeedback: null,
      terminationCondition:
        "Emit exactly one CLOSURE_REVIEW JSON value, then terminate.",
    },
  };
}

export function makeClosureResult(): ClosureReviewEnvelope {
  const task = makeClosureTask();
  const ref = {
    objectType: "run",
    objectId: "run-1",
    revision: null,
  };
  return {
    ...identity("CLOSURE_REVIEW"),
    messageType: "CLOSURE_REVIEW",
    stateBinding: structuredClone(STATE),
    inputHash: HASH_B,
    payload: {
      reviewId: "closure-review-1",
      stopCandidateId: "stop-1",
      canonicalRevision: 3,
      status: "complete",
      decision: "accept",
      verifiedClosureBasis: CLOSURE_CHECK_NAMES.map((check) => ({
        check,
        statement: `${check} is supported by the supplied canonical projection.`,
        objectRefs: [ref],
      })),
      closureChecks: Object.fromEntries(
        CLOSURE_CHECK_NAMES.map((check) => [check, true]),
      ) as never,
      blockingFindings: [],
      reopenScopes: [],
      allowsFinalization: true,
      finalizationRequirements: [...FINALIZATION_REQUIREMENTS],
      rationale:
        "All thirteen registered closure checks are true; finalization remains Controller-owned.",
    },
  };
}

export function makeWorkflowTask(): WorkflowTurnTask {
  const task = {
    ...identity("WORKFLOW_TURN_TASK"),
    messageType: "WORKFLOW_TURN_TASK" as const,
    stateSnapshot: structuredClone(STATE),
    decisionInputHash: HASH_B,
    trigger: "INITIALIZE_TOPIC" as const,
    immutableObjective: "Build a traceable semantic exploration.",
    immutableAcceptanceCriteria: ["Produce at least one reviewed Direction."],
    objectiveHash: HASH_A,
    acceptanceCriteriaHash: HASH_C,
    lifecycle: "initialized" as const,
    activeFocusRef: null,
    domainProjection: {
      topic: null,
      focusAnchor: null,
      focusDirection: null,
      searchNeeds: [],
      evidencePackets: [],
      directionReviews: [],
      stopCandidateBundle: null,
      completionProjection: null,
    },
    taskIndex: {
      pendingTaskIds: [],
      inFlightTaskIds: [],
      failedTaskIds: [],
      pendingOutputRetryTaskIds: [],
    },
    resultIndex: {
      committedUnconsumedResultRefs: [],
      consumedResultRefs: [],
    },
    relevantPlan: { revision: 2, stageIds: [], dependencyIds: [] },
    approvedArtifacts: [],
    triggerReport: {
      reportId: "trigger-1",
      trigger: "INITIALIZE_TOPIC" as const,
      sourceStageId: null,
      sourceAttemptId: null,
      facts: ["No TopicFrame exists."],
      issueCodes: [],
    },
    recentEvents: [],
    skill: skill("learning-semantic-loop-workflow-turn"),
    schema: schema("WORKFLOW_DECISION_PROPOSAL") as never,
    permission: {
      allowedActions: ["RUN_STAGE" as const],
      allowedStageTypes: ["SCRIPT_APPLY_TOPIC_FRAME" as const],
      allowedRoles: [],
      allowedTools: [],
      allowedPathPrefixes: [],
      registeredRubrics: [],
      maxBudgetByRole: {
        workflow_decision: structuredClone(ZERO_BUDGET),
      },
      suppliedObjectRefs: [],
      suppliedArtifactIds: [],
      suppliedResultRefs: [],
    },
    correctionFeedback: null,
    terminationCondition:
      "Emit exactly one WORKFLOW_DECISION_PROPOSAL JSON value, then terminate.",
  };
  return task;
}

export function makeWorkflowProposal(): WorkflowDecisionProposal {
  const topic = makeTopic();
  const contract = {
    proposalLocalStageKey: "local-topic-1",
    stageType: "SCRIPT_APPLY_TOPIC_FRAME" as const,
    objective: "Commit the initial TopicFrame.",
    scope: [],
    executionKind: "SCRIPT_TRANSITION" as const,
    role: null,
    requiredInputs: [],
    expectedOutputMessageType: null,
    requestedTools: [],
    requestedPaths: [],
    prohibitedActions: [
      "shell execution",
      "experiment execution",
      "state write outside Controller",
    ],
    budget: structuredClone(ZERO_BUDGET),
  };
  return {
    ...identity("WORKFLOW_DECISION_PROPOSAL"),
    messageType: "WORKFLOW_DECISION_PROPOSAL",
    expectedState: structuredClone(STATE),
    decisionInputHash: HASH_B,
    proposalId: "proposal-1",
    decision: "RUN_STAGE",
    reason: "The initialized run has no TopicFrame.",
    assumptions: [],
    proposedStageContract: contract,
    proposedGateDefinition: {
      proposalLocalStageKey: "local-topic-1",
      mechanicalChecks: [],
      semanticEvaluation: {
        required: false,
        evaluatorRole: null,
        rubricId: null,
        inputProjection: [],
        expectedOutputMessageType: null,
      },
    },
    proposedPlanPatch: null,
    targetStageId: null,
    domainProposal: { kind: "topic_frame", value: topic },
    basisArtifactRefs: [],
    basisResultRefs: [],
    requestedUserInput: null,
    blockedReport: null,
    pauseProposal: null,
    confidence: null,
  };
}

export function rehashPayloadEnvelope<T extends { inputHash: string }>(value: T): T {
  const clone = structuredClone(value);
  clone.inputHash = canonicalSha256({ ...clone, inputHash: "" });
  return clone;
}
