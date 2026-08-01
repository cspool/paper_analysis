import {
  CLOSURE_CHECK_NAMES,
  FINALIZATION_REQUIREMENTS,
  ROLE_MESSAGE_TYPES,
} from "../contracts/index.ts";

export type JsonSchema = Record<string, unknown>;

const id = { type: "string", minLength: 1 };
const text = { type: "string" };
const nonEmptyText = { type: "string", minLength: 1 };
const hash = { type: "string", pattern: "^[a-f0-9]{64}$" };
const positiveInteger = { type: "integer", minimum: 1 };
const nonNegativeInteger = { type: "integer", minimum: 0 };
const nullableText = { type: ["string", "null"] };
const stringArray = { type: "array", items: text };
const nonEmptyStringArray = {
  type: "array",
  items: nonEmptyText,
  minItems: 1,
};

function strictObject(
  properties: Record<string, JsonSchema>,
  required: readonly string[] = Object.keys(properties),
): JsonSchema {
  return {
    type: "object",
    properties,
    required: [...required],
    additionalProperties: false,
  };
}

function array(items: JsonSchema, extra: JsonSchema = {}): JsonSchema {
  return { type: "array", items, ...extra };
}

function nullable(schema: JsonSchema): JsonSchema {
  return { anyOf: [schema, { type: "null" }] };
}

function enumSchema(values: readonly (string | number | boolean)[]): JsonSchema {
  const types = new Set(
    values.map((value) =>
      typeof value === "number" && Number.isInteger(value)
        ? "integer"
        : typeof value,
    ),
  );
  return {
    type: types.size === 1 ? [...types][0] : [...types],
    enum: [...values],
  };
}

function constSchema(value: string | number | boolean): JsonSchema {
  return {
    type:
      typeof value === "number" && Number.isInteger(value)
        ? "integer"
        : typeof value,
    const: value,
  };
}

const layer = enumSchema(["L1", "L2", "L3", "L4", "L5", "L6"]);
const dimension = enumSchema([
  "idea",
  "knowledge",
  "experiment",
  "human",
  "paper",
]);

export const objectRefSchema = strictObject({
  objectType: nonEmptyText,
  objectId: nonEmptyText,
  revision: positiveInteger,
});

export const artifactRefSchema = strictObject({
  artifactId: id,
  kind: nonEmptyText,
  relativePath: nonEmptyText,
  sha256: hash,
  sizeBytes: nonNegativeInteger,
  trustClass: enumSchema([
    "canonical",
    "validated_result",
    "user_input",
    "untrusted_log",
  ]),
});

export const rubricBindingSchema = strictObject({
  rubricId: id,
  version: nonEmptyText,
  sha256: hash,
});

export const stateBindingSchema = strictObject({
  snapshotVersion: nonNegativeInteger,
  canonicalRevision: nonNegativeInteger,
  eventCursor: nonNegativeInteger,
  workflowPlanRevision: positiveInteger,
});

export const turnBudgetSchema = strictObject({
  timeoutMs: positiveInteger,
  maxInputTokens: positiveInteger,
  maxOutputTokens: positiveInteger,
  maxToolCalls: nonNegativeInteger,
  evidenceRead: nullable(
    strictObject({
      maxLogicalQueries: { type: "integer", minimum: 1, maximum: 3 },
      maxSearchToolCalls: positiveInteger,
      maxHitsConsidered: { type: "integer", minimum: 1, maximum: 50 },
      maxSelectedSources: positiveInteger,
      maxContextsRead: positiveInteger,
    }),
  ),
});

export const topicFrameSchema = strictObject({
  topicId: id,
  revision: positiveInteger,
  userTopic: nonEmptyText,
  objective: nonEmptyText,
  workloads: stringArray,
  phases: stringArray,
  regimes: stringArray,
  stackScope: stringArray,
  layerScope: array(layer, { uniqueItems: true }),
  targetMetrics: stringArray,
  invariants: stringArray,
  exclusions: stringArray,
  seedTerms: stringArray,
  synonymGroups: array(array(nonEmptyText, { minItems: 1 })),
  unresolvedScopeQuestions: stringArray,
  scopeAudit: strictObject({
    initialFingerprint: nonEmptyText,
    currentFingerprint: nonEmptyText,
    changes: array(
      strictObject({
        changeId: id,
        field: nonEmptyText,
        changeType: enumSchema(["narrow", "broaden", "clarify"]),
        userAuthorized: { type: "boolean" },
        reason: nonEmptyText,
      }),
    ),
  }),
});

export const anchorSchema = strictObject({
  anchorId: id,
  topicId: id,
  revision: positiveInteger,
  scenario: strictObject({
    workload: nonEmptyText,
    phase: nonEmptyText,
    regime: nonEmptyText,
    stack: nonEmptyStringArray,
  }),
  baseline: strictObject({
    name: nonEmptyText,
    executionPath: nonEmptyStringArray,
    configuration: stringArray,
    comparisonScope: nonEmptyStringArray,
  }),
  performanceTension: strictObject({
    symptom: nonEmptyText,
    suspectedMechanism: nonEmptyText,
    bottleneckResources: nonEmptyStringArray,
    targetMetrics: nonEmptyStringArray,
  }),
  constraints: stringArray,
  evidenceRefs: stringArray,
  openNeedIds: stringArray,
  directionIds: stringArray,
  status: enumSchema(["candidate", "active", "saturated", "rejected"]),
  statusReason: text,
  saturationReason: nullableText,
});

const modificationAtomSchema = strictObject({
  atomId: id,
  layer,
  object: nonEmptyText,
  fromState: nonEmptyText,
  toState: nonEmptyText,
  role: enumSchema(["primary", "enabler", "alternative", "constraint"]),
  conditions: stringArray,
  evidenceRefs: stringArray,
});

export const directionSchema = strictObject({
  directionId: id,
  anchorId: id,
  revision: positiveInteger,
  title: nonEmptyText,
  hypothesis: nonEmptyText,
  changes: array(modificationAtomSchema, { minItems: 1 }),
  causalLinks: array(
    strictObject({
      from: nonEmptyText,
      to: nonEmptyText,
      relation: enumSchema([
        "causes",
        "enables",
        "controls",
        "requires",
        "conflicts",
      ]),
      condition: text,
      evidenceRefs: stringArray,
      directness: enumSchema(["direct", "inferred", "unknown"]),
    }),
    { minItems: 1 },
  ),
  comparison: strictObject({
    baseline: nonEmptyText,
    controlledVariables: stringArray,
    ablations: stringArray,
  }),
  expectedEffects: array(
    strictObject({
      metric: nonEmptyText,
      expectedDirection: enumSchema(["increase", "decrease", "maintain"]),
      rationale: nonEmptyText,
    }),
    { minItems: 1 },
  ),
  implementation: strictObject({
    targetComponents: stringArray,
    knownEntryPoints: stringArray,
    unresolvedInterfaces: stringArray,
  }),
  falsifiers: stringArray,
  degradationConditions: stringArray,
  supportingEvidenceRefs: stringArray,
  contradictingEvidenceRefs: stringArray,
  inferredClaims: stringArray,
  unresolvedNeedIds: stringArray,
  status: enumSchema([
    "seed",
    "exploring",
    "testable",
    "experiment_required",
    "rejected",
  ]),
  statusReason: text,
  experimentHandoffId: nullableText,
});

export const searchNeedSchema = strictObject({
  needId: id,
  revision: positiveInteger,
  owner: strictObject({
    topicId: id,
    anchorId: nullableText,
    directionId: nullableText,
  }),
  intent: enumSchema([
    "discover_anchor",
    "define_baseline",
    "find_modification",
    "explain_mechanism",
    "find_implementation",
    "design_measurement",
    "challenge_direction",
    "verify_primary_source",
  ]),
  question: nonEmptyText,
  rationale: nonEmptyText,
  successCriteria: array(nonEmptyText, { minItems: 1 }),
  primaryDimension: dimension,
  auxiliaryDimension: nullable(dimension),
  targetDimensions: array(dimension, { minItems: 1, maxItems: 2, uniqueItems: true }),
  queryVariants: stringArray,
  technicalObjects: nonEmptyStringArray,
  knownTerms: stringArray,
  synonymGroups: array(array(nonEmptyText, { minItems: 1 })),
  scenarioTerms: nonEmptyStringArray,
  performanceRelations: stringArray,
  evidenceIntentTerms: stringArray,
  excludedSourceUnits: stringArray,
  previousAttemptIds: stringArray,
  critical: { type: "boolean" },
  answerability: enumSchema([
    "knowledge_base",
    "experiment_only",
    "unknown",
    "not_applicable",
  ]),
  status: enumSchema([
    "pending",
    "answered",
    "no_delta",
    "closed",
    "experiment_only",
  ]),
});

const evidenceTermUseSchema = strictObject({
  term: nonEmptyText,
  source: enumSchema(["task", "context"]),
  sourceRef: nonEmptyText,
  introducedAtSequence: nonNegativeInteger,
});

const evidenceSearchSchema = strictObject({
  searchId: id,
  sequence: positiveInteger,
  logicalQueryLevel: enumSchema([1, 2, 3]),
  dimension,
  query: nonEmptyText,
  pathFilter: {
    ...nonEmptyText,
    description:
      "The exact path filter token executed in query, including the path: prefix, for example path:idea_notes/.",
  },
  terms: array(evidenceTermUseSchema, { minItems: 1 }),
  page: positiveInteger,
  cursorUsed: nullableText,
  nextCursor: nullableText,
  toolCallIndex: positiveInteger,
  pageHitCount: nonNegativeInteger,
  cumulativeHitCount: nonNegativeInteger,
  outcome: enumSchema(["hits", "no_hits", "insufficient_context"]),
  stopReason: nullable(
    enumSchema([
      "success_criteria_met",
      "next_level_required",
      "budget_exhausted",
      "topic_boundary",
      "no_more_pages",
    ]),
  ),
});

const evidenceHitSchema = strictObject({
  hitId: id,
  searchId: id,
  sequence: positiveInteger,
  path: nonEmptyText,
  score: nullable({ type: "number" }),
  sourceFamily: nonEmptyText,
  selected: { type: "boolean" },
  selectionReason: text,
});

const evidenceContextSchema = strictObject({
  contextId: id,
  sequence: positiveInteger,
  hitId: id,
  path: nonEmptyText,
  format: {
    ...enumSchema(["document-map", "section", "content", "full"]),
    description:
      "Ledger entry for one actual obsidian_get_note call. Include the document-map call and every subsequent content-bearing read as separate entries.",
  },
  heading: text,
  sectionTarget: nullableText,
  sourceUnitId: nonEmptyText,
  sourceFamily: nonEmptyText,
  exactContext: nonEmptyText,
  summary: nonEmptyText,
});

export const evidenceFindingSchema = strictObject({
  evidenceId: id,
  claimKey: nonEmptyText,
  claim: nonEmptyText,
  evidenceRole: enumSchema([
    "scenario",
    "baseline",
    "mechanism",
    "modification",
    "implementation",
    "measurement",
    "constraint",
    "counterexample",
    "human_prior",
  ]),
  directness: enumSchema(["direct", "inferred"]),
  attribution: enumSchema(["source_report", "workflow_inference"]),
  applicableConditions: stringArray,
  comparisonBaseline: nullableText,
  sourcePath: nonEmptyText,
  sourceUnitId: nonEmptyText,
  sourceFamily: nonEmptyText,
  heading: text,
  quoteOrExactContext: nonEmptyText,
});

const evidenceContradictionSchema = strictObject({
  contradictionId: id,
  findingIds: array(id, { minItems: 1, uniqueItems: true }),
  target: {
    anyOf: [
      strictObject({
        kind: constSchema("claim"),
        claimKey: nonEmptyText,
      }),
      strictObject({
        kind: constSchema("object"),
        objectRef: objectRefSchema,
      }),
    ],
  },
  summary: nonEmptyText,
  critical: { type: "boolean" },
});

const evidenceUnansweredSchema = strictObject({
  unansweredId: id,
  successCriterion: nonEmptyText,
  reason: enumSchema([
    "no_matching_source",
    "insufficient_context",
    "budget_exhausted",
    "topic_boundary",
  ]),
  attemptedSearchIds: array(id, { minItems: 1, uniqueItems: true }),
});

export const evidencePacketPayloadSchema = strictObject({
  packetId: id,
  needId: id,
  needRevision: positiveInteger,
  status: constSchema("complete"),
  searches: array(evidenceSearchSchema),
  hitsConsidered: array(evidenceHitSchema),
  contextsRead: array(evidenceContextSchema),
  findings: array(evidenceFindingSchema),
  contradictions: array(evidenceContradictionSchema),
  unanswered: array(evidenceUnansweredSchema),
  conclusion: enumSchema(["answered", "partial", "not_found"]),
  conclusionRationale: nonEmptyText,
});

const skillBindingSchema = strictObject({
  name: nonEmptyText,
  version: nonEmptyText,
  sha256: hash,
});

const schemaBindingSchema = strictObject({
  manifestSha256: hash,
  expectedOutputMessageType: enumSchema(Object.values(ROLE_MESSAGE_TYPES).map((v) => v.output)),
  expectedOutputSchemaSha256: hash,
});

const correctionErrorSchema = strictObject({
  code: { type: "string", minLength: 1, maxLength: 128 },
  jsonPointer: {
    anyOf: [
      { type: "string", minLength: 1, maxLength: 512 },
      { type: "null" },
    ],
  },
  message: { type: "string", minLength: 1, maxLength: 4096 },
  requiredRule: {
    type: "string",
    minLength: 1,
    maxLength: 4096,
  },
  validExamples: array(
    { type: "string", minLength: 1, maxLength: 4096 },
    { maxItems: 8 },
  ),
});

const correctionFeedbackSchema = strictObject({
  previousAttemptId: id,
  previousOutputSha256: hash,
  validationReportId: id,
  validationReportSha256: hash,
  failureClass: enumSchema([
    "STRUCTURE_INVALID",
    "BINDING_INVALID",
    "SEMANTIC_INVALID",
  ]),
  errors: array(correctionErrorSchema, {
    minItems: 1,
    maxItems: 32,
  }),
  retryInstruction: nonEmptyText,
});

const turnPermissionSchema = strictObject({
  role: enumSchema(Object.keys(ROLE_MESSAGE_TYPES)),
  tools: stringArray,
  allowedPathPrefixes: stringArray,
  filesystem: enumSchema(["none", "vault_read_only"]),
  network: constSchema(false),
  delegation: constSchema(false),
  goals: constSchema(false),
  stateWrite: constSchema(false),
  experimentExecution: constSchema(false),
  maxBudget: turnBudgetSchema,
});

function envelopeSchema(messageType: string, payload: JsonSchema): JsonSchema {
  return strictObject({
    protocolVersion: constSchema(1),
    messageType: constSchema(messageType),
    workflowId: id,
    runId: id,
    taskId: id,
    attemptId: id,
    stageId: id,
    stageContractHash: hash,
    stateBinding: stateBindingSchema,
    inputHash: hash,
    payload,
  });
}

const evidenceTaskPayloadSchema = strictObject({
  searchNeed: searchNeedSchema,
  focus: strictObject({
    topic: topicFrameSchema,
    anchor: nullable(anchorSchema),
    direction: nullable(directionSchema),
  }),
  previousQueries: array(
    strictObject({
      query: nonEmptyText,
      dimension,
      logicalQueryLevel: enumSchema([1, 2, 3]),
      outcome: nonEmptyText,
    }),
  ),
  previousReads: array(
    strictObject({
      path: nonEmptyText,
      sourceUnitId: nonEmptyText,
      contextHash: hash,
    }),
  ),
  consumedSourceUnitIds: stringArray,
  allowedVaultRoots: array(
    strictObject({
      dimension,
      relativePathPrefix: nonEmptyText,
    }),
    { minItems: 1, maxItems: 2 },
  ),
  budget: turnBudgetSchema,
  skill: skillBindingSchema,
  schema: schemaBindingSchema,
  permission: turnPermissionSchema,
  correctionFeedback: nullable(correctionFeedbackSchema),
  terminationCondition: nonEmptyText,
});

const duplicateProjectionSchema = strictObject({
  directionRef: objectRefSchema,
  baseline: nonEmptyText,
  comparisonScope: stringArray,
  controlledVariables: stringArray,
  primaryChanges: array(
    strictObject({
      layer,
      object: nonEmptyText,
      fromState: nonEmptyText,
      toState: nonEmptyText,
      conditions: stringArray,
    }),
    { minItems: 1 },
  ),
  causalTargets: nonEmptyStringArray,
  hypothesis: nonEmptyText,
});

const experimentHandoffSchema = strictObject({
  handoffId: id,
  directionId: id,
  tag: constSchema("EXPERIMENT_REQUIRED"),
  reason: nonEmptyText,
  requiredArtifact: enumSchema([
    "trace",
    "prototype",
    "benchmark",
    "equivalence_test",
    "code_audit",
    "hardware_measurement",
  ]),
  hypothesisToTest: nonEmptyText,
  suggestedEntryPoints: stringArray,
  controlledVariables: stringArray,
  metrics: nonEmptyStringArray,
  acceptanceCriteria: nonEmptyStringArray,
  failureStopConditions: nonEmptyStringArray,
  executionAuthorized: constSchema(false),
});

const readinessChecksSchema = strictObject(
  Object.fromEntries(
    [
      "inTopicAndAnchorScope",
      "baselineFair",
      "minimumChangeSetExplicit",
      "causalChainFalsifiable",
      "implementationPathBounded",
      "measurementPlanComplete",
      "falsifiersPresent",
      "criticalCounterexampleResolved",
      "evidenceTraceable",
      "knowledgeAnswerableCriticalGapRemaining",
      "newExperimentRequired",
    ].map((key) => [key, { type: "boolean" }]),
  ),
);

const duplicateComparisonSchema = strictObject({
  baselineScopeEquivalent: { type: "boolean" },
  primaryChangeEquivalent: { type: "boolean" },
  causalTargetEquivalent: { type: "boolean" },
  materialDifference: nullableText,
});

export const reviewDeltaPayloadSchema = strictObject({
  reviewId: id,
  directionId: id,
  directionRevision: positiveInteger,
  supportedParts: stringArray,
  evidenceRefsUsed: stringArray,
  weakestCausalLink: nullableText,
  baselineProblem: nullableText,
  implementationProblem: nullableText,
  measurementProblem: nullableText,
  strongestCounterexample: nullableText,
  counterexampleResolution: nullableText,
  nextQuestion: nullableText,
  nextQuestionAnswerableFromKnowledgeBase: { type: "boolean" },
  decision: enumSchema([
    "continue_search",
    "testable",
    "experiment_required",
    "rejected",
  ]),
  rationale: nonEmptyText,
  duplicateOfDirectionRef: nullable(objectRefSchema),
  duplicateComparison: nullable(duplicateComparisonSchema),
  rejectionCategory: nullable(
    enumSchema([
      "duplicate",
      "out_of_scope",
      "causal_contradiction",
      "unfair_comparison",
      "no_performance_mechanism",
      "invalid_evidence",
      "other",
    ]),
  ),
  readinessChecks: readinessChecksSchema,
  experimentHandoff: nullable(experimentHandoffSchema),
});

const directionReviewTaskPayloadSchema = strictObject({
  topic: topicFrameSchema,
  anchor: anchorSchema,
  direction: directionSchema,
  siblingDirections: array(duplicateProjectionSchema),
  evidenceFindings: array(evidenceFindingSchema),
  contradictingEvidence: array(evidenceFindingSchema),
  unresolvedSearchNeeds: array(searchNeedSchema),
  counterexamples: array(
    strictObject({
      counterexampleId: id,
      statement: nonEmptyText,
      evidenceRefs: stringArray,
      degradationCondition: nullableText,
    }),
  ),
  priorReviews: array(
    strictObject({
      reviewId: id,
      directionRevision: positiveInteger,
      decision: enumSchema([
        "continue_search",
        "testable",
        "experiment_required",
        "rejected",
      ]),
    }),
  ),
  reviewPurpose: enumSchema([
    "initial",
    "after_evidence",
    "terminal_check",
    "adversarial_recheck",
  ]),
  rubric: rubricBindingSchema,
  allowedEvidenceIds: stringArray,
  allowedObjectRefs: array(objectRefSchema),
  inputArtifacts: array(artifactRefSchema),
  budget: turnBudgetSchema,
  skill: skillBindingSchema,
  schema: schemaBindingSchema,
  permission: turnPermissionSchema,
  correctionFeedback: nullable(correctionFeedbackSchema),
  terminationCondition: nonEmptyText,
});

const stopProofClaimsSchema = strictObject(
  Object.fromEntries(
    [
      "topicScopePreserved",
      "noKnowledgeAnswerableCriticalNeed",
      "allAnchorsClosed",
      "allDirectionsTerminal",
      "lastTopicExpansionNoDelta",
      "noUnconsumedOrUncommittedWork",
      "criticalContradictionsReviewed",
      "experimentHandoffsComplete",
      "runtimeEligibleForCompletion",
      "finalOutputTraceable",
    ].map((key) => [key, { type: "boolean" }]),
  ),
);

const stopCandidateBundleSchema = strictObject({
  candidate: strictObject({
    stopCandidateId: id,
    stopProofId: id,
    runId: id,
    topicId: id,
    canonicalRevision: nonNegativeInteger,
    reason: nonEmptyText,
  }),
  proof: strictObject({
    proofId: id,
    stopCandidateId: id,
    canonicalRevision: nonNegativeInteger,
    topicFrameRevision: positiveInteger,
    anchorRefs: array(objectRefSchema),
    directionRefs: array(objectRefSchema),
    openNeedIds: stringArray,
    pendingTaskIds: stringArray,
    inFlightTaskIds: stringArray,
    pendingOutputRetryTaskIds: stringArray,
    unconsumedResultIds: stringArray,
    uncommittedDeltaIds: stringArray,
    unresolvedValidationFailureIds: stringArray,
    failedTaskIds: stringArray,
    unreviewedCriticalContradictionIds: stringArray,
    experimentHandoffIds: stringArray,
    lastTopicExpansionNeedId: nullableText,
    outputCoverageProjectionId: id,
    claims: stopProofClaimsSchema,
  }),
});

const closureChecksSchema = strictObject(
  Object.fromEntries(
    CLOSURE_CHECK_NAMES.map((key) => [key, { type: "boolean" }]),
  ),
);

const closureScopeRefSchema = strictObject({
  objectType: nonEmptyText,
  objectId: nonEmptyText,
  revision: { type: ["integer", "null"], minimum: 1 },
});

const closureFindingType = enumSchema([
  "knowledge_gap",
  "state_inconsistency",
  "incomplete_handoff",
  "runtime_pause",
]);
const closureFindingCode = enumSchema([
  "stale_stop_proof_revision",
  "stop_proof_canonical_mismatch",
  "mechanical_preflight_failed",
  "topic_scope_silently_narrowed",
  "knowledge_answerable_open_need",
  "anchor_not_closed",
  "anchor_missing_saturation_reason",
  "anchor_missing_status_reason",
  "direction_nonterminal",
  "direction_missing_terminal_reason",
  "last_topic_expansion_missing",
  "last_topic_expansion_not_quiet",
  "pending_task",
  "in_flight_task",
  "pending_output_retry",
  "unconsumed_result",
  "uncommitted_delta",
  "unresolved_validation_failure",
  "failed_task",
  "unreviewed_critical_contradiction",
  "experiment_handoff_missing",
  "experiment_handoff_invalid",
  "runtime_budget_exhausted",
  "runtime_failed_or_paused",
  "final_output_missing_field",
  "final_output_untraceable",
]);

export const closureReviewPayloadSchema = strictObject({
  reviewId: id,
  stopCandidateId: id,
  canonicalRevision: nonNegativeInteger,
  status: constSchema("complete"),
  decision: enumSchema(["accept", "reject"]),
  verifiedClosureBasis: array(
    strictObject({
      check: enumSchema(CLOSURE_CHECK_NAMES),
      statement: nonEmptyText,
      objectRefs: array(closureScopeRefSchema),
    }),
  ),
  closureChecks: closureChecksSchema,
  blockingFindings: array(
    strictObject({
      findingId: id,
      check: enumSchema(CLOSURE_CHECK_NAMES),
      type: closureFindingType,
      code: closureFindingCode,
      summary: nonEmptyText,
      objectRefs: array(closureScopeRefSchema),
      reopenScope: closureScopeRefSchema,
      recoveryAction: enumSchema([
        "REOPEN_FRONTIER",
        "REPAIR_STATE",
        "COMPLETE_HANDOFF",
        "RESUME_RUNTIME",
      ]),
    }),
  ),
  reopenScopes: array(closureScopeRefSchema),
  allowsFinalization: { type: "boolean" },
  finalizationRequirements: array(enumSchema(FINALIZATION_REQUIREMENTS), {
    uniqueItems: true,
  }),
  rationale: nonEmptyText,
});

const budgetStateSchema = strictObject({
  turnsUsed: nonNegativeInteger,
  maxTurns: positiveInteger,
  inputTokensUsed: nonNegativeInteger,
  maxInputTokens: positiveInteger,
  outputTokensUsed: nonNegativeInteger,
  maxOutputTokens: positiveInteger,
  toolCallsUsed: nonNegativeInteger,
  maxToolCalls: nonNegativeInteger,
  elapsedMs: nonNegativeInteger,
  maxElapsedMs: positiveInteger,
  exhaustedDimensions: array(
    enumSchema([
      "turns",
      "input_tokens",
      "output_tokens",
      "tool_calls",
      "elapsed",
    ]),
    { uniqueItems: true },
  ),
});

const outputCoverageFields = [
  "topic_scope",
  "anchor_summaries",
  "direction_statuses",
  "evidence_provenance",
  "contradictions_and_limits",
  "experiment_handoffs",
  "unresolved_questions",
] as const;

const workIndexItemSchema = strictObject({
  id,
  status: nonEmptyText,
  resolvedById: nullableText,
  objectRef: nullable(objectRefSchema),
});

const closureReviewTaskPayloadSchema = strictObject({
  stopCandidateBundle: stopCandidateBundleSchema,
  currentCanonicalRevision: nonNegativeInteger,
  topic: topicFrameSchema,
  anchors: array(
    strictObject({
      anchorRef: objectRefSchema,
      status: enumSchema(["candidate", "active", "saturated", "rejected"]),
      statusReason: text,
      saturationReason: nullableText,
    }),
  ),
  directions: array(
    strictObject({
      directionRef: objectRefSchema,
      anchorId: id,
      status: enumSchema([
        "seed",
        "exploring",
        "testable",
        "experiment_required",
        "rejected",
      ]),
      statusReason: text,
      experimentHandoffId: nullableText,
    }),
  ),
  needs: array(
    strictObject({
      needId: id,
      revision: positiveInteger,
      owner: strictObject({
        topicId: id,
        anchorId: nullableText,
        directionId: nullableText,
      }),
      critical: { type: "boolean" },
      answerability: enumSchema([
        "knowledge_base",
        "experiment_only",
        "unknown",
        "not_applicable",
      ]),
      status: enumSchema([
        "pending",
        "answered",
        "no_delta",
        "closed",
        "experiment_only",
      ]),
    }),
  ),
  taskIndex: array(workIndexItemSchema),
  resultIndex: array(workIndexItemSchema),
  deltaIndex: array(workIndexItemSchema),
  outputAttemptIndex: array(workIndexItemSchema),
  validationFailureIndex: array(workIndexItemSchema),
  recentSemanticRecords: array(objectRefSchema),
  recentNoDeltaRecords: array(objectRefSchema),
  lastTopicExpansion: nullable(
    strictObject({
      needId: id,
      needRevision: positiveInteger,
      intent: constSchema("discover_anchor"),
      ownerTopicId: id,
      completed: { type: "boolean" },
      outcome: enumSchema([
        "no_new_anchor_no_critical_delta",
        "semantic_delta",
      ]),
      noDeltaRecordId: nullableText,
      semanticDeltaId: nullableText,
    }),
  ),
  contradictions: array(
    strictObject({
      contradictionId: id,
      critical: { type: "boolean" },
      dispositionReviewId: nullableText,
      objectRef: objectRefSchema,
    }),
  ),
  experimentHandoffs: array(
    strictObject({
      handoffId: id,
      directionId: id,
      complete: { type: "boolean" },
      executionAuthorized: constSchema(false),
    }),
  ),
  mechanicalPreflight: strictObject({
    preflightId: id,
    stopCandidateId: id,
    canonicalRevision: nonNegativeInteger,
    checks: array(
      strictObject({
        checkId: id,
        passed: { type: "boolean" },
        issueIds: stringArray,
      }),
    ),
    passed: { type: "boolean" },
  }),
  rubric: rubricBindingSchema,
  budgetState: budgetStateSchema,
  lifecycle: enumSchema([
    "initialized",
    "running",
    "waiting_turn",
    "waiting_user",
    "waiting_external",
    "closure_preflight",
    "waiting_closure_review",
    "finalizing",
    "paused_budget",
    "paused_operator",
    "failed_retriable",
    "failed_terminal",
    "blocked_semantic",
    "blocked_external",
    "completed",
    "cancelled",
  ]),
  runtimeEligibility: strictObject({
    budgetExhausted: { type: "boolean" },
    paused: { type: "boolean" },
    blocked: { type: "boolean" },
    failed: { type: "boolean" },
    reason: nullableText,
  }),
  outputCoverage: strictObject({
    projectionId: id,
    fields: strictObject(
      Object.fromEntries(
        outputCoverageFields.map((field) => [field, array(objectRefSchema)]),
      ),
    ),
  }),
  freshTurn: constSchema(true),
  providerHistoryIncluded: constSchema(false),
  canonicalOnly: constSchema(true),
  budget: turnBudgetSchema,
  skill: skillBindingSchema,
  schema: schemaBindingSchema,
  permission: turnPermissionSchema,
  correctionFeedback: nullable(correctionFeedbackSchema),
  terminationCondition: nonEmptyText,
});

const registeredStageTypeSchema = enumSchema([
  "SCRIPT_APPLY_TOPIC_FRAME",
  "SCRIPT_APPLY_SEMANTIC_DELTA",
  "WORKFLOW_DECISION",
  "EVIDENCE_READ",
  "DIRECTION_REVIEW",
  "CLOSURE_REVIEW",
  "RENDER_FINAL",
]);
const registeredRoleSchema = enumSchema(Object.keys(ROLE_MESSAGE_TYPES));
const outputMessageTypeSchema = enumSchema(
  Object.values(ROLE_MESSAGE_TYPES).map((entry) => entry.output),
);

const gateValueTypeSchema = enumSchema([
  "string",
  "number",
  "boolean",
  "null",
  "string_array",
  "object",
]);

const gateActualSchema = {
  anyOf: [
    strictObject({
      source: constSchema("result"),
      pointer: nonEmptyText,
      valueType: gateValueTypeSchema,
    }),
    strictObject({
      source: constSchema("task"),
      pointer: nonEmptyText,
      valueType: gateValueTypeSchema,
    }),
    strictObject({
      source: constSchema("canonical"),
      objectRef: objectRefSchema,
      pointer: nonEmptyText,
      valueType: gateValueTypeSchema,
    }),
    strictObject({
      source: constSchema("runtime"),
      fact: enumSchema([
        "allowed_tool_events_only",
        "allowed_paths_only",
        "turn_budget_within_contract",
        "experiment_execution_count",
        "external_evidence_used",
      ]),
      valueType: gateValueTypeSchema,
    }),
    strictObject({
      source: constSchema("validator"),
      fact: enumSchema([
        "schema_valid",
        "message_binding_matches",
        "registered_validator_passes",
        "references_resolve",
        "source_context_present",
        "duplicate_commit",
        "script_transition_valid",
      ]),
      pointer: nullableText,
      valueType: gateValueTypeSchema,
    }),
    strictObject({
      source: constSchema("artifact"),
      artifactId: id,
      fact: enumSchema(["exists", "sha256"]),
      valueType: gateValueTypeSchema,
    }),
  ],
};

const mechanicalCheckSchema = strictObject({
  checkId: id,
  predicate: enumSchema(["equals", "contains_fields"]),
  actual: gateActualSchema,
  expected: {
    anyOf: [
      { type: "string" },
      { type: "number" },
      { type: "boolean" },
      { type: "null" },
      { type: "array", items: { type: "string" } },
    ],
  },
});

export const stageContractDraftSchema = strictObject({
  proposalLocalStageKey: id,
  stageType: registeredStageTypeSchema,
  objective: nonEmptyText,
  scope: array(objectRefSchema),
  executionKind: enumSchema([
    "SCRIPT_TRANSITION",
    "DECISION_TURN",
    "WORKER_TURN",
    "EVALUATOR_TURN",
  ]),
  role: nullable(registeredRoleSchema),
  requiredInputs: array(artifactRefSchema),
  expectedOutputMessageType: nullable(outputMessageTypeSchema),
  requestedTools: stringArray,
  requestedPaths: stringArray,
  prohibitedActions: stringArray,
  budget: turnBudgetSchema,
});

export const gateDefinitionDraftSchema = strictObject({
  proposalLocalStageKey: id,
  mechanicalChecks: array(mechanicalCheckSchema),
  semanticEvaluation: strictObject({
    required: { type: "boolean" },
    evaluatorRole: nullable(registeredRoleSchema),
    rubricId: nullableText,
    inputProjection: stringArray,
    expectedOutputMessageType: nullable(outputMessageTypeSchema),
  }),
});

const stageNodeDraftSchema = strictObject({
  proposalLocalStageKey: id,
  stageType: registeredStageTypeSchema,
  executionKind: enumSchema([
    "SCRIPT_TRANSITION",
    "DECISION_TURN",
    "WORKER_TURN",
    "EVALUATOR_TURN",
  ]),
  role: nullable(registeredRoleSchema),
  objective: nonEmptyText,
  dependsOnStageIds: stringArray,
  contract: stageContractDraftSchema,
  gate: gateDefinitionDraftSchema,
});

const stageDependencySchema = strictObject({
  dependencyId: id,
  predecessorStageId: id,
  successorStageId: id,
  kind: enumSchema(["requires_committed", "requires_consumed"]),
});

const workflowPlanPatchSchema = strictObject({
  expectedPlanRevision: positiveInteger,
  operations: array(
    {
      anyOf: [
        strictObject({
          op: constSchema("add_stage"),
          stage: stageNodeDraftSchema,
        }),
        strictObject({
          op: constSchema("supersede_stage"),
          stageId: id,
          reason: nonEmptyText,
        }),
        strictObject({
          op: constSchema("add_dependency"),
          dependency: stageDependencySchema,
        }),
        strictObject({
          op: constSchema("remove_dependency"),
          dependencyId: id,
          reason: nonEmptyText,
        }),
      ],
    },
    { minItems: 1 },
  ),
  objectiveHash: hash,
  acceptanceCriteriaHash: hash,
  rationale: nonEmptyText,
});

const semanticDeltaSchema = strictObject({
  deltaId: id,
  basisResultRefs: array(objectRefSchema),
  expectedTargetRevision: nonNegativeInteger,
  target: {
    anyOf: [
      strictObject({ type: constSchema("topic"), id }),
      strictObject({ type: constSchema("anchor"), id }),
      strictObject({ type: constSchema("direction"), id }),
    ],
  },
  action: enumSchema([
    "create",
    "revise",
    "add_evidence",
    "add_contradiction",
    "no_semantic_delta",
    "reject",
  ]),
  changedFields: stringArray,
  rationale: nonEmptyText,
  proposedObject: nullable({
    anyOf: [topicFrameSchema, anchorSchema, directionSchema],
  }),
});

const domainProposalSchema = {
  anyOf: [
    strictObject({
      kind: constSchema("topic_frame"),
      value: topicFrameSchema,
    }),
    strictObject({
      kind: constSchema("search_need"),
      value: searchNeedSchema,
    }),
    strictObject({
      kind: constSchema("semantic_delta"),
      value: semanticDeltaSchema,
    }),
    strictObject({
      kind: constSchema("direction_review_request"),
      value: strictObject({
        directionRef: objectRefSchema,
        purpose: enumSchema([
          "initial",
          "after_evidence",
          "terminal_check",
          "adversarial_recheck",
        ]),
        rubric: rubricBindingSchema,
      }),
    }),
    strictObject({
      kind: constSchema("stop_candidate"),
      value: stopCandidateBundleSchema,
    }),
  ],
};

const userQuestionSchema = strictObject({
  questionId: id,
  prompt: nonEmptyText,
  rationale: nonEmptyText,
  choices: array(
    strictObject({
      choiceId: id,
      label: nonEmptyText,
      consequence: nonEmptyText,
    }),
  ),
  requiredAuthority: nullableText,
  relatedRefs: array(objectRefSchema),
});

const blockedReportSchema = strictObject({
  blockedReportId: id,
  kind: enumSchema(["semantic", "external"]),
  summary: nonEmptyText,
  blockingFacts: nonEmptyStringArray,
  exhaustedAlternatives: stringArray,
  relatedRefs: array(objectRefSchema),
  userActionNeeded: nullableText,
});

const pauseProposalSchema = strictObject({
  pauseProposalId: id,
  reason: nonEmptyText,
  category: enumSchema(["budget", "operator", "external_dependency"]),
  resumableWhen: nonEmptyText,
  relatedRefs: array(objectRefSchema),
});

const workflowPermissionSchema = strictObject({
  allowedActions: array(
    enumSchema([
      "RUN_STAGE",
      "RETRY_STAGE",
      "REPLAN",
      "REQUEST_EVALUATION",
      "ASK_USER",
      "REPORT_BLOCKED",
      "PROPOSE_PAUSE",
      "PROPOSE_COMPLETE",
    ]),
    { minItems: 1, uniqueItems: true },
  ),
  allowedStageTypes: array(registeredStageTypeSchema, { uniqueItems: true }),
  allowedRoles: array(registeredRoleSchema, { uniqueItems: true }),
  allowedTools: stringArray,
  allowedPathPrefixes: stringArray,
  registeredRubrics: stringArray,
  maxBudgetByRole: {
    type: "object",
    properties: Object.fromEntries(
      Object.keys(ROLE_MESSAGE_TYPES).map((role) => [role, turnBudgetSchema]),
    ),
    additionalProperties: false,
  },
  suppliedObjectRefs: array(objectRefSchema),
  suppliedArtifactIds: stringArray,
  suppliedResultRefs: array(objectRefSchema),
});

const workflowTriggerSchema = enumSchema([
  "INITIALIZE_TOPIC",
  "COMMITTED_RESULT_REQUIRES_INTEGRATION",
  "FRONTIER_SELECTION_REQUIRED",
  "MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE",
  "GATE_FAILED_WITHOUT_RECOVERY_RULE",
  "PLAN_EXHAUSTED_OBJECTIVE_OPEN",
  "EVIDENCE_CONTRADICTION",
  "NO_PROGRESS_THRESHOLD_REACHED",
  "CLOSURE_REJECTED",
  "NO_RUNNABLE_STAGE",
  "USER_DECISION_REQUIRED",
]);

const workflowCompletionProjectionSchema = strictObject({
  canonicalRevision: nonNegativeInteger,
  topicFrameRevision: positiveInteger,
  anchorRefs: array(objectRefSchema),
  directionRefs: array(objectRefSchema),
  openNeedIds: stringArray,
  pendingTaskIds: stringArray,
  inFlightTaskIds: stringArray,
  pendingOutputRetryTaskIds: stringArray,
  unconsumedResultIds: stringArray,
  uncommittedDeltaIds: stringArray,
  unresolvedValidationFailureIds: stringArray,
  failedTaskIds: stringArray,
  unreviewedCriticalContradictionIds: stringArray,
  experimentHandoffIds: stringArray,
  lastTopicExpansionNeedId: nullableText,
  outputCoverageProjectionId: id,
  claims: stopProofClaimsSchema,
  eligibleForProposal: { type: "boolean" },
  blockingClaims: array(enumSchema([
    "topicScopePreserved",
    "noKnowledgeAnswerableCriticalNeed",
    "allAnchorsClosed",
    "allDirectionsTerminal",
    "lastTopicExpansionNoDelta",
    "noUnconsumedOrUncommittedWork",
    "criticalContradictionsReviewed",
    "experimentHandoffsComplete",
    "runtimeEligibleForCompletion",
    "finalOutputTraceable",
  ])),
});

const workflowTurnTaskSchema = strictObject({
  protocolVersion: constSchema(1),
  messageType: constSchema("WORKFLOW_TURN_TASK"),
  workflowId: id,
  runId: id,
  taskId: id,
  attemptId: id,
  stageId: id,
  stageContractHash: hash,
  stateSnapshot: stateBindingSchema,
  decisionInputHash: hash,
  trigger: workflowTriggerSchema,
  immutableObjective: nonEmptyText,
  immutableAcceptanceCriteria: nonEmptyStringArray,
  objectiveHash: hash,
  acceptanceCriteriaHash: hash,
  lifecycle: enumSchema([
    "initialized",
    "running",
    "waiting_turn",
    "waiting_user",
    "waiting_external",
    "closure_preflight",
    "waiting_closure_review",
    "finalizing",
    "paused_budget",
    "paused_operator",
    "failed_retriable",
    "failed_terminal",
    "blocked_semantic",
    "blocked_external",
    "completed",
    "cancelled",
  ]),
  activeFocusRef: nullable(objectRefSchema),
  domainProjection: strictObject({
    topic: nullable(topicFrameSchema),
    focusAnchor: nullable(anchorSchema),
    focusDirection: nullable(directionSchema),
    searchNeeds: array(searchNeedSchema),
    evidencePackets: array(evidencePacketPayloadSchema),
    directionReviews: array(reviewDeltaPayloadSchema),
    stopCandidateBundle: nullable(stopCandidateBundleSchema),
    completionProjection: nullable(workflowCompletionProjectionSchema),
  }),
  taskIndex: strictObject({
    pendingTaskIds: stringArray,
    inFlightTaskIds: stringArray,
    failedTaskIds: stringArray,
    pendingOutputRetryTaskIds: stringArray,
  }),
  resultIndex: strictObject({
    committedUnconsumedResultRefs: array(objectRefSchema),
    consumedResultRefs: array(objectRefSchema),
  }),
  relevantPlan: strictObject({
    revision: positiveInteger,
    stageIds: stringArray,
    dependencyIds: stringArray,
  }),
  approvedArtifacts: array(artifactRefSchema),
  triggerReport: strictObject({
    reportId: id,
    trigger: workflowTriggerSchema,
    sourceStageId: nullableText,
    sourceAttemptId: nullableText,
    facts: stringArray,
    issueCodes: stringArray,
  }),
  recentEvents: array(
    strictObject({
      eventCursor: nonNegativeInteger,
      eventType: nonEmptyText,
      summary: nonEmptyText,
      objectRefs: array(objectRefSchema),
    }),
  ),
  skill: skillBindingSchema,
  schema: schemaBindingSchema,
  permission: workflowPermissionSchema,
  correctionFeedback: nullable(correctionFeedbackSchema),
  terminationCondition: nonEmptyText,
});

const workflowDecisionProposalSchema = strictObject({
    protocolVersion: constSchema(1),
    messageType: constSchema("WORKFLOW_DECISION_PROPOSAL"),
    workflowId: id,
    runId: id,
    taskId: id,
    attemptId: id,
    stageId: id,
    stageContractHash: hash,
    expectedState: stateBindingSchema,
    decisionInputHash: hash,
    proposalId: id,
    decision: enumSchema([
      "RUN_STAGE",
      "RETRY_STAGE",
      "REPLAN",
      "REQUEST_EVALUATION",
      "ASK_USER",
      "REPORT_BLOCKED",
      "PROPOSE_PAUSE",
      "PROPOSE_COMPLETE",
    ]),
    reason: nonEmptyText,
    assumptions: stringArray,
    proposedStageContract: nullable(stageContractDraftSchema),
    proposedGateDefinition: nullable(gateDefinitionDraftSchema),
    proposedPlanPatch: nullable(workflowPlanPatchSchema),
    targetStageId: nullableText,
    domainProposal: nullable(domainProposalSchema),
    basisArtifactRefs: array(artifactRefSchema),
    basisResultRefs: array(objectRefSchema),
    requestedUserInput: nullable(userQuestionSchema),
    blockedReport: nullable(blockedReportSchema),
    pauseProposal: nullable(pauseProposalSchema),
    confidence: nullable({ type: "number", minimum: 0, maximum: 1 }),
});

export const SCHEMA_DEFINITIONS = Object.freeze({
  TOPIC_FRAME: topicFrameSchema,
  ANCHOR: anchorSchema,
  DIRECTION: directionSchema,
  SEARCH_NEED: searchNeedSchema,
  EVIDENCE_READER_TASK: envelopeSchema(
    "EVIDENCE_READER_TASK",
    evidenceTaskPayloadSchema,
  ),
  EVIDENCE_PACKET: envelopeSchema(
    "EVIDENCE_PACKET",
    evidencePacketPayloadSchema,
  ),
  DIRECTION_REVIEW_TASK: envelopeSchema(
    "DIRECTION_REVIEW_TASK",
    directionReviewTaskPayloadSchema,
  ),
  REVIEW_DELTA: envelopeSchema("REVIEW_DELTA", reviewDeltaPayloadSchema),
  CLOSURE_REVIEW_TASK: envelopeSchema(
    "CLOSURE_REVIEW_TASK",
    closureReviewTaskPayloadSchema,
  ),
  CLOSURE_REVIEW: envelopeSchema(
    "CLOSURE_REVIEW",
    closureReviewPayloadSchema,
  ),
  WORKFLOW_TURN_TASK: workflowTurnTaskSchema,
  WORKFLOW_DECISION_PROPOSAL: workflowDecisionProposalSchema,
} as const);

export type SchemaName = keyof typeof SCHEMA_DEFINITIONS;

export const SCHEMA_FILENAMES: Readonly<Record<SchemaName, string>> =
  Object.freeze({
    TOPIC_FRAME: "topic_frame.schema.json",
    ANCHOR: "anchor.schema.json",
    DIRECTION: "direction.schema.json",
    SEARCH_NEED: "search_need.schema.json",
    EVIDENCE_READER_TASK: "evidence_task.schema.json",
    EVIDENCE_PACKET: "evidence_packet.schema.json",
    DIRECTION_REVIEW_TASK: "direction_review_task.schema.json",
    REVIEW_DELTA: "review_delta.schema.json",
    CLOSURE_REVIEW_TASK: "closure_review_task.schema.json",
    CLOSURE_REVIEW: "closure_review.schema.json",
    WORKFLOW_TURN_TASK: "workflow_turn_task.schema.json",
    WORKFLOW_DECISION_PROPOSAL: "workflow_decision_proposal.schema.json",
  });

export function publishableSchema(name: SchemaName): JsonSchema {
  return providerCompatibleNode({
    $schema: "https://json-schema.org/draft/2020-12/schema",
    $id: `urn:simple-semantic-loop:${name.toLowerCase()}:v1`,
    title: name,
    ...SCHEMA_DEFINITIONS[name],
  }) as JsonSchema;
}

function providerCompatibleNode(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(providerCompatibleNode);
  }
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([key]) => key !== "uniqueItems")
      .map(([key, child]) => [key, providerCompatibleNode(child)]),
  );
}
