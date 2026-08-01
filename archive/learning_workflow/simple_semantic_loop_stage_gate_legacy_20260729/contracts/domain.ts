export type Layer = "L1" | "L2" | "L3" | "L4" | "L5" | "L6";

/**
 * The six layers are coordinates for modifiable performance objects. They are
 * not a required pipeline and a Direction need only use the Topic-authorized
 * subset that its causal path actually changes.
 */
export const LAYER_DEFINITIONS = Object.freeze({
  L1: {
    name: "Algorithm/Pipeline",
    modifiableObjects:
      "computation graph, workload decomposition, dynamic parameters, approximation, and parallelism",
  },
  L2: {
    name: "Serving/Runtime",
    modifiableObjects:
      "requests, batches, stages, queues, placement, caching, and resource scheduling",
  },
  L3: {
    name: "Compiler",
    modifiableObjects:
      "IR, dependency representation, passes, fusion, multiversioning, and code generation",
  },
  L4: {
    name: "Kernel",
    modifiableObjects:
      "tiles, warps, instruction pipelines, synchronization, data movement, and kernel composition",
  },
  L5: {
    name: "Architecture",
    modifiableObjects:
      "execution/control units, memory hierarchy, schedulers, NoC, and hardware primitives",
  },
  L6: {
    name: "Chip/System",
    modifiableObjects:
      "chiplets, PIM, wafer-scale systems, packaging, die-to-die links, and chip-level resource boundaries",
  },
} as const satisfies Readonly<
  Record<Layer, { name: string; modifiableObjects: string }>
>);

export interface ObjectRef {
  objectType: string;
  objectId: string;
  revision: number;
}

export interface ArtifactRef {
  artifactId: string;
  kind: string;
  relativePath: string;
  sha256: string;
  sizeBytes: number;
  trustClass:
    | "canonical"
    | "validated_result"
    | "user_input"
    | "untrusted_log";
}

export interface RubricBinding {
  rubricId: string;
  version: string;
  sha256: string;
}

export interface TopicFrame {
  topicId: string;
  revision: number;
  userTopic: string;
  objective: string;
  workloads: string[];
  phases: string[];
  regimes: string[];
  stackScope: string[];
  layerScope: Layer[];
  targetMetrics: string[];
  invariants: string[];
  exclusions: string[];
  seedTerms: string[];
  synonymGroups: string[][];
  unresolvedScopeQuestions: string[];
  scopeAudit: {
    initialFingerprint: string;
    currentFingerprint: string;
    changes: Array<{
      changeId: string;
      field: string;
      changeType: "narrow" | "broaden" | "clarify";
      userAuthorized: boolean;
      reason: string;
    }>;
  };
}

export interface Anchor {
  anchorId: string;
  topicId: string;
  revision: number;
  scenario: {
    workload: string;
    phase: string;
    regime: string;
    stack: string[];
  };
  baseline: {
    name: string;
    executionPath: string[];
    configuration: string[];
    comparisonScope: string[];
  };
  performanceTension: {
    symptom: string;
    suspectedMechanism: string;
    bottleneckResources: string[];
    targetMetrics: string[];
  };
  constraints: string[];
  evidenceRefs: string[];
  openNeedIds: string[];
  directionIds: string[];
  status: "candidate" | "active" | "saturated" | "rejected";
  statusReason: string;
  saturationReason: string | null;
}

export interface ModificationAtom {
  atomId: string;
  layer: Layer;
  object: string;
  fromState: string;
  toState: string;
  role: "primary" | "enabler" | "alternative" | "constraint";
  conditions: string[];
  evidenceRefs: string[];
}

export interface Direction {
  directionId: string;
  anchorId: string;
  revision: number;
  title: string;
  hypothesis: string;
  changes: ModificationAtom[];
  causalLinks: Array<{
    from: string;
    to: string;
    relation: "causes" | "enables" | "controls" | "requires" | "conflicts";
    condition: string;
    evidenceRefs: string[];
    directness: "direct" | "inferred" | "unknown";
  }>;
  comparison: {
    baseline: string;
    controlledVariables: string[];
    ablations: string[];
  };
  expectedEffects: Array<{
    metric: string;
    expectedDirection: "increase" | "decrease" | "maintain";
    rationale: string;
  }>;
  implementation: {
    targetComponents: string[];
    knownEntryPoints: string[];
    unresolvedInterfaces: string[];
  };
  falsifiers: string[];
  degradationConditions: string[];
  supportingEvidenceRefs: string[];
  contradictingEvidenceRefs: string[];
  inferredClaims: string[];
  unresolvedNeedIds: string[];
  status:
    | "seed"
    | "exploring"
    | "testable"
    | "experiment_required"
    | "rejected";
  statusReason: string;
  experimentHandoffId: string | null;
}

export type SearchIntent =
  | "discover_anchor"
  | "define_baseline"
  | "find_modification"
  | "explain_mechanism"
  | "find_implementation"
  | "design_measurement"
  | "challenge_direction"
  | "verify_primary_source";

export type KnowledgeDimension =
  | "idea"
  | "knowledge"
  | "experiment"
  | "human"
  | "paper";

export interface SearchNeed {
  needId: string;
  revision: number;
  owner: {
    topicId: string;
    anchorId: string | null;
    directionId: string | null;
  };
  intent: SearchIntent;
  question: string;
  rationale: string;
  successCriteria: string[];
  primaryDimension: KnowledgeDimension;
  auxiliaryDimension: KnowledgeDimension | null;
  targetDimensions: KnowledgeDimension[];
  queryVariants: string[];
  technicalObjects: string[];
  knownTerms: string[];
  synonymGroups: string[][];
  scenarioTerms: string[];
  performanceRelations: string[];
  evidenceIntentTerms: string[];
  excludedSourceUnits: string[];
  previousAttemptIds: string[];
  critical: boolean;
  answerability:
    | "knowledge_base"
    | "experiment_only"
    | "unknown"
    | "not_applicable";
  status:
    | "pending"
    | "answered"
    | "no_delta"
    | "closed"
    | "experiment_only";
}

export interface EvidenceTermUse {
  term: string;
  source: "task" | "context";
  sourceRef: string;
  introducedAtSequence: number;
}

export interface EvidenceSearch {
  searchId: string;
  sequence: number;
  logicalQueryLevel: 1 | 2 | 3;
  dimension: KnowledgeDimension;
  query: string;
  pathFilter: string;
  terms: EvidenceTermUse[];
  page: number;
  cursorUsed: string | null;
  nextCursor: string | null;
  toolCallIndex: number;
  pageHitCount: number;
  cumulativeHitCount: number;
  outcome: "hits" | "no_hits" | "insufficient_context";
  stopReason:
    | "success_criteria_met"
    | "next_level_required"
    | "budget_exhausted"
    | "topic_boundary"
    | "no_more_pages"
    | null;
}

export interface EvidenceHit {
  hitId: string;
  searchId: string;
  sequence: number;
  path: string;
  score: number | null;
  sourceFamily: string;
  selected: boolean;
  selectionReason: string;
}

export interface EvidenceContext {
  contextId: string;
  sequence: number;
  hitId: string;
  path: string;
  format: "document-map" | "section" | "content" | "full";
  heading: string;
  sectionTarget: string | null;
  sourceUnitId: string;
  sourceFamily: string;
  exactContext: string;
  summary: string;
}

export interface EvidenceFinding {
  evidenceId: string;
  claimKey: string;
  claim: string;
  evidenceRole:
    | "scenario"
    | "baseline"
    | "mechanism"
    | "modification"
    | "implementation"
    | "measurement"
    | "constraint"
    | "counterexample"
    | "human_prior";
  directness: "direct" | "inferred";
  attribution: "source_report" | "workflow_inference";
  applicableConditions: string[];
  comparisonBaseline: string | null;
  sourcePath: string;
  sourceUnitId: string;
  sourceFamily: string;
  heading: string;
  quoteOrExactContext: string;
}

export interface EvidenceContradiction {
  contradictionId: string;
  findingIds: string[];
  target:
    | { kind: "claim"; claimKey: string }
    | { kind: "object"; objectRef: ObjectRef };
  summary: string;
  critical: boolean;
}

export interface EvidenceUnanswered {
  unansweredId: string;
  successCriterion: string;
  reason:
    | "no_matching_source"
    | "insufficient_context"
    | "budget_exhausted"
    | "topic_boundary";
  attemptedSearchIds: string[];
}

export interface EvidencePacket {
  packetId: string;
  needId: string;
  needRevision: number;
  status: "complete";
  searches: EvidenceSearch[];
  hitsConsidered: EvidenceHit[];
  contextsRead: EvidenceContext[];
  findings: EvidenceFinding[];
  contradictions: EvidenceContradiction[];
  unanswered: EvidenceUnanswered[];
  conclusion: "answered" | "partial" | "not_found";
  conclusionRationale: string;
}

export interface SemanticDelta {
  deltaId: string;
  basisResultRefs: ObjectRef[];
  expectedTargetRevision: number;
  target:
    | { type: "topic"; id: string }
    | { type: "anchor"; id: string }
    | { type: "direction"; id: string };
  action:
    | "create"
    | "revise"
    | "add_evidence"
    | "add_contradiction"
    | "no_semantic_delta"
    | "reject";
  changedFields: string[];
  rationale: string;
  proposedObject: TopicFrame | Anchor | Direction | null;
}

export interface ExperimentHandoff {
  handoffId: string;
  directionId: string;
  tag: "EXPERIMENT_REQUIRED";
  reason: string;
  requiredArtifact:
    | "trace"
    | "prototype"
    | "benchmark"
    | "equivalence_test"
    | "code_audit"
    | "hardware_measurement";
  hypothesisToTest: string;
  suggestedEntryPoints: string[];
  controlledVariables: string[];
  metrics: string[];
  acceptanceCriteria: string[];
  failureStopConditions: string[];
  executionAuthorized: false;
}

export type DirectionReviewPurpose =
  | "initial"
  | "after_evidence"
  | "terminal_check"
  | "adversarial_recheck";

export interface DirectionDuplicateProjection {
  directionRef: ObjectRef;
  baseline: string;
  comparisonScope: string[];
  controlledVariables: string[];
  primaryChanges: Array<{
    layer: Layer;
    object: string;
    fromState: string;
    toState: string;
    conditions: string[];
  }>;
  causalTargets: string[];
  hypothesis: string;
}

export interface DirectionDuplicateComparison {
  baselineScopeEquivalent: boolean;
  primaryChangeEquivalent: boolean;
  causalTargetEquivalent: boolean;
  materialDifference: string | null;
}

export interface ReviewDelta {
  reviewId: string;
  directionId: string;
  directionRevision: number;
  supportedParts: string[];
  evidenceRefsUsed: string[];
  weakestCausalLink: string | null;
  baselineProblem: string | null;
  implementationProblem: string | null;
  measurementProblem: string | null;
  strongestCounterexample: string | null;
  counterexampleResolution: string | null;
  nextQuestion: string | null;
  nextQuestionAnswerableFromKnowledgeBase: boolean;
  decision:
    | "continue_search"
    | "testable"
    | "experiment_required"
    | "rejected";
  rationale: string;
  duplicateOfDirectionRef: ObjectRef | null;
  duplicateComparison: DirectionDuplicateComparison | null;
  rejectionCategory:
    | "duplicate"
    | "out_of_scope"
    | "causal_contradiction"
    | "unfair_comparison"
    | "no_performance_mechanism"
    | "invalid_evidence"
    | "other"
    | null;
  readinessChecks: {
    inTopicAndAnchorScope: boolean;
    baselineFair: boolean;
    minimumChangeSetExplicit: boolean;
    causalChainFalsifiable: boolean;
    implementationPathBounded: boolean;
    measurementPlanComplete: boolean;
    falsifiersPresent: boolean;
    criticalCounterexampleResolved: boolean;
    evidenceTraceable: boolean;
    knowledgeAnswerableCriticalGapRemaining: boolean;
    newExperimentRequired: boolean;
  };
  experimentHandoff: ExperimentHandoff | null;
}

export interface StopCandidate {
  stopCandidateId: string;
  stopProofId: string;
  runId: string;
  topicId: string;
  canonicalRevision: number;
  reason: string;
}

export interface StopProofClaims {
  topicScopePreserved: boolean;
  noKnowledgeAnswerableCriticalNeed: boolean;
  allAnchorsClosed: boolean;
  allDirectionsTerminal: boolean;
  lastTopicExpansionNoDelta: boolean;
  noUnconsumedOrUncommittedWork: boolean;
  criticalContradictionsReviewed: boolean;
  experimentHandoffsComplete: boolean;
  runtimeEligibleForCompletion: boolean;
  finalOutputTraceable: boolean;
}

export interface StopProof {
  proofId: string;
  stopCandidateId: string;
  canonicalRevision: number;
  topicFrameRevision: number;
  anchorRefs: ObjectRef[];
  directionRefs: ObjectRef[];
  openNeedIds: string[];
  pendingTaskIds: string[];
  inFlightTaskIds: string[];
  pendingOutputRetryTaskIds: string[];
  unconsumedResultIds: string[];
  uncommittedDeltaIds: string[];
  unresolvedValidationFailureIds: string[];
  failedTaskIds: string[];
  unreviewedCriticalContradictionIds: string[];
  experimentHandoffIds: string[];
  lastTopicExpansionNeedId: string | null;
  outputCoverageProjectionId: string;
  claims: StopProofClaims;
}

export interface StopCandidateBundle {
  candidate: StopCandidate;
  proof: StopProof;
}

export interface ClosureChecks {
  stopProofRevisionCurrent: boolean;
  stopProofMatchesCanonical: boolean;
  mechanicalPreflightPassed: boolean;
  topicScopePreserved: boolean;
  noKnowledgeAnswerableCriticalNeed: boolean;
  allAnchorsClosed: boolean;
  allDirectionsTerminal: boolean;
  lastTopicExpansionNoDelta: boolean;
  noUnconsumedOrUncommittedWork: boolean;
  criticalContradictionsReviewed: boolean;
  experimentHandoffsComplete: boolean;
  runtimeEligibleForCompletion: boolean;
  finalOutputTraceable: boolean;
}

export type ClosureFindingType =
  | "knowledge_gap"
  | "state_inconsistency"
  | "incomplete_handoff"
  | "runtime_pause";

export type ClosureRecoveryAction =
  | "REOPEN_FRONTIER"
  | "REPAIR_STATE"
  | "COMPLETE_HANDOFF"
  | "RESUME_RUNTIME";

export type ClosureFindingCode =
  | "stale_stop_proof_revision"
  | "stop_proof_canonical_mismatch"
  | "mechanical_preflight_failed"
  | "topic_scope_silently_narrowed"
  | "knowledge_answerable_open_need"
  | "anchor_not_closed"
  | "anchor_missing_saturation_reason"
  | "anchor_missing_status_reason"
  | "direction_nonterminal"
  | "direction_missing_terminal_reason"
  | "last_topic_expansion_missing"
  | "last_topic_expansion_not_quiet"
  | "pending_task"
  | "in_flight_task"
  | "pending_output_retry"
  | "unconsumed_result"
  | "uncommitted_delta"
  | "unresolved_validation_failure"
  | "failed_task"
  | "unreviewed_critical_contradiction"
  | "experiment_handoff_missing"
  | "experiment_handoff_invalid"
  | "runtime_budget_exhausted"
  | "runtime_failed_or_paused"
  | "final_output_missing_field"
  | "final_output_untraceable";

export interface ClosureScopeRef {
  objectType: string;
  objectId: string;
  revision: number | null;
}

export interface ClosureBlockingFinding {
  findingId: string;
  check: keyof ClosureChecks;
  type: ClosureFindingType;
  code: ClosureFindingCode;
  summary: string;
  objectRefs: ClosureScopeRef[];
  reopenScope: ClosureScopeRef;
  recoveryAction: ClosureRecoveryAction;
}

export interface ClosureBasis {
  check: keyof ClosureChecks;
  statement: string;
  objectRefs: ClosureScopeRef[];
}

export type FinalizationRequirement =
  | "canonical_revision_unchanged"
  | "full_validator_passed"
  | "final_output_rendered"
  | "final_output_coverage_validated"
  | "atomic_completed_commit";

export interface ClosureReview {
  reviewId: string;
  stopCandidateId: string;
  canonicalRevision: number;
  status: "complete";
  decision: "accept" | "reject";
  verifiedClosureBasis: ClosureBasis[];
  closureChecks: ClosureChecks;
  blockingFindings: ClosureBlockingFinding[];
  reopenScopes: ClosureScopeRef[];
  allowsFinalization: boolean;
  finalizationRequirements: FinalizationRequirement[];
  rationale: string;
}
