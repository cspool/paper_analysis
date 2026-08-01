import type {
  Anchor,
  ArtifactRef,
  BlockedReport,
  BudgetState,
  ClosureReview,
  Direction,
  DirectionDuplicateProjection,
  DirectionReviewPurpose,
  EvidenceFinding,
  EvidencePacket,
  GateDefinitionDraft,
  KnowledgeDimension,
  ObjectRef,
  PauseProposal,
  RegisteredRole,
  RegisteredTurnOutputMessageType,
  ReviewDelta,
  RubricBinding,
  SearchNeed,
  SemanticDelta,
  StageContractDraft,
  StateBinding,
  StopCandidateBundle,
  StopProofClaims,
  TopicFrame,
  TurnCorrectionFeedback,
  TurnBudget,
  UserQuestion,
  WorkflowDecisionAction,
  WorkflowLifecycle,
  WorkflowPermissionEnvelope,
  WorkflowPlanPatch,
  WorkflowTrigger,
} from "./index.ts";

export interface TurnIdentity {
  protocolVersion: 1;
  messageType: string;
  workflowId: string;
  runId: string;
  taskId: string;
  attemptId: string;
  stageId: string;
  stageContractHash: string;
}

export interface PayloadTurnEnvelope<T> extends TurnIdentity {
  stateBinding: StateBinding;
  inputHash: string;
  payload: T;
}

export interface SkillBinding {
  name: string;
  version: string;
  sha256: string;
}

export interface SchemaBinding {
  manifestSha256: string;
  expectedOutputMessageType: RegisteredTurnOutputMessageType;
  expectedOutputSchemaSha256: string;
}

export interface TurnPermissionEnvelope {
  role: RegisteredRole;
  tools: string[];
  allowedPathPrefixes: string[];
  filesystem: "none" | "vault_read_only";
  network: false;
  delegation: false;
  goals: false;
  stateWrite: false;
  experimentExecution: false;
  maxBudget: TurnBudget;
}

export interface PreviousEvidenceQuery {
  query: string;
  dimension: KnowledgeDimension;
  logicalQueryLevel: 1 | 2 | 3;
  outcome: string;
}

export interface PreviousEvidenceRead {
  path: string;
  sourceUnitId: string;
  contextHash: string;
}

export interface EvidenceFocusProjection {
  topic: TopicFrame;
  anchor: Anchor | null;
  direction: Direction | null;
}

export interface EvidenceReaderTask {
  searchNeed: SearchNeed;
  focus: EvidenceFocusProjection;
  previousQueries: PreviousEvidenceQuery[];
  previousReads: PreviousEvidenceRead[];
  consumedSourceUnitIds: string[];
  allowedVaultRoots: Array<{
    dimension: KnowledgeDimension;
    relativePathPrefix: string;
  }>;
  budget: TurnBudget;
  skill: SkillBinding;
  schema: SchemaBinding;
  permission: TurnPermissionEnvelope;
  correctionFeedback: TurnCorrectionFeedback | null;
  terminationCondition: string;
}

export type EvidenceReaderTaskEnvelope =
  PayloadTurnEnvelope<EvidenceReaderTask> & {
    messageType: "EVIDENCE_READER_TASK";
  };

export type EvidencePacketEnvelope = PayloadTurnEnvelope<EvidencePacket> & {
  messageType: "EVIDENCE_PACKET";
};

export interface DirectionReviewTask {
  topic: TopicFrame;
  anchor: Anchor;
  direction: Direction;
  siblingDirections: DirectionDuplicateProjection[];
  evidenceFindings: EvidenceFinding[];
  contradictingEvidence: EvidenceFinding[];
  unresolvedSearchNeeds: SearchNeed[];
  counterexamples: Array<{
    counterexampleId: string;
    statement: string;
    evidenceRefs: string[];
    degradationCondition: string | null;
  }>;
  priorReviews: Array<{
    reviewId: string;
    directionRevision: number;
    decision: ReviewDelta["decision"];
  }>;
  reviewPurpose: DirectionReviewPurpose;
  rubric: RubricBinding;
  allowedEvidenceIds: string[];
  allowedObjectRefs: ObjectRef[];
  inputArtifacts: ArtifactRef[];
  budget: TurnBudget;
  skill: SkillBinding;
  schema: SchemaBinding;
  permission: TurnPermissionEnvelope;
  correctionFeedback: TurnCorrectionFeedback | null;
  terminationCondition: string;
}

export type DirectionReviewTaskEnvelope =
  PayloadTurnEnvelope<DirectionReviewTask> & {
    messageType: "DIRECTION_REVIEW_TASK";
  };

export type ReviewDeltaEnvelope = PayloadTurnEnvelope<ReviewDelta> & {
  messageType: "REVIEW_DELTA";
};

export interface AnchorClosureProjection {
  anchorRef: ObjectRef;
  status: Anchor["status"];
  statusReason: string;
  saturationReason: string | null;
}

export interface DirectionTerminalProjection {
  directionRef: ObjectRef;
  anchorId: string;
  status: Direction["status"];
  statusReason: string;
  experimentHandoffId: string | null;
}

export interface NeedClosureProjection {
  needId: string;
  revision: number;
  owner: SearchNeed["owner"];
  critical: boolean;
  answerability: SearchNeed["answerability"];
  status: SearchNeed["status"];
}

export interface WorkIndexItem {
  id: string;
  status: string;
  resolvedById: string | null;
  objectRef: ObjectRef | null;
}

export interface ContradictionClosureProjection {
  contradictionId: string;
  critical: boolean;
  dispositionReviewId: string | null;
  objectRef: ObjectRef;
}

export interface HandoffClosureProjection {
  handoffId: string;
  directionId: string;
  complete: boolean;
  executionAuthorized: false;
}

export interface TopicExpansionRecord {
  needId: string;
  needRevision: number;
  intent: "discover_anchor";
  ownerTopicId: string;
  completed: boolean;
  outcome: "no_new_anchor_no_critical_delta" | "semantic_delta";
  noDeltaRecordId: string | null;
  semanticDeltaId: string | null;
}

export interface MechanicalPreflightReport {
  preflightId: string;
  stopCandidateId: string;
  canonicalRevision: number;
  checks: Array<{
    checkId: string;
    passed: boolean;
    issueIds: string[];
  }>;
  passed: boolean;
}

export type OutputCoverageField =
  | "topic_scope"
  | "anchor_summaries"
  | "direction_statuses"
  | "evidence_provenance"
  | "contradictions_and_limits"
  | "experiment_handoffs"
  | "unresolved_questions";

export interface OutputCoverageProjection {
  projectionId: string;
  fields: Record<OutputCoverageField, ObjectRef[]>;
}

export interface ClosureReviewTask {
  stopCandidateBundle: StopCandidateBundle;
  currentCanonicalRevision: number;
  topic: TopicFrame;
  anchors: AnchorClosureProjection[];
  directions: DirectionTerminalProjection[];
  needs: NeedClosureProjection[];
  taskIndex: WorkIndexItem[];
  resultIndex: WorkIndexItem[];
  deltaIndex: WorkIndexItem[];
  outputAttemptIndex: WorkIndexItem[];
  validationFailureIndex: WorkIndexItem[];
  recentSemanticRecords: ObjectRef[];
  recentNoDeltaRecords: ObjectRef[];
  lastTopicExpansion: TopicExpansionRecord | null;
  contradictions: ContradictionClosureProjection[];
  experimentHandoffs: HandoffClosureProjection[];
  mechanicalPreflight: MechanicalPreflightReport;
  rubric: RubricBinding;
  budgetState: BudgetState;
  lifecycle: WorkflowLifecycle;
  runtimeEligibility: {
    budgetExhausted: boolean;
    paused: boolean;
    blocked: boolean;
    failed: boolean;
    reason: string | null;
  };
  outputCoverage: OutputCoverageProjection;
  freshTurn: true;
  providerHistoryIncluded: false;
  canonicalOnly: true;
  budget: TurnBudget;
  skill: SkillBinding;
  schema: SchemaBinding;
  permission: TurnPermissionEnvelope;
  correctionFeedback: TurnCorrectionFeedback | null;
  terminationCondition: string;
}

export type ClosureReviewTaskEnvelope =
  PayloadTurnEnvelope<ClosureReviewTask> & {
    messageType: "CLOSURE_REVIEW_TASK";
  };

export type ClosureReviewEnvelope = PayloadTurnEnvelope<ClosureReview> & {
  messageType: "CLOSURE_REVIEW";
};

export interface WorkflowTaskIndex {
  pendingTaskIds: string[];
  inFlightTaskIds: string[];
  failedTaskIds: string[];
  pendingOutputRetryTaskIds: string[];
}

export interface WorkflowResultIndex {
  committedUnconsumedResultRefs: ObjectRef[];
  consumedResultRefs: ObjectRef[];
}

export interface WorkflowEventProjection {
  eventCursor: number;
  eventType: string;
  summary: string;
  objectRefs: ObjectRef[];
}

export interface WorkflowCompletionProjection {
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
  eligibleForProposal: boolean;
  blockingClaims: Array<keyof StopProofClaims>;
}

export interface WorkflowTriggerReport {
  reportId: string;
  trigger: WorkflowTrigger;
  sourceStageId: string | null;
  sourceAttemptId: string | null;
  facts: string[];
  issueCodes: string[];
}

export interface WorkflowTurnTask extends TurnIdentity {
  messageType: "WORKFLOW_TURN_TASK";
  stateSnapshot: StateBinding;
  decisionInputHash: string;
  trigger: WorkflowTrigger;
  immutableObjective: string;
  immutableAcceptanceCriteria: string[];
  objectiveHash: string;
  acceptanceCriteriaHash: string;
  lifecycle: WorkflowLifecycle;
  activeFocusRef: ObjectRef | null;
  domainProjection: {
    topic: TopicFrame | null;
    focusAnchor: Anchor | null;
    focusDirection: Direction | null;
    searchNeeds: SearchNeed[];
    evidencePackets: EvidencePacket[];
    directionReviews: ReviewDelta[];
    stopCandidateBundle: StopCandidateBundle | null;
    completionProjection: WorkflowCompletionProjection | null;
  };
  taskIndex: WorkflowTaskIndex;
  resultIndex: WorkflowResultIndex;
  relevantPlan: {
    revision: number;
    stageIds: string[];
    dependencyIds: string[];
  };
  approvedArtifacts: ArtifactRef[];
  triggerReport: WorkflowTriggerReport;
  recentEvents: WorkflowEventProjection[];
  skill: SkillBinding;
  schema: SchemaBinding;
  permission: WorkflowPermissionEnvelope;
  correctionFeedback: TurnCorrectionFeedback | null;
  terminationCondition: string;
}

export type DomainProposal =
  | { kind: "topic_frame"; value: TopicFrame }
  | { kind: "search_need"; value: SearchNeed }
  | { kind: "semantic_delta"; value: SemanticDelta }
  | {
      kind: "direction_review_request";
      value: {
        directionRef: ObjectRef;
        purpose: DirectionReviewPurpose;
        rubric: RubricBinding;
      };
    }
  | { kind: "stop_candidate"; value: StopCandidateBundle };

export interface WorkflowDecisionProposal extends TurnIdentity {
  messageType: "WORKFLOW_DECISION_PROPOSAL";
  expectedState: StateBinding;
  decisionInputHash: string;
  proposalId: string;
  decision: WorkflowDecisionAction;
  reason: string;
  assumptions: string[];
  proposedStageContract: StageContractDraft | null;
  proposedGateDefinition: GateDefinitionDraft | null;
  proposedPlanPatch: WorkflowPlanPatch | null;
  targetStageId: string | null;
  domainProposal: DomainProposal | null;
  basisArtifactRefs: ArtifactRef[];
  basisResultRefs: ObjectRef[];
  requestedUserInput: UserQuestion | null;
  blockedReport: BlockedReport | null;
  pauseProposal: PauseProposal | null;
  confidence: number | null;
}

export type AnyTurnTask =
  | WorkflowTurnTask
  | EvidenceReaderTaskEnvelope
  | DirectionReviewTaskEnvelope
  | ClosureReviewTaskEnvelope;

export type AnyTurnResult =
  | WorkflowDecisionProposal
  | EvidencePacketEnvelope
  | ReviewDeltaEnvelope
  | ClosureReviewEnvelope;
