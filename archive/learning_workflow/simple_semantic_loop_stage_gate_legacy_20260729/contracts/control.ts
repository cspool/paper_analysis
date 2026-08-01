import type {
  ArtifactRef,
  ObjectRef,
  RegisteredRole,
  RegisteredTurnOutputMessageType,
} from "./index.ts";

export type WorkflowLifecycle =
  | "initialized"
  | "running"
  | "waiting_turn"
  | "waiting_user"
  | "waiting_external"
  | "closure_preflight"
  | "waiting_closure_review"
  | "finalizing"
  | "paused_budget"
  | "paused_operator"
  | "failed_retriable"
  | "failed_terminal"
  | "blocked_semantic"
  | "blocked_external"
  | "completed"
  | "cancelled";

export interface StateBinding {
  snapshotVersion: number;
  canonicalRevision: number;
  eventCursor: number;
  workflowPlanRevision: number;
}

export interface BudgetState {
  turnsUsed: number;
  maxTurns: number;
  inputTokensUsed: number;
  maxInputTokens: number;
  outputTokensUsed: number;
  maxOutputTokens: number;
  toolCallsUsed: number;
  maxToolCalls: number;
  elapsedMs: number;
  maxElapsedMs: number;
  exhaustedDimensions: Array<
    "turns" | "input_tokens" | "output_tokens" | "tool_calls" | "elapsed"
  >;
}

export interface WorkflowState {
  workflowId: string;
  runId: string;
  snapshotVersion: number;
  canonicalRevision: number;
  eventCursor: number;
  workflowPlanRevision: number;
  lifecycle: WorkflowLifecycle;
  currentStageId: string | null;
  activeFocusRef: ObjectRef | null;
  runnableStageIds: string[];
  pendingTaskIds: string[];
  inFlightTaskIds: string[];
  committedUnconsumedResultIds: string[];
  pendingProposalIds: string[];
  retryCounters: Record<string, number>;
  noProgressCounters: Record<string, number>;
  budgetState: BudgetState;
  pauseOrBlockReason: string | null;
}

export type RegisteredStageType =
  | "SCRIPT_APPLY_TOPIC_FRAME"
  | "SCRIPT_APPLY_SEMANTIC_DELTA"
  | "WORKFLOW_DECISION"
  | "EVIDENCE_READ"
  | "DIRECTION_REVIEW"
  | "CLOSURE_REVIEW"
  | "RENDER_FINAL";

export type StageExecutionKind =
  | "SCRIPT_TRANSITION"
  | "DECISION_TURN"
  | "WORKER_TURN"
  | "EVALUATOR_TURN";

export type StageLifecycle =
  | "draft_proposed"
  | "validated"
  | "frozen"
  | "runnable"
  | "dispatched"
  | "result_received"
  | "gate_running"
  | "passed"
  | "failed"
  | "blocked"
  | "committed"
  | "consumed"
  | "superseded";

export interface EvidenceReadBudget {
  maxLogicalQueries: number;
  maxSearchToolCalls: number;
  maxHitsConsidered: number;
  maxSelectedSources: number;
  maxContextsRead: number;
}

export interface TurnBudget {
  timeoutMs: number;
  maxInputTokens: number;
  maxOutputTokens: number;
  maxToolCalls: number;
  evidenceRead: EvidenceReadBudget | null;
}

export interface StageNode {
  stageId: string;
  stageType: RegisteredStageType;
  executionKind: StageExecutionKind;
  role: RegisteredRole | null;
  contractId: string;
  gateId: string;
  lifecycle: StageLifecycle;
  createdAtSnapshotVersion: number;
  supersededReason: string | null;
}

export interface StageDependency {
  dependencyId: string;
  predecessorStageId: string;
  successorStageId: string;
  kind: "requires_committed" | "requires_consumed";
}

export interface WorkflowPlan {
  workflowId: string;
  revision: number;
  objectiveHash: string;
  acceptanceCriteriaHash: string;
  stageNodes: StageNode[];
  dependencies: StageDependency[];
  planStatus: "active" | "superseded" | "closed";
}

export interface WorkflowPlanPatch {
  expectedPlanRevision: number;
  operations: Array<
    | { op: "add_stage"; stage: StageNodeDraft }
    | { op: "supersede_stage"; stageId: string; reason: string }
    | { op: "add_dependency"; dependency: StageDependency }
    | { op: "remove_dependency"; dependencyId: string; reason: string }
  >;
  objectiveHash: string;
  acceptanceCriteriaHash: string;
  rationale: string;
}

export interface StageContractDraft {
  proposalLocalStageKey: string;
  stageType: RegisteredStageType;
  objective: string;
  scope: ObjectRef[];
  executionKind: StageExecutionKind;
  role: RegisteredRole | null;
  requiredInputs: ArtifactRef[];
  expectedOutputMessageType: RegisteredTurnOutputMessageType | null;
  requestedTools: string[];
  requestedPaths: string[];
  prohibitedActions: string[];
  budget: TurnBudget;
}

export interface StageNodeDraft {
  proposalLocalStageKey: string;
  stageType: RegisteredStageType;
  executionKind: StageExecutionKind;
  role: RegisteredRole | null;
  objective: string;
  dependsOnStageIds: string[];
  contract: StageContractDraft;
  gate: GateDefinitionDraft;
}

export type GateValueType =
  | "string"
  | "number"
  | "boolean"
  | "null"
  | "string_array"
  | "object";

export type RuntimeGateFact =
  | "allowed_tool_events_only"
  | "allowed_paths_only"
  | "turn_budget_within_contract"
  | "experiment_execution_count"
  | "external_evidence_used";

export type ValidatorGateFact =
  | "schema_valid"
  | "message_binding_matches"
  | "registered_validator_passes"
  | "references_resolve"
  | "source_context_present"
  | "duplicate_commit"
  | "script_transition_valid";

export type GateActual =
  | {
      source: "result";
      pointer: string;
      valueType: GateValueType;
    }
  | {
      source: "task";
      pointer: string;
      valueType: GateValueType;
    }
  | {
      source: "canonical";
      objectRef: ObjectRef;
      pointer: string;
      valueType: GateValueType;
    }
  | {
      source: "runtime";
      fact: RuntimeGateFact;
      valueType: GateValueType;
    }
  | {
      source: "validator";
      fact: ValidatorGateFact;
      pointer: string | null;
      valueType: GateValueType;
    }
  | {
      source: "artifact";
      artifactId: string;
      fact: "exists" | "sha256";
      valueType: GateValueType;
    };

export type MechanicalGatePredicate = "equals" | "contains_fields";

export type GateExpected =
  | string
  | number
  | boolean
  | null
  | string[];

export interface MechanicalGateCheck {
  checkId: string;
  predicate: MechanicalGatePredicate;
  actual: GateActual;
  expected: GateExpected;
}

export interface GateDefinitionDraft {
  proposalLocalStageKey: string;
  mechanicalChecks: MechanicalGateCheck[];
  semanticEvaluation: {
    required: boolean;
    evaluatorRole: RegisteredRole | null;
    rubricId: string | null;
    inputProjection: string[];
    expectedOutputMessageType: RegisteredTurnOutputMessageType | null;
  };
}

export interface StageContract extends StageContractDraft {
  contractId: string;
  stageId: string;
  revision: number;
  definedAtSnapshotVersion: number;
  sha256: string;
}

export interface GateDefinition extends GateDefinitionDraft {
  gateId: string;
  stageId: string;
  stageContractHash: string;
  proposedCriteriaSha256: string;
  compilerPolicyVersion: string;
  evaluatorVersion: string;
  revision: number;
  definedAtSnapshotVersion: number;
  sha256: string;
}

export interface UserQuestion {
  questionId: string;
  prompt: string;
  rationale: string;
  choices: Array<{ choiceId: string; label: string; consequence: string }>;
  requiredAuthority: string | null;
  relatedRefs: ObjectRef[];
}

export interface BlockedReport {
  blockedReportId: string;
  kind: "semantic" | "external";
  summary: string;
  blockingFacts: string[];
  exhaustedAlternatives: string[];
  relatedRefs: ObjectRef[];
  userActionNeeded: string | null;
}

export interface PauseProposal {
  pauseProposalId: string;
  reason: string;
  category: "budget" | "operator" | "external_dependency";
  resumableWhen: string;
  relatedRefs: ObjectRef[];
}

export interface ValidationError {
  code: string;
  jsonPointer: string | null;
  message: string;
}

export interface ValidationReport {
  validatorVersion: string;
  valid: boolean;
  errors: ValidationError[];
  checkedArtifactHashes: string[];
  checkedObjectRefs: ObjectRef[];
}

export interface TurnCorrectionError extends ValidationError {
  requiredRule: string;
  validExamples: string[];
}

export interface TurnCorrectionFeedback {
  previousAttemptId: string;
  previousOutputSha256: string;
  validationReportId: string;
  validationReportSha256: string;
  failureClass:
    | "STRUCTURE_INVALID"
    | "BINDING_INVALID"
    | "SEMANTIC_INVALID";
  errors: TurnCorrectionError[];
  retryInstruction: string;
}

export interface RolePermission {
  role: RegisteredRole;
  allowedStageTypes: RegisteredStageType[];
  allowedTools: string[];
  allowedPathPrefixes: string[];
  maxBudget: TurnBudget;
}

export interface WorkflowPermissionEnvelope {
  allowedActions: WorkflowDecisionAction[];
  allowedStageTypes: RegisteredStageType[];
  allowedRoles: RegisteredRole[];
  allowedTools: string[];
  allowedPathPrefixes: string[];
  registeredRubrics: string[];
  maxBudgetByRole: Partial<Record<RegisteredRole, TurnBudget>>;
  suppliedObjectRefs: ObjectRef[];
  suppliedArtifactIds: string[];
  suppliedResultRefs: ObjectRef[];
}

export type WorkflowTrigger =
  | "INITIALIZE_TOPIC"
  | "COMMITTED_RESULT_REQUIRES_INTEGRATION"
  | "FRONTIER_SELECTION_REQUIRED"
  | "MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE"
  | "GATE_FAILED_WITHOUT_RECOVERY_RULE"
  | "PLAN_EXHAUSTED_OBJECTIVE_OPEN"
  | "EVIDENCE_CONTRADICTION"
  | "NO_PROGRESS_THRESHOLD_REACHED"
  | "CLOSURE_REJECTED"
  | "NO_RUNNABLE_STAGE"
  | "USER_DECISION_REQUIRED";

export type WorkflowDecisionAction =
  | "RUN_STAGE"
  | "RETRY_STAGE"
  | "REPLAN"
  | "REQUEST_EVALUATION"
  | "ASK_USER"
  | "REPORT_BLOCKED"
  | "PROPOSE_PAUSE"
  | "PROPOSE_COMPLETE";
