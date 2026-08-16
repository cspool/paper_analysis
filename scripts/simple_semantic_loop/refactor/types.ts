export const LOOP_ROLES = [
  "DECISION",
  "WORKER",
  "REVIEWER",
  "EXP_GOAL",
] as const;
export type LoopRole = (typeof LOOP_ROLES)[number];
/** Provider-facing roles may be extended by sibling deterministic workflows. */
export type RuntimeRole =
  | LoopRole
  | "EXPERIMENT_DECISION"
  | "EVIDENCE_JUDGE"
  | "DIRECTION_LAB_GOAL";

export const CURRENT_FORMAT_VERSION = 8 as const;
export const READABLE_FORMAT_VERSIONS = [5, 6, 7, CURRENT_FORMAT_VERSION] as const;
export type RunFormatVersion = (typeof READABLE_FORMAT_VERSIONS)[number];

export const TASK_ACTIONS = [
  "CREATE_ANCHOR",
  "DEEPEN_ANCHOR",
  "CREATE_DIRECTION",
  "DEEPEN_DIRECTION",
  "REVIEW_ANCHOR",
  "REVIEW_DIRECTION",
] as const;
export type TaskAction = (typeof TASK_ACTIONS)[number];
export type WorkAction = Extract<
  TaskAction,
  | "CREATE_ANCHOR"
  | "DEEPEN_ANCHOR"
  | "CREATE_DIRECTION"
  | "DEEPEN_DIRECTION"
>;
export type ReviewAction = Extract<
  TaskAction,
  "REVIEW_ANCHOR" | "REVIEW_DIRECTION"
>;

export const LOOP_DECISIONS = [
  "RUN_WORKER",
  "RUN_REVIEWER",
  "RUN_EXP_GOAL",
  "FINISH_WORKFLOW",
  "RETRY_WORKER",
  "RETRY_REVIEWER",
] as const;
export type LoopDecision = (typeof LOOP_DECISIONS)[number];

export const TURN_STATES = [
  "RUNNING",
  "INVALID_OUTPUT",
  "PENDING_DECISION",
  "COMMITTED",
  "SUPERSEDED_BY_RETRY",
  "RUNTIME_FAILED",
] as const;
export type TurnState = (typeof TURN_STATES)[number];

export const OUTPUT_CAPTURE_STATES = [
  "NONE",
  "PARTIAL",
  "COMPLETE",
] as const;
export type OutputCaptureState = (typeof OUTPUT_CAPTURE_STATES)[number];

export const RUNTIME_FAILURE_KINDS = [
  "IDLE_TIMEOUT",
  "HARD_TIMEOUT",
  "PROVIDER_ERROR",
] as const;
export type RuntimeFailureKind = (typeof RUNTIME_FAILURE_KINDS)[number];

export type WorkflowFailureKind =
  | "RUNTIME_RETRY_EXHAUSTED"
  | "NON_RECOVERABLE";

export type Lifecycle = "RUNNING" | "PAUSED" | "FINISHED" | "FAILED";
export const PAUSE_KINDS = [
  "ROUND_BUDGET_EXHAUSTED",
  "EXP_GOAL_PAUSED",
  "OPERATOR_REQUESTED",
] as const;
export type PauseKind = (typeof PAUSE_KINDS)[number];
export type ObjectKind = "ANCHOR" | "DIRECTION";
export const WORK_OUTCOMES = [
  "READY_FOR_REVIEW",
  "PARTIAL_RESULT",
  "BLOCKED_NO_RESULT",
] as const;
export type WorkOutcome = (typeof WORK_OUTCOMES)[number];
export const REVIEW_VERDICTS = ["PASS", "REVISE", "REJECT"] as const;
export type ReviewVerdict = (typeof REVIEW_VERDICTS)[number];

// G01: the only immutable requirement message visible to every Agent.
export interface WorkflowGoal {
  topic: string;
  objective: string;
  acceptanceCriteria: string[];
}

// T01: the only task message visible to Worker and Reviewer.
export interface TurnTaskInputs {
  boundAnchor?: string;
  currentWork?: string;
  latestReview?: string;
  reviewTarget?: string;
  previousReview?: string;
  researchMemory?: string;
  experimentResults?: string;
  /** Script-generated, task-local navigation over reviewed negative EXP results. */
  negativeExperimentHistoryRef?: string;
}

export interface TurnTask {
  goalRef: "workflow_goal.json";
  action: TaskAction;
  objective: string;
  inputs: TurnTaskInputs;
  requirements: string[];
  constraints: string[];
}

export interface CommittedAnchorResult {
  objectKind: "ANCHOR";
  work: string;
  review: string;
}

export interface CommittedDirectionResult {
  objectKind: "DIRECTION";
  anchorWork: string;
  work: string;
  review: string;
}

export type CommittedResult =
  | CommittedAnchorResult
  | CommittedDirectionResult;

export interface PendingAnchorResults {
  objectKind: "ANCHOR";
  workTask: string;
  work: string;
  review: string;
}

export interface PendingDirectionResults {
  objectKind: "DIRECTION";
  anchorWork: string;
  workTask: string;
  work: string;
  review: string;
}

export type PendingResults =
  | PendingAnchorResults
  | PendingDirectionResults;

// D01: the complete projection visible to Decision.
export interface DecisionContext {
  goalRef: "workflow_goal.json";
  committedResults: CommittedResult[];
  pendingResults: PendingResults | null;
  remainingRequirementsAfterPendingCommit: string[];
  experimentContext: ExperimentContext | null;
  observationRef: string;
}

export interface ExperimentContext {
  anchorWork: string;
  directionWork: string | null;
  previousResultRefs: string[];
}

export interface EvidenceItem {
  sourceRef: string;
  supports: string;
}

export interface AnchorContent {
  name: string;
  scenario: string;
  baseline: string;
  performanceTension: string;
  scope6L: {
    L1: string | null;
    L2: string | null;
    L3: string | null;
    L4: string | null;
    L5: string | null;
    L6: string | null;
  };
  constraints: string[];
}

export interface DirectionContent {
  name: string;
  mechanism: string;
  baselineChange: string;
  expectedEffects: Array<{
    metric: string;
    effect: string;
    conditions: string;
  }>;
  tradeoffs: string[];
  failureConditions: string[];
  measurementPlan: string[];
}

export interface WorkResult<T = unknown> {
  workOutcome: WorkOutcome;
  content?: T | null;
  evidence?: unknown;
  unresolved?: unknown;
  [key: string]: unknown;
}

export interface ReviewFinding {
  severity: "BLOCKING" | "NON_BLOCKING";
  issue: string;
  basis: string;
  expected: string;
}

export interface QueryGap {
  question: string;
  dimension: "experiment" | "idea" | "knowledge" | "human";
  reason: string;
}

export interface ReviewResult {
  reviewVerdict: ReviewVerdict;
  summary?: string;
  findings?: ReviewFinding[];
  queryGaps?: QueryGap[];
  [key: string]: unknown;
}

export interface OutputError {
  check: "DECISION_PROTOCOL" | "JSON_PARSE" | "CORE_CONTROL";
  path: string;
  message: string;
}

// E01: the only error message returned to an Agent.
export interface OutputErrorReport {
  errors: OutputError[];
}

// O01: the only run outcome returned to the caller.
export interface RunOutcome {
  workflowOutcome: "FINISHED" | "FAILED" | "PAUSED";
  reportRef: "final/report.md" | null;
  reason: string | null;
}

export interface DecisionProtocolResult {
  decision: LoopDecision;
  guidance: string | null;
}

export interface WorkControlProjection {
  role: "WORKER";
  workOutcome: WorkOutcome;
}

export interface ReviewControlProjection {
  role: "REVIEWER";
  reviewVerdict: ReviewVerdict;
}

export interface DecisionControlProjection {
  role: "DECISION";
  decision: LoopDecision;
  guidance: string | null;
}

export type CoreControlProjection =
  | WorkControlProjection
  | ReviewControlProjection
  | DecisionControlProjection;

// Controller-internal records below are never Agent communication messages.
export interface SkillPin {
  name: string;
  path: string;
  sha256: string;
}

export interface RefCatalogEntry {
  path: string;
  sha256: string;
  templateSchemaPath?: string;
}

export type RefCatalog = Record<string, RefCatalogEntry>;

export interface RunBudgets {
  maxRounds: number;
  maxExperimentGoals: number;
  /** null means the Controller does not set a Codex Goal token budget. */
  experimentGoalTokenBudget: number | null;
  maxOutputRetries: number;
  maxRuntimeRetries: number;
  maxSemanticRetries: number;
  timeoutProfiles: Record<LoopRole, TurnTimeoutProfile>;
  maxInputTokens: number;
  maxOutputTokens: number;
}

export interface TurnTimeoutProfile {
  idleTimeoutMs: number;
  hardTimeoutMs: number;
  interruptGraceMs: number;
}

export interface RunFile {
  formatVersion: RunFormatVersion;
  runId: string;
  createdAt: string;
  projectRoot: string;
  model: string;
  goalRef: "workflow_goal.json";
  skills: {
    decision: SkillPin;
    worker: SkillPin;
    reviewer: SkillPin;
    experiment: SkillPin;
  };
  budgets: RunBudgets;
  continuation?: {
    sourceRunId: string;
    sourceWorkDir: string;
    sourceStateRevision: number;
    sourceRunSha256: string;
    sourceStateSha256: string;
    sourceLifecycle?: Lifecycle;
    sourceManifestSha256: string | null;
    sourceRunRef: string;
    sourceStateRef: string;
    sourceManifestRef: string | null;
    sourceFinalReportRef: string | null;
    /** True when this branch receives a fresh round and EXP authorization. */
    budgetReset?: boolean;
    sourceExperimentCount?: number;
    continuedAt: string;
  };
}

export interface TaskBinding {
  bindingId: string;
  createdAt: string;
  role: "WORKER" | "REVIEWER";
  action: TaskAction;
  taskRef: string;
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  resultRefName:
    | "work-result-anchor-v2"
    | "work-result-direction-v2"
    | "review-result-v2";
  sourceDecisionTurnRef: string | null;
}

export interface ValidationAuditCheck {
  check: OutputError["check"];
  path: string;
  passed: boolean;
  message: string;
}

export interface ValidationAuditAdvisory {
  check: "REFERENCE_TEMPLATE";
  path: string;
  message: string;
}

export interface ValidationAudit {
  checks: ValidationAuditCheck[];
  advisories: ValidationAuditAdvisory[];
}

export interface TurnFile {
  turnId: string;
  role: LoopRole;
  round: number;
  attempt: number;
  taskBindingRef: string | null;
  decisionContextRef: string | null;
  retryOf: string | null;
  skill: string;
  turnState: TurnState;
  startedAt: string;
  completedAt: string | null;
  promptRef: string;
  outputCapture: OutputCaptureState;
  partialOutputRef: string | null;
  rawOutputRef: string | null;
  resultRef: string | null;
  controlRef: string | null;
  validationAuditRef: string;
  errorReportRef: string | null;
  runtimeErrorRef: string | null;
  runtimeRef: string;
  providerThreadId: string | null;
  providerTurnId: string | null;
  providerStatus: RawTurnResult["status"] | null;
  runtimeFailureKind: RuntimeFailureKind | null;
  timeoutProfile: TurnTimeoutProfile;
  recoveryRef: string | null;
}

export interface ObjectRevision {
  revision: number;
  workTaskRef: string;
  workRef: string;
  workOutcome: WorkOutcome;
  reviewRef: string;
  reviewVerdict: ReviewVerdict;
  committedByDecisionTurnRef: string;
}

export interface AnchorIndexEntry {
  objectId: string;
  latestRevision: number;
  revisions: Record<string, ObjectRevision>;
  directionIds: string[];
  rejected: boolean;
}

export interface DirectionIndexEntry {
  objectId: string;
  parentAnchorId: string;
  latestRevision: number;
  revisions: Record<string, ObjectRevision>;
  rejected: boolean;
}

export interface ObjectsIndex {
  revision: number;
  activeAnchorIds: string[];
  anchors: Record<string, AnchorIndexEntry>;
  directions: Record<string, DirectionIndexEntry>;
}

export interface RoundFile {
  round: number;
  branch: "INITIAL" | "CONTINUATION" | LoopDecision;
  turnRefs: string[];
  experimentRefs: string[];
  committedAt: string | null;
}

export type SequenceMode =
  | "NORMAL_WORK"
  | "PAIR_REVIEW"
  | "PRE_REVIEW"
  | "POST_EXP_REVIEW"
  | "ANCHOR_REASSESS"
  | "DECISION"
  | "EXP_GOAL"
  | "RETRY_WORK"
  | "RETRY_REVIEW";

export interface SequenceStep {
  role: LoopRole;
  mode: SequenceMode;
  bindingRef: string | null;
}

export interface PendingPair {
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  workTaskBindingRef: string;
  workTaskRef: string;
  workTurnRef: string;
  workRef: string;
  workOutcome: WorkOutcome;
  reviewTaskBindingRef: string | null;
  reviewTurnRef: string | null;
  reviewRef: string | null;
  reviewVerdict: ReviewVerdict | null;
}

export interface PreReview {
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  workRef: string;
  workOutcome: WorkOutcome;
  reviewTaskBindingRef: string;
  reviewTurnRef: string;
  reviewRef: string;
  reviewVerdict: ReviewVerdict;
}

export interface CorrectionState {
  role: LoopRole;
  retryOfTurnRef: string;
  previousOutputRef: string;
  errorReportRef: string;
  correctRefName: string;
}

export interface StateFile {
  formatVersion: RunFormatVersion;
  revision: number;
  lifecycle: Lifecycle;
  reason: string | null;
  pauseKind?: PauseKind | null;
  roundBudget?: {
    authorizedThroughRound: number;
    lastAuthorizationRef: string | null;
  };
  failureKind: WorkflowFailureKind | null;
  node: LoopRole | null;
  round: number;
  transitions: number;
  sequence: SequenceStep[];
  activeTaskBindingRef: string | null;
  activeTurnRef: string | null;
  activeExperimentRef: string | null;
  latestExperimentResultRef: string | null;
  /** EXP Goals charged to this run's current authorization window. */
  experimentGoalsStarted: number;
  pending: PendingPair | null;
  preReview: PreReview | null;
  decisionGuidance: string | null;
  correction: CorrectionState | null;
  runtimeRecovery: RuntimeRecoveryAuthorization | null;
  latestDecisionTurnRef: string | null;
  semanticRetries: {
    worker: number;
    reviewer: number;
  };
}

export interface RoundAuthorizationRecord {
  formatVersion: RunFormatVersion;
  authorizationId: string;
  createdAt: string;
  sourceStateRevision: number;
  additionalRounds: number;
  firstAuthorizedRound: number;
  authorizedThroughRound: number;
}

export interface RuntimeRecoveryAuthorization {
  recoveryRef: string;
  sourceTurnRef: string;
  role: LoopRole;
  taskBindingRef: string | null;
  decisionContextRef: string | null;
  timeoutOverride: Partial<TurnTimeoutProfile>;
}

export interface RuntimeRecoveryRecord
  extends RuntimeRecoveryAuthorization {
  formatVersion: RunFormatVersion;
  recoveryId: string;
  tokenHash: string;
  createdAt: string;
  sourceStateRevision: number;
}

export interface TokenUsage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningOutputTokens: number;
  totalTokens: number;
}

export interface RuntimeToolEvent {
  toolName: string;
  status: string;
  arguments: unknown;
  resultSummary: string | null;
  error: unknown;
}

export interface RawTurnResult {
  status: "completed" | "failed" | "timeout" | "interrupted";
  text: string;
  providerThreadId: string | null;
  providerTurnId: string | null;
  usage: TokenUsage;
  toolEvents: RuntimeToolEvent[];
  rawEvents: unknown[];
  compacted: boolean;
  outputCapture: OutputCaptureState;
  partialText: string;
  failureKind: RuntimeFailureKind | null;
  interruptError: string | null;
  lastActivityAt: string;
  incrementalEventsPersisted: boolean;
  error: string | null;
  elapsedMs: number;
}

export type RuntimePersistenceEvent =
  | {
      type: "provider_started";
      at: string;
      threadId: string;
      providerTurnId: string;
    }
  | { type: "raw_event"; at: string; event: unknown }
  | {
      type: "output_delta";
      at: string;
      itemId: string;
      delta: string;
    }
  | {
      type: "message_completed";
      at: string;
      itemId: string;
      phase: "commentary" | "final_answer" | null;
      text: string;
    }
  | {
      type: "tool";
      at: string;
      phase: "started" | "completed";
      event: RuntimeToolEvent;
    }
  | { type: "usage"; at: string; usage: TokenUsage }
  | { type: "compacted"; at: string }
  | {
      type: "timeout";
      at: string;
      kind: Extract<RuntimeFailureKind, "IDLE_TIMEOUT" | "HARD_TIMEOUT">;
      capture: OutputCaptureState;
      lastActivityAt: string;
    }
  | {
      type: "interrupt";
      at: string;
      completed: boolean;
      error: string | null;
    };

export interface TurnDispatch {
  turnId: string;
  role: RuntimeRole;
  prompt: string;
  outputSchema: Record<string, unknown> | null;
  cwd: string;
  model: string;
  effort: "high" | "max";
  timeoutProfile: TurnTimeoutProfile;
  maxInputTokens: number;
  maxOutputTokens: number;
  /** Sibling controllers may supply role-specific provider instructions. */
  developerInstructions?: string;
  onRuntimeEvent?: (event: RuntimePersistenceEvent) => void;
}

export const EXPERIMENT_GOAL_STATUSES = [
  "active",
  "paused",
  "blocked",
  "usageLimited",
  "budgetLimited",
  "complete",
] as const;
export type ExperimentGoalStatus =
  (typeof EXPERIMENT_GOAL_STATUSES)[number];

export interface ExperimentGoalTask {
  experimentId: string;
  goalRef: "workflow_goal.json";
  sourceDecisionTurnRef: string;
  sourceDecisionContextRef: string;
  anchorWork: string;
  directionWork: string | null;
  experimentObjective: string;
  workspaceRef: string;
}

export interface ExperimentGoalRecord {
  formatVersion: RunFormatVersion;
  experimentId: string;
  round: number;
  taskRef: string;
  promptRef: string;
  runtimeRef: string;
  resultRef: string;
  providerThreadId: string | null;
  providerTurnIds: string[];
  goalStatus: ExperimentGoalStatus | "pending" | "runtimeFailed";
  startedAt: string | null;
  completedAt: string | null;
  finalOutputRef: string | null;
  integrationTaskRef: string | null;
  /** Reviewer task/result that semantically integrated this EXP, when known. */
  reviewTaskRef?: string | null;
  reviewRef?: string | null;
  anchorReviewTaskRef?: string | null;
  anchorReviewRef?: string | null;
  error: string | null;
}

export interface ExperimentGoalResult {
  experimentId: string;
  anchorWork: string;
  directionWork: string | null;
  experimentObjective: string;
  goalStatus: Exclude<ExperimentGoalStatus, "active"> | "runtimeFailed";
  conclusionRef: string | null;
  workspaceRef: string;
  providerThreadId: string | null;
  providerTurnIds: string[];
  tokensUsed: number;
  timeUsedSeconds: number;
  error: string | null;
}

export interface GoalDispatch {
  experimentId: string;
  /** Defaults to EXP_GOAL for the Learning Flow. */
  role?: "EXP_GOAL" | "DIRECTION_LAB_GOAL";
  prompt: string;
  objective: string;
  cwd: string;
  model: string;
  effort: "high";
  /** EXP Goals are deliberately unbounded by tokens; timeout controls remain. */
  tokenBudget: null;
  timeoutProfile: TurnTimeoutProfile;
  resumeThreadId: string | null;
  /** Sibling controllers may supply role-specific provider instructions. */
  developerInstructions?: string;
  onRuntimeEvent?: (event: GoalRuntimePersistenceEvent) => void;
}

export interface RawGoalResult {
  goalStatus: Exclude<ExperimentGoalStatus, "active"> | "runtimeFailed";
  finalText: string;
  providerThreadId: string | null;
  providerTurnIds: string[];
  tokensUsed: number;
  timeUsedSeconds: number;
  failureKind:
    | "IDLE_TIMEOUT"
    | "HARD_TIMEOUT"
    | "OPERATOR_INTERRUPT"
    | "PROVIDER_ERROR"
    | null;
  error: string | null;
}

export type GoalRuntimePersistenceEvent =
  | { type: "goal_provider_started"; at: string; threadId: string }
  | { type: "goal_status"; at: string; status: ExperimentGoalStatus }
  | { type: "goal_raw_event"; at: string; event: unknown }
  | {
      type: "goal_turn_started";
      at: string;
      threadId: string;
      providerTurnId: string;
    }
  | {
      type: "goal_message_completed";
      at: string;
      providerTurnId: string;
      itemId: string;
      phase: "commentary" | "final_answer" | null;
      text: string;
    }
  | {
      type: "goal_tool";
      at: string;
      phase: "started" | "completed";
      providerTurnId: string;
      event: RuntimeToolEvent;
    };

export interface TurnRuntime {
  run(dispatch: TurnDispatch): Promise<RawTurnResult>;
  runGoal?(dispatch: GoalDispatch): Promise<RawGoalResult>;
  interruptGoal?(reason?: string): Promise<void>;
  close?(): Promise<void>;
}

export interface BranchEffect {
  decision: LoopDecision;
  nextRole: LoopRole | null;
  nextAction: TaskAction | "RUN_EXPERIMENT" | "FINALIZE" | null;
  targetRef: string | null;
  sequence: LoopRole[];
}

export interface ProgressTrajectoryRecord {
  kind: "DECISION_CYCLE" | "TERMINAL_RUNTIME_FAILURE";
  round: number;
  decisionTurnRef: string | null;
  action: TaskAction | null;
  workRef: string | null;
  workOutcome: WorkOutcome | null;
  reviewRef: string | null;
  reviewVerdict: ReviewVerdict | null;
  decision: LoopDecision | null;
  accepted: { anchors: number; directions: number };
  remainingRequirements: string[];
  retries: { outputCorrection: number; semantic: number; runtime: number };
  usage: TokenUsage & { elapsedMs: number };
  terminalTurnRef?: string;
  runtimeFailureKind?: RuntimeFailureKind | null;
  outputCapture?: OutputCaptureState;
  partialOutputRef?: string | null;
}

export interface ResearchMemoryEntry {
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  workRef: string;
  reviewRef: string;
  workOutcome: WorkOutcome;
  reviewVerdict: ReviewVerdict;
  name: string | null;
  summary: string | null;
  summaryAvailable: boolean;
}

export interface ResearchMemory {
  generatedAt: string;
  sourceStateRevision: number;
  accepted: ResearchMemoryEntry[];
  needsRevision: ResearchMemoryEntry[];
  rejectedLessons: ResearchMemoryEntry[];
  openQueryGaps: Array<{
    objectKind: ObjectKind;
    objectId: string;
    reviewRef: string;
    gap: unknown;
  }>;
  coverage: Record<"L1" | "L2" | "L3" | "L4" | "L5" | "L6", Array<{
    objectId: string;
    workRef: string;
    value: string;
  }>>;
  decisionTrail: Array<{
    turnRef: string;
    decision: LoopDecision;
    guidance: string | null;
  }>;
  experimentResults: Array<{
    resultRef: string;
    anchorWork: string;
    directionWork: string | null;
    goalStatus: ExperimentGoalResult["goalStatus"];
    experimentObjective: string;
    conclusionRef: string | null;
  }>;
  requirements: string[];
}

export interface NegativeExperimentIndexEntry {
  anchorWork: string;
  directionWork: string | null;
  experimentResultRef: string;
  reviewRef: string;
  reviewVerdict: "REJECT";
}

export interface ExperimentCountEntry {
  workRef: string;
  count: number;
}

/**
 * Script-derived navigation only. It deliberately contains no family ID or
 * semantic closure flag; Learning Agents read the referenced evidence.
 */
export interface NegativeExperimentIndex {
  generatedAt: string;
  sourceStateRevision: number;
  counts: {
    run: number;
    anchors: ExperimentCountEntry[];
    directions: ExperimentCountEntry[];
  };
  entries: NegativeExperimentIndexEntry[];
}

export interface DecisionObservation {
  generatedAt: string;
  stateRevision: number;
  round: number;
  researchMemoryRef: string;
  trajectoryRef: string;
  negativeExperimentHistoryRef: string;
  trajectoryTail: ProgressTrajectoryRecord[];
  branchEffects: BranchEffect[];
  accepted: { anchors: number; directions: number };
  remainingRequirements: string[];
  retries: { outputCorrection: number; semantic: number; runtime: number };
  recentRuntimeFailures: Array<{
    turnRef: string;
    role: LoopRole;
    failureKind: RuntimeFailureKind | null;
    outputCapture: OutputCaptureState;
    partialOutputRef: string | null;
  }>;
}
