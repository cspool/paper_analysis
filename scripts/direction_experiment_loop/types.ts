import type {
  ExperimentGoalStatus,
  RawTurnResult,
  TurnTimeoutProfile,
} from "../simple_semantic_loop/refactor/types.ts";

export const DIRECTION_EXPERIMENT_FORMAT_VERSION = 7 as const;
export const READABLE_DIRECTION_EXPERIMENT_FORMAT_VERSIONS = [2, 3, 4, 5, 6, 7] as const;

export const EXPERIMENT_DECISIONS = [
  "RUN_LAB",
  "RUN_JUDGE",
  "COMPLETE_SUPPORT",
  "COMPLETE_REJECT",
  "RETURN_TO_LEARNING",
  "BLOCKED",
] as const;
export type ExperimentDecision = (typeof EXPERIMENT_DECISIONS)[number];
export type TerminalExperimentDecision = Extract<
  ExperimentDecision,
  "COMPLETE_SUPPORT" | "COMPLETE_REJECT" | "RETURN_TO_LEARNING"
>;

export const JUDGE_ASSESSMENTS = [
  "VALID_POSITIVE",
  "VALID_NEGATIVE",
  "INCONCLUSIVE",
  "INVALID",
] as const;
export type JudgeAssessment = (typeof JUDGE_ASSESSMENTS)[number];

export const EVIDENCE_SCOPES = [
  "DESIGN_AUDIT_ONLY",
  "WEAKENED_PROXY_MECHANISM",
  "LOCAL_SINGLE_GPU_PERFORMANCE",
  "SIMULATED_HARDWARE_MECHANISM",
  "PAPER_EXTERNAL_VALIDITY",
] as const;
export type EvidenceScope = (typeof EVIDENCE_SCOPES)[number];

export type DirectionLifecycle = "RUNNING" | "PAUSED" | "FINISHED" | "FAILED";
export type DirectionNode = "DECISION" | "LAB_GOAL" | "JUDGE" | null;
export type DirectionPauseKind =
  | "CYCLE_BUDGET_EXHAUSTED"
  | "LAB_GOAL_PAUSED"
  | "DECISION_RETRY_EXHAUSTED"
  | "JUDGE_RETRY_EXHAUSTED"
  | "DECISION_BLOCKED"
  | "OPERATOR_REQUESTED";

export type GoalInterruptionKind =
  | "IDLE_TIMEOUT"
  | "HARD_TIMEOUT"
  | "OPERATOR_INTERRUPT"
  | "PROVIDER_ERROR"
  | null;

export interface FilePin {
  name: string;
  path: string;
  sha256: string;
}

export interface DirectionSourceBinding {
  sourceWorkDir: string;
  sourceRunId: string;
  directionId: string;
  directionRevision: number;
  directionWorkRef: string;
  directionReviewRef: string;
  parentAnchorId: string;
  parentAnchorRevision: number;
  parentAnchorWorkRef: string;
}

export interface DirectionInputPins {
  directionResult: FilePin;
  parentAnchorResult: FilePin;
  sourceReviewResult: FilePin;
  directionTarget: FilePin;
  evidenceManifest: FilePin;
  experimentPolicy: FilePin;
  sourceRun: FilePin;
}

export interface DirectionRunFile {
  formatVersion: typeof DIRECTION_EXPERIMENT_FORMAT_VERSION;
  workflow: "DIRECTION_EXPERIMENT_LOOP";
  runId: string;
  createdAt: string;
  projectRoot: string;
  model: string;
  source: DirectionSourceBinding;
  inputs: DirectionInputPins;
  /** Run-local immutable copies. Active global Skills are never run authority. */
  skills: {
    decision: FilePin;
    decisionMethod: FilePin;
    lab: FilePin;
    labMethod: FilePin;
    judge: FilePin;
    judgeContract: FilePin;
  };
  storage: {
    sharedCacheRoot: string;
    cycleRoot: "workspace/cycles";
  };
  budgets: {
    initialLabCycles: number;
    maxOutputRetries: number;
    maxRuntimeRetries: number;
    decision: TurnTimeoutProfile;
    lab: TurnTimeoutProfile;
    judge: TurnTimeoutProfile;
    labResultReserveMs: number;
    controlPollMs: number;
    maxInputTokens: number;
    maxOutputTokens: number;
  };
}

export interface DirectionStateFile {
  formatVersion: typeof DIRECTION_EXPERIMENT_FORMAT_VERSION;
  revision: number;
  lifecycle: DirectionLifecycle;
  node: DirectionNode;
  cycle: number;
  authorizedLabCycles: number;
  transitions: number;
  reason: string | null;
  pauseKind: DirectionPauseKind | null;
  activeContractRevision: number;
  activeContractRef: string | null;
  activeContractHash: string | null;
  activeGoalRecordRef: string | null;
  activeLabInvocationRef: string | null;
  activeJudgeRequestRef: string | null;
  latestLabResultRef: string | null;
  latestCheckpointRef: string | null;
  latestJudgeRef: string | null;
  latestDecisionRef: string | null;
  finalDecision: TerminalExperimentDecision | null;
  evidenceScope: EvidenceScope | null;
}

export interface EvidenceManifestItem {
  owner: "ANCHOR" | "DIRECTION";
  sourceRef: string;
  supports: string;
  resolvedPath: string | null;
  sourceUnit: string | null;
  sha256: string | null;
  resolution: "RESOLVED" | "UNRESOLVED";
}

export interface ExperimentContractProposal {
  objective: string;
  comparison: string;
  conditions: string;
  stopConditions: string[];
  estimatedMinutes: number;
  allowedWeakening: string[];
  forbiddenWeakening: string[];
  completionEvidence: string;
}

export interface ExperimentContractRecord extends ExperimentContractProposal {
  formatVersion: 2;
  contractRevision: number;
  directionId: string;
  directionRevision: number;
  sourceDirectionRef: string;
  decisionRef: string;
  targetEvidenceScope: EvidenceScope;
  createdAt: string;
}

export interface ExperimentDecisionResult {
  decision: ExperimentDecision;
  evidenceScope: EvidenceScope;
  reason: string;
  experimentContract: ExperimentContractProposal | null;
  reviewFocus: string | null;
}

export interface JudgeResult {
  assessment: JudgeAssessment;
  evidenceScope: EvidenceScope;
  reason: string;
  remainingUncertainty: string;
}

export interface LabRuntimeEnvelope {
  idleTimeoutMs: number;
  hardTimeoutMs: number;
  resultReserveMs: number;
  maxContractMinutes: number;
  currentCycle: number;
  remainingAuthorizedCycles: number;
  activeGoalTimeUsedSeconds: number;
  priorLabTimeUsedSeconds: number;
  latestCheckpointRef: string | null;
  latestLabResultRef: string | null;
}

export interface FreshTurnRecord {
  turnId: string;
  role: "EXPERIMENT_DECISION" | "EVIDENCE_JUDGE";
  ordinal: number;
  attempt: number;
  startedAt: string;
  completedAt: string | null;
  promptRef: string;
  stateSnapshotRef: string;
  historySnapshotRef: string;
  runtimeEnvelopeRef: string | null;
  rawOutputRef: string | null;
  resultRef: string | null;
  errorRef: string | null;
  runtimeRef: string;
  providerThreadId: string | null;
  providerTurnId: string | null;
  providerStatus: RawTurnResult["status"] | null;
}

export interface LabCycleBinding {
  formatVersion: 1;
  cycle: number;
  goalId: string;
  contractRevision: number;
  contractRef: string;
  contractHash: string;
  sourceDecisionRef: string;
  resultRef: string;
  checkpointRef: string;
  cycleSourceRef: string;
  createdAt: string;
}

export interface DirectionGoalRecord {
  goalId: string;
  cycle: number;
  contractRevision: number;
  contractRef: string;
  contractHash: string;
  sourceDecisionRef: string;
  workspaceRef: string;
  cycleRef: string;
  cycleSourceRef: string;
  outputRef: string;
  checkpointRef: string;
  bindingRef: string;
  bindingHash: string;
  providerThreadId: string | null;
  providerTurnIds: string[];
  goalStatus: ExperimentGoalStatus | "pending" | "runtimeFailed";
  invocationRefs: string[];
  activeInvocationRef: string | null;
  startedAt: string | null;
  completedAt: string | null;
  tokensUsed: number;
  timeUsedSeconds: number;
  error: string | null;
}

export interface LabGoalInvocationRecord {
  invocationId: string;
  ordinal: number;
  goalRef: string;
  cycle: number;
  contractRevision: number;
  contractHash: string;
  resumed: boolean;
  resumeThreadId: string | null;
  startedAt: string;
  deadlineAt: string;
  completedAt: string | null;
  promptRef: string;
  runtimeRef: string;
  providerRawRef: string;
  providerFinalRef: string | null;
  providerThreadId: string | null;
  providerTurnIds: string[];
  providerStatus: DirectionGoalRecord["goalStatus"] | null;
  interruptionKind: GoalInterruptionKind;
  operatorPauseRequested: boolean;
  checkpointRef: string | null;
  resultRef: string | null;
  tokensUsed: number;
  timeUsedSeconds: number;
  error: string | null;
}

export interface LabCheckpoint {
  cycle: number;
  contractRevision: number;
  contractHash: string;
  phase: string;
  completedUnits: string[];
  validatedArtifacts: string[];
  lastProgressAt: string;
  activeCommand: string | null;
  resumeAction: string;
  partialExcludedRefs: string[];
}

export interface DecisionHistoryEntry {
  kind: "DECISION";
  ordinal: number;
  decisionRef: string;
  decision: ExperimentDecision;
  evidenceScope: EvidenceScope;
  reason: string;
  contractRevision: number | null;
  contractRef: string | null;
  contractHash: string | null;
  reviewFocusRef: string | null;
  completedAt: string;
}

export interface ExperimentHistoryEntry {
  kind: "EXPERIMENT";
  cycle: number;
  contractRevision: number;
  contractRef: string;
  contractHash: string;
  sourceDecisionRef: string;
  goalRecordRef: string;
  invocationRefs: string[];
  providerStatus: DirectionGoalRecord["goalStatus"];
  interruptionKind: GoalInterruptionKind;
  resultRef: string;
  checkpointRef: string | null;
  adoptedAfterInterruption: boolean;
  tokensUsed: number;
  timeUsedSeconds: number;
  completedAt: string;
}

export interface JudgmentHistoryEntry {
  kind: "JUDGMENT";
  ordinal: number;
  contractRevision: number | null;
  contractRef: string | null;
  labResultRef: string | null;
  requestRef: string;
  judgmentRef: string;
  assessment: JudgeAssessment;
  evidenceScope: EvidenceScope;
  reason: string;
  completedAt: string;
}

export type DirectionHistoryEntry =
  | DecisionHistoryEntry
  | ExperimentHistoryEntry
  | JudgmentHistoryEntry;

export interface DirectionLoopOutcome {
  workflowOutcome: "FINISHED" | "PAUSED" | "FAILED";
  reportRef: "final/report.md" | null;
  handoffRef: "final/handoff.json" | null;
  evidenceScope: EvidenceScope | null;
  reason: string | null;
}

export interface DirectionHandoff {
  sourceDirectionRef: "inputs/direction_result.json";
  directionId: string;
  directionRevision: number;
  directionTargetRef: "inputs/direction_target.md";
  sourceEvidenceManifestRef: "inputs/evidence_manifest.json";
  experimentPolicyRef: "inputs/experiment_policy.md";
  outcome: "SUPPORTED" | "NOT_SUPPORTED" | "RETURN_TO_LEARNING";
  finalDecision: TerminalExperimentDecision;
  evidenceScope: EvidenceScope;
  summary: string;
  activeContractRevision: number;
  activeContractRef: string | null;
  activeContractHash: string | null;
  contractRefs: string[];
  experimentResultRefs: string[];
  judgmentRefs: string[];
  finalDecisionRef: string;
  reportRef: "final/report.md";
}
