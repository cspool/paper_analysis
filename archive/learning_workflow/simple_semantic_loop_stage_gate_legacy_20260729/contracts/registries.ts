import type {
  ClosureChecks,
  ClosureFindingCode,
  ClosureFindingType,
  ClosureRecoveryAction,
  RegisteredStageType,
  StageExecutionKind,
  WorkflowDecisionAction,
  WorkflowTrigger,
} from "./index.ts";
import { canonicalSha256 } from "./canonical_json.ts";

export type RegisteredRole =
  | "workflow_decision"
  | "evidence_reader"
  | "direction_reviewer"
  | "closure_reviewer";

export type ReasoningEffort = "high" | "max";

export type RegisteredTurnInputMessageType =
  | "WORKFLOW_TURN_TASK"
  | "EVIDENCE_READER_TASK"
  | "DIRECTION_REVIEW_TASK"
  | "CLOSURE_REVIEW_TASK";

export type RegisteredTurnOutputMessageType =
  | "WORKFLOW_DECISION_PROPOSAL"
  | "EVIDENCE_PACKET"
  | "REVIEW_DELTA"
  | "CLOSURE_REVIEW";

export const ROLE_REASONING_EFFORT = Object.freeze({
  workflow_decision: "max",
  evidence_reader: "high",
  direction_reviewer: "high",
  closure_reviewer: "high",
} as const satisfies Readonly<Record<RegisteredRole, ReasoningEffort>>);

export const ROLE_MESSAGE_TYPES = Object.freeze({
  workflow_decision: {
    input: "WORKFLOW_TURN_TASK",
    output: "WORKFLOW_DECISION_PROPOSAL",
  },
  evidence_reader: {
    input: "EVIDENCE_READER_TASK",
    output: "EVIDENCE_PACKET",
  },
  direction_reviewer: {
    input: "DIRECTION_REVIEW_TASK",
    output: "REVIEW_DELTA",
  },
  closure_reviewer: {
    input: "CLOSURE_REVIEW_TASK",
    output: "CLOSURE_REVIEW",
  },
} as const satisfies Readonly<
  Record<
    RegisteredRole,
    {
      input: RegisteredTurnInputMessageType;
      output: RegisteredTurnOutputMessageType;
    }
  >
>);

export const ROLE_SKILLS = Object.freeze({
  workflow_decision: "learning-semantic-loop-workflow-turn",
  evidence_reader: "learning-semantic-loop-evidence-reader",
  direction_reviewer: "learning-semantic-loop-direction-reviewer",
  closure_reviewer: "learning-semantic-loop-closure-reviewer",
} as const satisfies Readonly<Record<RegisteredRole, string>>);

export type StageCreationAuthority =
  | "workflow_run_stage"
  | "workflow_request_evaluation"
  | "controller_trigger"
  | "controller_closure"
  | "controller_finalization";

export interface StageRegistryEntry {
  executionKind: StageExecutionKind;
  role: RegisteredRole | null;
  output: RegisteredTurnOutputMessageType | null;
  creationAuthority: StageCreationAuthority;
}

export const STAGE_REGISTRY = Object.freeze({
  SCRIPT_APPLY_TOPIC_FRAME: {
    executionKind: "SCRIPT_TRANSITION",
    role: null,
    output: null,
    creationAuthority: "workflow_run_stage",
  },
  SCRIPT_APPLY_SEMANTIC_DELTA: {
    executionKind: "SCRIPT_TRANSITION",
    role: null,
    output: null,
    creationAuthority: "workflow_run_stage",
  },
  WORKFLOW_DECISION: {
    executionKind: "DECISION_TURN",
    role: "workflow_decision",
    output: "WORKFLOW_DECISION_PROPOSAL",
    creationAuthority: "controller_trigger",
  },
  EVIDENCE_READ: {
    executionKind: "WORKER_TURN",
    role: "evidence_reader",
    output: "EVIDENCE_PACKET",
    creationAuthority: "workflow_run_stage",
  },
  DIRECTION_REVIEW: {
    executionKind: "EVALUATOR_TURN",
    role: "direction_reviewer",
    output: "REVIEW_DELTA",
    creationAuthority: "workflow_request_evaluation",
  },
  CLOSURE_REVIEW: {
    executionKind: "EVALUATOR_TURN",
    role: "closure_reviewer",
    output: "CLOSURE_REVIEW",
    creationAuthority: "controller_closure",
  },
  RENDER_FINAL: {
    executionKind: "SCRIPT_TRANSITION",
    role: null,
    output: null,
    creationAuthority: "controller_finalization",
  },
} as const satisfies Readonly<Record<RegisteredStageType, StageRegistryEntry>>);

export const TRIGGER_ALLOWED_ACTIONS = Object.freeze({
  INITIALIZE_TOPIC: ["RUN_STAGE"],
  COMMITTED_RESULT_REQUIRES_INTEGRATION: [
    "RUN_STAGE",
    "REQUEST_EVALUATION",
  ],
  FRONTIER_SELECTION_REQUIRED: [
    "RUN_STAGE",
    "REQUEST_EVALUATION",
    "PROPOSE_COMPLETE",
  ],
  MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE: ["RUN_STAGE", "REPLAN"],
  GATE_FAILED_WITHOUT_RECOVERY_RULE: [
    "RETRY_STAGE",
    "REPLAN",
    "REQUEST_EVALUATION",
    "ASK_USER",
    "REPORT_BLOCKED",
  ],
  PLAN_EXHAUSTED_OBJECTIVE_OPEN: [
    "RUN_STAGE",
    "REQUEST_EVALUATION",
    "REPLAN",
    "PROPOSE_COMPLETE",
    "ASK_USER",
    "REPORT_BLOCKED",
  ],
  EVIDENCE_CONTRADICTION: ["RUN_STAGE", "REQUEST_EVALUATION", "REPLAN"],
  NO_PROGRESS_THRESHOLD_REACHED: [
    "REPLAN",
    "ASK_USER",
    "REPORT_BLOCKED",
    "PROPOSE_PAUSE",
  ],
  CLOSURE_REJECTED: ["RUN_STAGE", "REPLAN", "ASK_USER"],
  NO_RUNNABLE_STAGE: [
    "RUN_STAGE",
    "REQUEST_EVALUATION",
    "REPLAN",
    "ASK_USER",
    "REPORT_BLOCKED",
    "PROPOSE_PAUSE",
    "PROPOSE_COMPLETE",
  ],
  USER_DECISION_REQUIRED: ["ASK_USER"],
} as const satisfies Readonly<
  Record<WorkflowTrigger, readonly WorkflowDecisionAction[]>
>);

export const CLOSURE_CHECK_NAMES = Object.freeze([
  "stopProofRevisionCurrent",
  "stopProofMatchesCanonical",
  "mechanicalPreflightPassed",
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
] as const satisfies readonly (keyof ClosureChecks)[]);

export const FINALIZATION_REQUIREMENTS = Object.freeze([
  "canonical_revision_unchanged",
  "full_validator_passed",
  "final_output_rendered",
  "final_output_coverage_validated",
  "atomic_completed_commit",
] as const);

type ClosureFindingRule = {
  check: keyof ClosureChecks;
  type: ClosureFindingType;
  recoveryAction: ClosureRecoveryAction;
};

export const CLOSURE_FINDING_REGISTRY = Object.freeze({
  stale_stop_proof_revision: rule(
    "stopProofRevisionCurrent",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  stop_proof_canonical_mismatch: rule(
    "stopProofMatchesCanonical",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  mechanical_preflight_failed: rule(
    "mechanicalPreflightPassed",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  topic_scope_silently_narrowed: rule(
    "topicScopePreserved",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  knowledge_answerable_open_need: rule(
    "noKnowledgeAnswerableCriticalNeed",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  anchor_not_closed: rule(
    "allAnchorsClosed",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  anchor_missing_saturation_reason: rule(
    "allAnchorsClosed",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  anchor_missing_status_reason: rule(
    "allAnchorsClosed",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  direction_nonterminal: rule(
    "allDirectionsTerminal",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  direction_missing_terminal_reason: rule(
    "allDirectionsTerminal",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  last_topic_expansion_missing: rule(
    "lastTopicExpansionNoDelta",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  last_topic_expansion_not_quiet: rule(
    "lastTopicExpansionNoDelta",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  pending_task: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  in_flight_task: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  pending_output_retry: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  unconsumed_result: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  uncommitted_delta: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  unresolved_validation_failure: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  failed_task: rule(
    "noUnconsumedOrUncommittedWork",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  unreviewed_critical_contradiction: rule(
    "criticalContradictionsReviewed",
    "knowledge_gap",
    "REOPEN_FRONTIER",
  ),
  experiment_handoff_missing: rule(
    "experimentHandoffsComplete",
    "incomplete_handoff",
    "COMPLETE_HANDOFF",
  ),
  experiment_handoff_invalid: rule(
    "experimentHandoffsComplete",
    "incomplete_handoff",
    "COMPLETE_HANDOFF",
  ),
  runtime_budget_exhausted: rule(
    "runtimeEligibleForCompletion",
    "runtime_pause",
    "RESUME_RUNTIME",
  ),
  runtime_failed_or_paused: rule(
    "runtimeEligibleForCompletion",
    "runtime_pause",
    "RESUME_RUNTIME",
  ),
  final_output_missing_field: rule(
    "finalOutputTraceable",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
  final_output_untraceable: rule(
    "finalOutputTraceable",
    "state_inconsistency",
    "REPAIR_STATE",
  ),
} as const satisfies Readonly<Record<ClosureFindingCode, ClosureFindingRule>>);

function rule(
  check: keyof ClosureChecks,
  type: ClosureFindingType,
  recoveryAction: ClosureRecoveryAction,
): ClosureFindingRule {
  return { check, type, recoveryAction };
}

export const MAX_OUTPUT_ATTEMPTS_PER_TASK = 3;
export const MAX_PROVIDER_FAILURES_PER_TASK = 2;
export const MAX_TOTAL_ATTEMPTS_PER_TASK = 4;
export const MAX_WORKFLOW_PLAN_REVISIONS = 100;

export const RUBRIC_DEFINITIONS = Object.freeze({
  direction_readiness_v1: Object.freeze({
    rubricId: "direction-readiness-v1",
    version: "1.0.0",
    orderedChecks: Object.freeze([
      "scope",
      "baseline",
      "minimum_change",
      "causal_chain",
      "implementation",
      "measurement",
      "counterexample",
      "evidence_traceability",
      "knowledge_answerability",
      "decision",
    ]),
  }),
  closure_v1: Object.freeze({
    rubricId: "closure-rubric-v1",
    version: "1.0.0",
    orderedChecks: CLOSURE_CHECK_NAMES,
  }),
});

export const RUBRIC_REGISTRY = Object.freeze({
  direction_readiness_v1: Object.freeze({
    rubricId: RUBRIC_DEFINITIONS.direction_readiness_v1.rubricId,
    version: RUBRIC_DEFINITIONS.direction_readiness_v1.version,
    sha256: canonicalSha256(RUBRIC_DEFINITIONS.direction_readiness_v1),
  }),
  closure_v1: Object.freeze({
    rubricId: RUBRIC_DEFINITIONS.closure_v1.rubricId,
    version: RUBRIC_DEFINITIONS.closure_v1.version,
    sha256: canonicalSha256(RUBRIC_DEFINITIONS.closure_v1),
  }),
} as const);
