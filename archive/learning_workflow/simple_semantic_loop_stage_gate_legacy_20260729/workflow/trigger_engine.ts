import type {
  WorkflowState,
  WorkflowTrigger,
} from "../contracts/index.ts";

export interface TriggerSignals {
  hasTopicFrame: boolean;
  committedResultRequiresIntegration: boolean;
  nonEquivalentFrontierCount: number;
  nonEquivalentRunnableStageCount: number;
  gateFailedWithoutRecovery: boolean;
  planExhaustedObjectiveOpen: boolean;
  criticalEvidenceContradiction: boolean;
  noProgressThresholdReached: boolean;
  closureRejected: boolean;
  userDecisionRequired: boolean;
  hasPendingOrInFlight: boolean;
  hasRunnableStage: boolean;
}

export interface RegisteredTrigger {
  trigger: WorkflowTrigger;
  issueCodes: string[];
  facts: string[];
}

export function buildRegisteredTrigger(
  state: WorkflowState,
  signals: TriggerSignals,
): RegisteredTrigger | null {
  if (state.lifecycle === "initialized" && !signals.hasTopicFrame) {
    return trigger("INITIALIZE_TOPIC", "topic_frame_missing");
  }
  if (signals.userDecisionRequired) {
    return trigger("USER_DECISION_REQUIRED", "user_choice_or_authority_missing");
  }
  if (signals.closureRejected) {
    return trigger("CLOSURE_REJECTED", "closure_reviewer_rejected");
  }
  if (signals.gateFailedWithoutRecovery) {
    return trigger(
      "GATE_FAILED_WITHOUT_RECOVERY_RULE",
      "gate_failed_without_unique_mechanical_recovery",
    );
  }
  if (signals.criticalEvidenceContradiction) {
    return trigger("EVIDENCE_CONTRADICTION", "critical_contradiction_committed");
  }
  if (signals.committedResultRequiresIntegration) {
    return trigger(
      "COMMITTED_RESULT_REQUIRES_INTEGRATION",
      "committed_result_unconsumed",
    );
  }
  if (signals.noProgressThresholdReached) {
    return trigger("NO_PROGRESS_THRESHOLD_REACHED", "no_progress_threshold");
  }
  if (signals.nonEquivalentRunnableStageCount > 1) {
    return trigger(
      "MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE",
      "multiple_non_equivalent_runnable_stages",
    );
  }
  if (signals.nonEquivalentFrontierCount > 1) {
    return trigger(
      "FRONTIER_SELECTION_REQUIRED",
      "multiple_non_equivalent_frontiers",
    );
  }
  if (
    !signals.hasTopicFrame &&
    !signals.hasPendingOrInFlight &&
    !signals.hasRunnableStage
  ) {
    return trigger("INITIALIZE_TOPIC", "topic_frame_missing_after_recovery");
  }
  if (signals.planExhaustedObjectiveOpen) {
    return trigger(
      "PLAN_EXHAUSTED_OBJECTIVE_OPEN",
      "plan_exhausted_acceptance_open",
    );
  }
  if (
    !signals.hasRunnableStage &&
    !signals.hasPendingOrInFlight &&
    ![
      "waiting_user",
      "waiting_external",
      "paused_budget",
      "paused_operator",
      "completed",
      "cancelled",
    ].includes(state.lifecycle)
  ) {
    return trigger("NO_RUNNABLE_STAGE", "no_runnable_pending_or_waiting_work");
  }
  return null;
}

function trigger(
  name: WorkflowTrigger,
  issueCode: string,
): RegisteredTrigger {
  return {
    trigger: name,
    issueCodes: [issueCode],
    facts: [`Controller emitted registered trigger ${name}.`],
  };
}
