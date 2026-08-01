import { MAX_PROVIDER_FAILURES_PER_TASK } from "../contracts/index.ts";
import type { ClassifiedFailure } from "./failure_classifier.ts";

export type RecoveryDecision =
  | { action: "fresh_same_role_attempt"; nextAttemptNo: number }
  | { action: "transient_provider_retry" }
  | { action: "reconcile_stale_state" }
  | { action: "trigger_workflow_decision" }
  | { action: "fail_terminal" }
  | { action: "fail_retriable" };

export function deterministicRecovery(
  failure: ClassifiedFailure,
  attemptNo: number,
): RecoveryDecision {
  if (
    failure.sameRoleOutputRetry &&
    attemptNo < MAX_PROVIDER_FAILURES_PER_TASK
  ) {
    return { action: "fresh_same_role_attempt", nextAttemptNo: attemptNo + 1 };
  }
  if (failure.class === "transient_provider") {
    return { action: "transient_provider_retry" };
  }
  if (failure.class === "stale_state") {
    return { action: "reconcile_stale_state" };
  }
  if (failure.workflowSemanticTrigger) {
    return { action: "trigger_workflow_decision" };
  }
  return failure.terminal
    ? { action: "fail_terminal" }
    : { action: "fail_retriable" };
}
