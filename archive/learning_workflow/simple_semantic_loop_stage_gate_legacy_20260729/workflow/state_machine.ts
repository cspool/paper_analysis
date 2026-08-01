import type {
  StateBinding,
  WorkflowLifecycle,
} from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";

const TERMINAL = new Set<WorkflowLifecycle>(["completed", "cancelled"]);

export const LIFECYCLE_TRANSITIONS: Readonly<
  Record<WorkflowLifecycle, readonly WorkflowLifecycle[]>
> = Object.freeze({
  initialized: [
    "running",
    "waiting_turn",
    "paused_budget",
    "paused_operator",
    "failed_retriable",
    "failed_terminal",
    "cancelled",
  ],
  running: [
    "waiting_turn",
    "waiting_user",
    "waiting_external",
    "closure_preflight",
    "paused_budget",
    "paused_operator",
    "failed_retriable",
    "failed_terminal",
    "blocked_semantic",
    "blocked_external",
    "cancelled",
  ],
  waiting_turn: [
    "running",
    "failed_retriable",
    "failed_terminal",
    "paused_operator",
    "cancelled",
  ],
  waiting_user: ["running", "paused_operator", "cancelled"],
  waiting_external: ["running", "blocked_external", "paused_operator", "cancelled"],
  closure_preflight: [
    "waiting_closure_review",
    "running",
    "failed_retriable",
    "failed_terminal",
    "paused_operator",
    "cancelled",
  ],
  waiting_closure_review: [
    "finalizing",
    "running",
    "failed_retriable",
    "failed_terminal",
    "cancelled",
  ],
  finalizing: ["completed", "running", "failed_terminal", "cancelled"],
  paused_budget: ["running", "cancelled"],
  paused_operator: ["running", "cancelled"],
  failed_retriable: ["running", "cancelled", "failed_terminal"],
  failed_terminal: [],
  blocked_semantic: ["running", "cancelled"],
  blocked_external: ["running", "cancelled"],
  completed: [],
  cancelled: [],
});

export function canTransition(
  from: WorkflowLifecycle,
  to: WorkflowLifecycle,
): boolean {
  if (from === to) {
    return !["completed", "cancelled", "failed_terminal"].includes(from);
  }
  return LIFECYCLE_TRANSITIONS[from].includes(to);
}

export function assertLifecycleTransition(
  from: WorkflowLifecycle,
  to: WorkflowLifecycle,
): void {
  if (!canTransition(from, to)) {
    throw new Error(`illegal workflow lifecycle transition ${from} -> ${to}`);
  }
}

export function transitionLifecycle(
  store: WorkflowStore,
  runId: string,
  expected: StateBinding,
  to: WorkflowLifecycle,
  reason: string | null,
  eventType = "lifecycle_transition",
): StateBinding {
  const from = store.getRun(runId).lifecycle;
  assertLifecycleTransition(from, to);
  return store.casTransition(runId, expected, {
    lifecycle: to,
    pauseOrBlockReason: reason,
    eventType,
    eventPayload: { from, to, reason },
  });
}

export function isQuiescent(lifecycle: WorkflowLifecycle): boolean {
  return [
    "waiting_user",
    "waiting_external",
    "paused_budget",
    "paused_operator",
    "failed_retriable",
    "failed_terminal",
    "blocked_semantic",
    "blocked_external",
    "completed",
    "cancelled",
  ].includes(lifecycle);
}

export function isTerminal(lifecycle: WorkflowLifecycle): boolean {
  return TERMINAL.has(lifecycle);
}
