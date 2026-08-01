import { randomUUID } from "node:crypto";
import type {
  ArtifactRef,
  WorkflowPermissionEnvelope,
  WorkflowTrigger,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";
import type { RegisteredTrigger } from "./trigger_engine.ts";

export interface WorkflowSnapshotBuildInput {
  taskId: string;
  attemptId: string;
  stageId: string;
  stageContractHash: string;
  skill: WorkflowTurnTask["skill"];
  schema: WorkflowTurnTask["schema"];
  permission: WorkflowPermissionEnvelope;
  domainProjection: WorkflowTurnTask["domainProjection"];
  approvedArtifacts: ArtifactRef[];
  trigger: RegisteredTrigger;
  maxRecentEvents?: number;
}

export function buildWorkflowTurnTask(
  store: WorkflowStore,
  runId: string,
  input: WorkflowSnapshotBuildInput,
): WorkflowTurnTask {
  const run = store.getRun(runId);
  const state = store.readWorkflowState(runId);
  const eventLimit = Math.max(0, Math.min(input.maxRecentEvents ?? 12, 50));
  const recentEvents = store.db
    .prepare(
      `SELECT event_cursor, event_type, payload_json
       FROM events WHERE run_id = ?
       ORDER BY event_cursor DESC LIMIT ?`,
    )
    .all(runId, eventLimit)
    .reverse()
    .map((row) => {
      const value = row as {
        event_cursor: number;
        event_type: string;
        payload_json: string;
      };
      return {
        eventCursor: value.event_cursor,
        eventType: value.event_type,
        summary: summarizeEvent(JSON.parse(value.payload_json)),
        objectRefs: [],
      };
    });
  return {
    protocolVersion: 1,
    messageType: "WORKFLOW_TURN_TASK",
    workflowId: run.workflowId,
    runId,
    taskId: input.taskId,
    attemptId: input.attemptId,
    stageId: input.stageId,
    stageContractHash: input.stageContractHash,
    stateSnapshot: store.stateBinding(runId),
    decisionInputHash: "",
    trigger: input.trigger.trigger as WorkflowTrigger,
    immutableObjective: run.objective,
    immutableAcceptanceCriteria: run.acceptanceCriteria,
    objectiveHash: run.objectiveHash,
    acceptanceCriteriaHash: run.acceptanceCriteriaHash,
    lifecycle: run.lifecycle,
    activeFocusRef: state.activeFocusRef,
    domainProjection: input.domainProjection,
    taskIndex: {
      pendingTaskIds: state.pendingTaskIds,
      inFlightTaskIds: state.inFlightTaskIds,
      failedTaskIds: store
        .query(
          `SELECT task_id FROM tasks WHERE run_id = ? AND status = 'failed' ORDER BY task_id`,
          runId,
        )
        .map((row) => String(row.task_id)),
      pendingOutputRetryTaskIds: store
        .query(
          `SELECT task_id FROM tasks WHERE run_id = ? AND status = 'pending_output_retry' ORDER BY task_id`,
          runId,
        )
        .map((row) => String(row.task_id)),
    },
    resultIndex: {
      committedUnconsumedResultRefs: store
        .query(
          `SELECT r.result_id AS result_id FROM turn_results r
           LEFT JOIN result_consumptions c ON c.result_id = r.result_id
           WHERE r.run_id = ? AND r.status = 'committed' AND c.result_id IS NULL
           ORDER BY r.result_id`,
          runId,
        )
        .map((row) => ({
          objectType: "turn_result",
          objectId: String(row.result_id),
          revision: 1,
        })),
      consumedResultRefs: store
        .query(
          `SELECT result_id FROM result_consumptions WHERE run_id = ? ORDER BY result_id`,
          runId,
        )
        .map((row) => ({
          objectType: "turn_result",
          objectId: String(row.result_id),
          revision: 1,
        })),
    },
    relevantPlan: {
      revision: run.workflowPlanRevision,
      stageIds: store
        .query(
          `SELECT stage_id FROM workflow_plan_nodes
           WHERE run_id = ? AND plan_revision = ? ORDER BY stage_id`,
          runId,
          run.workflowPlanRevision,
        )
        .map((row) => String(row.stage_id)),
      dependencyIds: store
        .query(
          `SELECT dependency_id FROM workflow_plan_edges
           WHERE run_id = ? AND plan_revision = ? ORDER BY dependency_id`,
          runId,
          run.workflowPlanRevision,
        )
        .map((row) => String(row.dependency_id)),
    },
    approvedArtifacts: input.approvedArtifacts,
    triggerReport: {
      reportId: `trigger-report-${randomUUID()}`,
      trigger: input.trigger.trigger,
      sourceStageId: run.currentStageId,
      sourceAttemptId: null,
      facts: input.trigger.facts,
      issueCodes: input.trigger.issueCodes,
    },
    recentEvents,
    skill: input.skill,
    schema: input.schema,
    permission: input.permission,
    correctionFeedback: null,
    terminationCondition:
      "Emit exactly one WORKFLOW_DECISION_PROPOSAL JSON value, then terminate.",
  };
}

function summarizeEvent(value: unknown): string {
  if (value && typeof value === "object") {
    const keys = Object.keys(value as object).slice(0, 6);
    return `Committed event data keys: ${keys.join(", ") || "none"}.`;
  }
  return `Committed event data: ${String(value).slice(0, 160)}`;
}
