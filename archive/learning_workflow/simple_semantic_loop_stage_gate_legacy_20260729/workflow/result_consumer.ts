import { randomUUID } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import type {
  RegisteredRole,
  StageContract,
  StateBinding,
  ValidationReport,
  WorkflowLifecycle,
} from "../contracts/index.ts";
import {
  canonicalEqual,
  canonicalJson,
  canonicalSha256,
} from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";
import type { GateEvaluation } from "../stages/gate_engine.ts";

export interface CommitTurnResultInput {
  resultId: string;
  taskId: string;
  attemptId: string;
  stageId: string;
  gateId: string;
  role: Exclude<RegisteredRole, "workflow_decision">;
  messageType: string;
  result: unknown;
  validationReport: ValidationReport;
  gateEvaluation: GateEvaluation;
  rawResponseArtifactId: string;
  committedLifecycle?: WorkflowLifecycle;
}

export function commitTurnResult(
  store: WorkflowStore,
  runId: string,
  expected: StateBinding,
  input: CommitTurnResultInput,
): { resultId: string; nextState: StateBinding; duplicate: boolean } {
  const payloadHash = canonicalSha256(input.result);
  const existing = store.db
    .prepare(
      `SELECT result_id, run_id, stage_id, role, message_type,
              payload_hash
       FROM turn_results WHERE task_id = ?`,
    )
    .get(input.taskId) as
    | {
        result_id: string;
        run_id: string;
        stage_id: string;
        role: string;
        message_type: string;
        payload_hash: string;
      }
    | undefined;
  if (existing) {
    if (
      existing.payload_hash !== payloadHash ||
      existing.run_id !== runId ||
      existing.stage_id !== input.stageId ||
      existing.role !== input.role ||
      existing.message_type !== input.messageType
    ) {
      throw new Error(
        "duplicate result payload conflicts with its committed control binding",
      );
    }
    return {
      resultId: existing.result_id,
      nextState: store.stateBinding(runId),
      duplicate: true,
    };
  }
  if (!input.validationReport.valid || !input.gateEvaluation.passed) {
    throw new Error("only protocol/domain/Gate-passed results may be committed");
  }
  const recoveredFailures = findRecoverableFailedTasks(
    store,
    runId,
    input.taskId,
    input.stageId,
    input.role,
  );
  const nextState = store.casTransition(
    runId,
    expected,
    {
      lifecycle: input.committedLifecycle ?? "running",
      currentStageId: input.stageId,
      eventType: "turn_result_committed",
      eventPayload: {
        resultId: input.resultId,
        taskId: input.taskId,
        stageId: input.stageId,
        role: input.role,
        messageType: input.messageType,
        recoveredTaskIds: recoveredFailures.map((item) => item.taskId),
      },
    },
    (db, nextSnapshotVersion) => {
      const now = new Date().toISOString();
      db.prepare(
        `INSERT INTO turn_results(
           result_id, run_id, task_id, attempt_id, stage_id, role, message_type,
           payload_json, payload_hash, status, committed_snapshot_version,
           committed_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'committed', ?, ?)`,
      ).run(
        input.resultId,
        runId,
        input.taskId,
        input.attemptId,
        input.stageId,
        input.role,
        input.messageType,
        canonicalJson(input.result),
        payloadHash,
        nextSnapshotVersion,
        now,
      );
      db.prepare(
        `INSERT INTO gate_results(
           gate_result_id, run_id, stage_id, gate_id, result_id, passed,
           report_json, created_at
         ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)`,
      ).run(
        `gate-result-${randomUUID()}`,
        runId,
        input.stageId,
        input.gateId,
        input.resultId,
        canonicalJson(input.gateEvaluation),
        now,
      );
      db.prepare(
        "UPDATE tasks SET status = 'committed', updated_at = ? WHERE task_id = ?",
      ).run(now, input.taskId);
      db.prepare(
        `UPDATE attempts
         SET status = 'committed', raw_response_artifact_id = ?,
             error_code = NULL, finished_at = ?
         WHERE attempt_id = ?`,
      ).run(input.rawResponseArtifactId, now, input.attemptId);
      db.prepare(
        `UPDATE validation_reports SET resolved_by_id = ?
         WHERE task_id = ? AND valid = 0 AND resolved_by_id IS NULL`,
      ).run(input.resultId, input.taskId);
      db.prepare(
        `UPDATE workflow_plan_nodes SET lifecycle = 'committed'
         WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
      ).run(runId, expected.workflowPlanRevision, input.stageId);
      applyRecoveredFailures(
        db,
        runId,
        expected.workflowPlanRevision,
        recoveredFailures,
        input.resultId,
        now,
      );
    },
  );
  return { resultId: input.resultId, nextState, duplicate: false };
}

interface RecoverableFailedTask {
  taskId: string;
  stageId: string;
}

export function reconcileCommittedReplacementFailures(
  store: WorkflowStore,
  runId: string,
): number {
  const committed = store.query(
    `SELECT result_id, task_id, stage_id, role
     FROM turn_results
     WHERE run_id = ? AND status = 'committed'
     ORDER BY committed_at, result_id`,
    runId,
  );
  const byTask = new Map<
    string,
    RecoverableFailedTask & { resolutionId: string }
  >();
  for (const row of committed) {
    const role = String(row.role) as Exclude<
      RegisteredRole,
      "workflow_decision"
    >;
    for (const recovered of findRecoverableFailedTasks(
      store,
      runId,
      String(row.task_id),
      String(row.stage_id),
      role,
    )) {
      byTask.set(recovered.taskId, {
        ...recovered,
        resolutionId: String(row.result_id),
      });
    }
  }
  if (byTask.size === 0) return 0;
  const expected = store.stateBinding(runId);
  const recoveries = [...byTask.values()].sort((left, right) =>
    left.taskId.localeCompare(right.taskId),
  );
  store.casTransition(
    runId,
    expected,
    {
      eventType: "replacement_failures_reconciled",
      eventPayload: {
        recoveries: recoveries.map((item) => ({
          taskId: item.taskId,
          stageId: item.stageId,
          resolutionId: item.resolutionId,
        })),
      },
    },
    (db) => {
      const now = new Date().toISOString();
      for (const recovery of recoveries) {
        applyRecoveredFailures(
          db,
          runId,
          expected.workflowPlanRevision,
          [recovery],
          recovery.resolutionId,
          now,
        );
      }
    },
  );
  return recoveries.length;
}

function findRecoverableFailedTasks(
  store: WorkflowStore,
  runId: string,
  successfulTaskId: string,
  successfulStageId: string,
  role: Exclude<RegisteredRole, "workflow_decision">,
): RecoverableFailedTask[] {
  const successful = loadLatestStageContract(
    store,
    runId,
    successfulStageId,
  );
  if (!successful || successful.scope.length === 0) return [];
  return store
    .query(
      `SELECT t.task_id, t.stage_id, sc.contract_json
       FROM tasks t
       JOIN stage_contracts sc
         ON sc.run_id = t.run_id AND sc.stage_id = t.stage_id
       WHERE t.run_id = ? AND t.status = 'failed' AND t.role = ?
         AND t.task_id <> ?
         AND sc.revision = (
           SELECT MAX(sc2.revision) FROM stage_contracts sc2
           WHERE sc2.run_id = sc.run_id AND sc2.stage_id = sc.stage_id
         )
       ORDER BY t.created_at, t.task_id`,
      runId,
      role,
      successfulTaskId,
    )
    .flatMap((row) => {
      const candidate = JSON.parse(
        String(row.contract_json),
      ) as StageContract;
      return candidate.stageType === successful.stageType &&
        candidate.role === successful.role &&
        candidate.definedAtSnapshotVersion <=
          successful.definedAtSnapshotVersion &&
        canonicalEqual(candidate.scope, successful.scope)
        ? [
            {
              taskId: String(row.task_id),
              stageId: String(row.stage_id),
            },
          ]
        : [];
    });
}

function loadLatestStageContract(
  store: WorkflowStore,
  runId: string,
  stageId: string,
): StageContract | null {
  const row = store.db
    .prepare(
      `SELECT contract_json FROM stage_contracts
       WHERE run_id = ? AND stage_id = ?
       ORDER BY revision DESC LIMIT 1`,
    )
    .get(runId, stageId) as { contract_json: string } | undefined;
  return row ? (JSON.parse(row.contract_json) as StageContract) : null;
}

function applyRecoveredFailures(
  db: DatabaseSync,
  runId: string,
  planRevision: number,
  recoveries: RecoverableFailedTask[],
  resolutionId: string,
  now: string,
): void {
  for (const recovery of recoveries) {
    db.prepare(
      `UPDATE tasks SET status = 'superseded', updated_at = ?
       WHERE run_id = ? AND task_id = ? AND status = 'failed'`,
    ).run(now, runId, recovery.taskId);
    db.prepare(
      `UPDATE validation_reports SET resolved_by_id = ?
       WHERE run_id = ? AND task_id = ? AND valid = 0
         AND resolved_by_id IS NULL`,
    ).run(resolutionId, runId, recovery.taskId);
    db.prepare(
      `UPDATE workflow_plan_nodes SET lifecycle = 'superseded'
       WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
         AND lifecycle = 'failed'`,
    ).run(runId, planRevision, recovery.stageId);
  }
}

export interface ConsumeResultInput {
  resultId: string;
  deltaId: string | null;
  canonicalObject:
    | {
        objectType: string;
        objectId: string;
        revision: number;
        value: unknown;
      }
    | null;
}

export function consumeResult(
  store: WorkflowStore,
  runId: string,
  expected: StateBinding,
  input: ConsumeResultInput,
): { commitId: string; nextState: StateBinding; duplicate: boolean } {
  const existing = store.db
    .prepare(
      `SELECT commit_id, run_id, delta_id
       FROM result_consumptions WHERE result_id = ?`,
    )
    .get(input.resultId) as
    | { commit_id: string; run_id: string; delta_id: string | null }
    | undefined;
  if (existing) {
    if (
      existing.run_id !== runId ||
      existing.delta_id !== input.deltaId ||
      !duplicateCanonicalConsumptionMatches(store, runId, input)
    ) {
      throw new Error(
        "duplicate result consumption conflicts with the committed integration",
      );
    }
    return {
      commitId: existing.commit_id,
      nextState: store.stateBinding(runId),
      duplicate: true,
    };
  }
  const result = store.db
    .prepare(
      "SELECT stage_id, status FROM turn_results WHERE run_id = ? AND result_id = ?",
    )
    .get(runId, input.resultId) as
    | { stage_id: string; status: string }
    | undefined;
  if (!result || result.status !== "committed") {
    throw new Error("result is not committed and consumable");
  }
  const commitId = `consume-${randomUUID()}`;
  const nextState = store.casTransition(
    runId,
    expected,
    {
      canonicalRevisionDelta: input.canonicalObject ? 1 : 0,
      lifecycle: "running",
      currentStageId: result.stage_id,
      eventType: "turn_result_consumed",
      eventPayload: {
        resultId: input.resultId,
        commitId,
        deltaId: input.deltaId,
        canonicalObject: input.canonicalObject
          ? {
              objectType: input.canonicalObject.objectType,
              objectId: input.canonicalObject.objectId,
              revision: input.canonicalObject.revision,
            }
          : null,
      },
    },
    (db, nextSnapshotVersion) => {
      if (input.canonicalObject) {
        store.saveCanonicalObject(
          db,
          runId,
          input.canonicalObject.objectType,
          input.canonicalObject.objectId,
          input.canonicalObject.revision,
          input.canonicalObject.value,
          input.resultId,
        );
      }
      db.prepare(
        `INSERT INTO result_consumptions(
           result_id, run_id, commit_id, delta_id, consumed_snapshot_version,
           consumed_at
         ) VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(
        input.resultId,
        runId,
        commitId,
        input.deltaId,
        nextSnapshotVersion,
        new Date().toISOString(),
      );
      db.prepare(
        `UPDATE workflow_plan_nodes SET lifecycle = 'consumed'
         WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
      ).run(runId, expected.workflowPlanRevision, result.stage_id);
    },
  );
  return { commitId, nextState, duplicate: false };
}

function duplicateCanonicalConsumptionMatches(
  store: WorkflowStore,
  runId: string,
  input: ConsumeResultInput,
): boolean {
  const rows = store.query(
    `SELECT object_type, object_id, revision, object_hash
     FROM canonical_objects
     WHERE run_id = ? AND source_result_id = ?
     ORDER BY object_type, object_id, revision`,
    runId,
    input.resultId,
  );
  if (input.canonicalObject === null) return rows.length === 0;
  if (rows.length !== 1) return false;
  const row = rows[0]!;
  return (
    row.object_type === input.canonicalObject.objectType &&
    row.object_id === input.canonicalObject.objectId &&
    Number(row.revision) === input.canonicalObject.revision &&
    row.object_hash === canonicalSha256(input.canonicalObject.value)
  );
}
