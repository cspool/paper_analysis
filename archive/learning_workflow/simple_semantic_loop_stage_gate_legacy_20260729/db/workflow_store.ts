import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  ArtifactRef,
  GateDefinition,
  RegisteredRole,
  StageContract,
  StateBinding,
  ValidationReport,
  WorkflowLifecycle,
  WorkflowState,
} from "../contracts/index.ts";
import {
  canonicalJson,
  canonicalSha256,
} from "../contracts/index.ts";
import { DATABASE_SCHEMA_VERSION, migrate } from "./migrations.ts";
import { assertLifecycleTransition } from "../workflow/state_machine.ts";

export interface RunRecord {
  runId: string;
  workflowId: string;
  objective: string;
  objectiveHash: string;
  acceptanceCriteria: string[];
  acceptanceCriteriaHash: string;
  lifecycle: WorkflowLifecycle;
  snapshotVersion: number;
  canonicalRevision: number;
  eventCursor: number;
  workflowPlanRevision: number;
  currentStageId: string | null;
  activeFocusRef: unknown | null;
  config: Record<string, unknown>;
  pauseOrBlockReason: string | null;
  completedAt: string | null;
}

export interface CreateRunInput {
  runId: string;
  workflowId: string;
  objective: string;
  acceptanceCriteria: string[];
  config: Record<string, unknown>;
}

export interface CasTransition {
  lifecycle?: WorkflowLifecycle;
  currentStageId?: string | null;
  activeFocusRef?: unknown | null;
  pauseOrBlockReason?: string | null;
  canonicalRevisionDelta?: 0 | 1;
  workflowPlanRevisionDelta?: 0 | 1;
  eventType: string;
  eventPayload: unknown;
}

export interface TaskRecordInput {
  taskId: string;
  stageId: string;
  role: RegisteredRole;
  inputMessageType: string;
  expectedOutputMessageType: string;
  stateBinding: StateBinding;
  inputHash: string;
  stageContractHash: string;
  skillHash: string;
  schemaManifestHash: string;
  task: unknown;
}

export interface AttemptInput {
  attemptId: string;
  taskId: string;
  attemptNo: number;
  role: RegisteredRole;
  logicalEffort: string;
  providerWireEffort: string;
}

export class CasConflictError extends Error {
  readonly expected: StateBinding;
  readonly actual: StateBinding;

  constructor(
    expected: StateBinding,
    actual: StateBinding,
  ) {
    super(
      `stale StateBinding: expected=${canonicalJson(expected)} actual=${canonicalJson(actual)}`,
    );
    this.name = "CasConflictError";
    this.expected = expected;
    this.actual = actual;
  }
}

export class WorkflowStore {
  readonly dbPath: string;
  readonly db: DatabaseSync;

  constructor(dbPath: string) {
    this.dbPath = resolve(dbPath);
    mkdirSync(dirname(this.dbPath), { recursive: true });
    this.db = new DatabaseSync(this.dbPath);
    migrate(this.db);
  }

  close(): void {
    this.db.close();
  }

  schemaVersion(): number {
    return DATABASE_SCHEMA_VERSION;
  }

  createRun(input: CreateRunInput): RunRecord {
    const now = timestamp();
    const objectiveHash = canonicalSha256(input.objective);
    const acceptanceCriteriaHash = canonicalSha256(input.acceptanceCriteria);
    this.db.exec("BEGIN IMMEDIATE");
    try {
      this.db
        .prepare(
          `INSERT INTO runs(
             run_id, workflow_id, objective, objective_hash,
             acceptance_criteria_json, acceptance_criteria_hash,
             lifecycle, snapshot_version, canonical_revision, event_cursor,
             workflow_plan_revision, current_stage_id, active_focus_ref_json,
             config_json, pause_or_block_reason, completed_at, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, 'initialized', 0, 0, 0, 1, NULL, NULL, ?, NULL, NULL, ?, ?)`,
        )
        .run(
          input.runId,
          input.workflowId,
          input.objective,
          objectiveHash,
          canonicalJson(input.acceptanceCriteria),
          acceptanceCriteriaHash,
          canonicalJson(input.config),
          now,
          now,
        );
      const initialPlan = {
        workflowId: input.workflowId,
        revision: 1,
        objectiveHash,
        acceptanceCriteriaHash,
        stageNodes: [],
        dependencies: [],
        planStatus: "active",
      };
      this.db
        .prepare(
          `INSERT INTO workflow_plans(
             run_id, revision, objective_hash, acceptance_criteria_hash,
             status, plan_json, created_at
           ) VALUES (?, 1, ?, ?, 'active', ?, ?)`,
        )
        .run(
          input.runId,
          objectiveHash,
          acceptanceCriteriaHash,
          canonicalJson(initialPlan),
          now,
        );
      const initialState = this.materializeStateFromRow(
        this.requireRunRow(input.runId),
      );
      this.insertSnapshot(input.runId, 0, initialState, now);
      this.db.exec("COMMIT");
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
    return this.getRun(input.runId);
  }

  getRun(runId: string): RunRecord {
    return mapRunRow(this.requireRunRow(runId));
  }

  stateBinding(runId: string): StateBinding {
    const run = this.getRun(runId);
    return {
      snapshotVersion: run.snapshotVersion,
      canonicalRevision: run.canonicalRevision,
      eventCursor: run.eventCursor,
      workflowPlanRevision: run.workflowPlanRevision,
    };
  }

  readWorkflowState(runId: string): WorkflowState {
    return this.materializeStateFromRow(this.requireRunRow(runId));
  }

  casTransition(
    runId: string,
    expected: StateBinding,
    transition: CasTransition,
    apply?: (db: DatabaseSync, nextSnapshotVersion: number) => void,
  ): StateBinding {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const row = this.requireRunRow(runId);
      const actual = bindingFromRow(row);
      if (!sameBinding(actual, expected)) {
        throw new CasConflictError(expected, actual);
      }
      if (row.lifecycle === "completed") {
        throw new Error("completed run is immutable");
      }
      if (transition.lifecycle !== undefined) {
        assertLifecycleTransition(
          row.lifecycle as WorkflowLifecycle,
          transition.lifecycle,
        );
      }
      const nextSnapshot = actual.snapshotVersion + 1;
      const nextEventCursor = actual.eventCursor + 1;
      const nextCanonical =
        actual.canonicalRevision + (transition.canonicalRevisionDelta ?? 0);
      const nextPlan =
        actual.workflowPlanRevision +
        (transition.workflowPlanRevisionDelta ?? 0);
      apply?.(this.db, nextSnapshot);
      const now = timestamp();
      this.db
        .prepare(
          `INSERT INTO events(
             run_id, event_cursor, snapshot_version, event_type, payload_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          runId,
          nextEventCursor,
          nextSnapshot,
          transition.eventType,
          canonicalJson(transition.eventPayload),
          now,
        );
      this.db
        .prepare(
          `UPDATE runs SET
             lifecycle = ?,
             snapshot_version = ?,
             canonical_revision = ?,
             event_cursor = ?,
             workflow_plan_revision = ?,
             current_stage_id = ?,
             active_focus_ref_json = ?,
             pause_or_block_reason = ?,
             updated_at = ?
           WHERE run_id = ?`,
        )
        .run(
          transition.lifecycle ?? row.lifecycle,
          nextSnapshot,
          nextCanonical,
          nextEventCursor,
          nextPlan,
          transition.currentStageId === undefined
            ? row.current_stage_id
            : transition.currentStageId,
          transition.activeFocusRef === undefined
            ? row.active_focus_ref_json
            : transition.activeFocusRef === null
              ? null
              : canonicalJson(transition.activeFocusRef),
          transition.pauseOrBlockReason === undefined
            ? row.pause_or_block_reason
            : transition.pauseOrBlockReason,
          now,
          runId,
        );
      const state = this.materializeStateFromRow(this.requireRunRow(runId));
      this.insertSnapshot(runId, nextSnapshot, state, now);
      this.db.exec("COMMIT");
      return {
        snapshotVersion: nextSnapshot,
        canonicalRevision: nextCanonical,
        eventCursor: nextEventCursor,
        workflowPlanRevision: nextPlan,
      };
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  freezeStage(
    runId: string,
    expected: StateBinding,
    contract: StageContract,
    gate: GateDefinition,
  ): StateBinding {
    if (
      contract.stageId !== gate.stageId ||
      contract.definedAtSnapshotVersion !== expected.snapshotVersion ||
      gate.definedAtSnapshotVersion !== expected.snapshotVersion
    ) {
      throw new Error("StageContract/GateDefinition are not bound before execution");
    }
    if (
      contract.sha256 !== canonicalSha256(stripContractHash(contract)) ||
      gate.sha256 !== canonicalSha256(stripGateHash(gate))
    ) {
      throw new Error("frozen StageContract/GateDefinition hash mismatch");
    }
    return this.casTransition(
      runId,
      expected,
      {
        lifecycle: "running",
        currentStageId: contract.stageId,
        eventType: "stage_frozen",
        eventPayload: {
          stageId: contract.stageId,
          contractId: contract.contractId,
          gateId: gate.gateId,
          contractHash: contract.sha256,
          gateHash: gate.sha256,
        },
      },
      (db) => {
        const now = timestamp();
        db.prepare(
          `INSERT INTO stage_contracts(
             contract_id, run_id, stage_id, revision, stage_type, role,
             defined_at_snapshot_version, sha256, contract_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          contract.contractId,
          runId,
          contract.stageId,
          contract.revision,
          contract.stageType,
          contract.role,
          contract.definedAtSnapshotVersion,
          contract.sha256,
          canonicalJson(contract),
          now,
        );
        db.prepare(
          `INSERT INTO gate_definitions(
             gate_id, run_id, stage_id, revision, defined_at_snapshot_version,
             sha256, gate_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          gate.gateId,
          runId,
          gate.stageId,
          gate.revision,
          gate.definedAtSnapshotVersion,
          gate.sha256,
          canonicalJson(gate),
          now,
        );
      },
    );
  }

  createTask(runId: string, input: TaskRecordInput): void {
    this.ensureWritable(runId);
    const now = timestamp();
    this.db
      .prepare(
        `INSERT INTO tasks(
           task_id, run_id, stage_id, role, input_message_type,
           expected_output_message_type, state_binding_json, input_hash,
           stage_contract_hash, skill_hash, schema_manifest_hash, task_json,
           status, output_attempt_count, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)`,
      )
      .run(
        input.taskId,
        runId,
        input.stageId,
        input.role,
        input.inputMessageType,
        input.expectedOutputMessageType,
        canonicalJson(input.stateBinding),
        input.inputHash,
        input.stageContractHash,
        input.skillHash,
        input.schemaManifestHash,
        canonicalJson(input.task),
        now,
        now,
      );
  }

  createAttempt(runId: string, input: AttemptInput): void {
    this.ensureWritable(runId);
    const now = timestamp();
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const task = this.db
        .prepare(
          "SELECT output_attempt_count, status FROM tasks WHERE task_id = ? AND run_id = ?",
        )
        .get(input.taskId, runId) as
        | { output_attempt_count: number; status: string }
        | undefined;
      if (!task) throw new Error(`unknown task ${input.taskId}`);
      if (input.attemptNo !== task.output_attempt_count + 1) {
        throw new Error("attempt number must be contiguous");
      }
      this.db
        .prepare(
          `INSERT INTO attempts(
             attempt_id, run_id, task_id, attempt_no, role, logical_effort,
             provider_wire_effort, status, started_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, 'created', ?)`,
        )
        .run(
          input.attemptId,
          runId,
          input.taskId,
          input.attemptNo,
          input.role,
          input.logicalEffort,
          input.providerWireEffort,
          now,
        );
      this.db
        .prepare(
          `UPDATE tasks SET output_attempt_count = ?, status = 'dispatched', updated_at = ?
           WHERE task_id = ?`,
        )
        .run(input.attemptNo, now, input.taskId);
      this.db.exec("COMMIT");
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  markAttemptRunning(attemptId: string): void {
    const result = this.db
      .prepare(
        `UPDATE attempts SET status = 'running'
         WHERE attempt_id = ? AND status = 'created'`,
      )
      .run(attemptId);
    if (Number(result.changes) !== 1) {
      throw new Error(`attempt ${attemptId} is not startable`);
    }
  }

  updateAttemptProviderIds(
    attemptId: string,
    threadId: string,
    turnId: string,
  ): void {
    this.db
      .prepare(
        `UPDATE attempts SET provider_thread_id = ?, provider_turn_id = ?, status = 'running'
         WHERE attempt_id = ?`,
      )
      .run(threadId, turnId, attemptId);
  }

  finishAttempt(
    attemptId: string,
    status: string,
    options: {
      rawResponseArtifactId?: string;
      errorCode?: string;
    } = {},
  ): void {
    this.db
      .prepare(
        `UPDATE attempts SET status = ?, raw_response_artifact_id = ?,
         error_code = ?, finished_at = ? WHERE attempt_id = ?`,
      )
      .run(
        status,
        options.rawResponseArtifactId ?? null,
        options.errorCode ?? null,
        timestamp(),
        attemptId,
      );
  }

  recordUsage(
    runId: string,
    attemptId: string,
    usage: {
      inputTokens: number;
      cachedInputTokens: number;
      outputTokens: number;
      reasoningOutputTokens: number;
      toolCalls: number;
      elapsedMs: number;
    },
  ): void {
    this.ensureWritable(runId);
    this.db
      .prepare(
        `INSERT INTO usage_records(
           usage_id, run_id, attempt_id, input_tokens, cached_input_tokens,
           output_tokens, reasoning_output_tokens, tool_calls, elapsed_ms,
           created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        `usage-${randomUUID()}`,
        runId,
        attemptId,
        nonNegativeInteger(usage.inputTokens),
        nonNegativeInteger(usage.cachedInputTokens),
        nonNegativeInteger(usage.outputTokens),
        nonNegativeInteger(usage.reasoningOutputTokens),
        nonNegativeInteger(usage.toolCalls),
        nonNegativeInteger(usage.elapsedMs),
        timestamp(),
      );
  }

  insertValidationReport(
    runId: string,
    report: ValidationReport,
    taskId: string | null,
    attemptId: string | null,
  ): string {
    const id = `validation-${randomUUID()}`;
    this.db
      .prepare(
        `INSERT INTO validation_reports(
           validation_report_id, run_id, task_id, attempt_id, validator_version,
           valid, report_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        id,
        runId,
        taskId,
        attemptId,
        report.validatorVersion,
        report.valid ? 1 : 0,
        canonicalJson(report),
        timestamp(),
      );
    return id;
  }

  recordOutputContractFailure(
    runId: string,
    report: ValidationReport,
    taskId: string,
    attemptId: string,
    rawResponseArtifactId: string,
    errorCode: string,
    retryable: boolean,
  ): string {
    const id = `validation-${randomUUID()}`;
    const now = timestamp();
    this.db.exec("BEGIN IMMEDIATE");
    try {
      this.db
        .prepare(
          `INSERT INTO validation_reports(
             validation_report_id, run_id, task_id, attempt_id,
             validator_version, valid, report_json, created_at
           ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)`,
        )
        .run(
          id,
          runId,
          taskId,
          attemptId,
          report.validatorVersion,
          canonicalJson(report),
          now,
        );
      const attempt = this.db
        .prepare(
          `UPDATE attempts
           SET status = 'output_contract_invalid',
               raw_response_artifact_id = ?, error_code = ?, finished_at = ?
           WHERE attempt_id = ? AND task_id = ?`,
        )
        .run(
          rawResponseArtifactId,
          errorCode,
          now,
          attemptId,
          taskId,
        );
      if (Number(attempt.changes) !== 1) {
        throw new Error("output failure attempt binding is stale");
      }
      const task = this.db
        .prepare(
          `UPDATE tasks SET status = ?, updated_at = ?
           WHERE task_id = ? AND run_id = ?`,
        )
        .run(
          retryable ? "pending_output_retry" : "failed",
          now,
          taskId,
          runId,
        );
      if (Number(task.changes) !== 1) {
        throw new Error("output failure task binding is stale");
      }
      this.db.exec("COMMIT");
      return id;
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  registerArtifact(runId: string, artifact: ArtifactRef): void {
    this.ensureWritable(runId);
    this.db
      .prepare(
        `INSERT INTO artifact_manifests(
           artifact_id, run_id, kind, relative_path, sha256, size_bytes,
           trust_class, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        artifact.artifactId,
        runId,
        artifact.kind,
        artifact.relativePath,
        artifact.sha256,
        artifact.sizeBytes,
        artifact.trustClass,
        timestamp(),
      );
  }

  saveCanonicalObject(
    db: DatabaseSync,
    runId: string,
    objectType: string,
    objectId: string,
    revision: number,
    value: unknown,
    sourceResultId: string | null,
  ): void {
    db.prepare(
      `UPDATE canonical_objects SET active = 0
       WHERE run_id = ? AND object_type = ? AND object_id = ? AND active = 1`,
    ).run(runId, objectType, objectId);
    db.prepare(
      `INSERT INTO canonical_objects(
         run_id, object_type, object_id, revision, object_json, object_hash,
         active, source_result_id, created_at
       ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)`,
    ).run(
      runId,
      objectType,
      objectId,
      revision,
      canonicalJson(value),
      canonicalSha256(value),
      sourceResultId,
      timestamp(),
    );
  }

  query(sql: string, ...params: unknown[]): Record<string, unknown>[] {
    return this.db.prepare(sql).all(...(params as never[])) as Record<
      string,
      unknown
    >[];
  }

  acquireLock(
    runId: string,
    ownerId: string,
    staleAfterMs = 300_000,
  ): void {
    const now = timestamp();
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const current = this.db
        .prepare(
          "SELECT owner_id, heartbeat_at FROM run_locks WHERE run_id = ?",
        )
        .get(runId) as
        | { owner_id: string; heartbeat_at: string }
        | undefined;
      if (current && !lockIsReclaimable(current, staleAfterMs)) {
        throw new Error(`run is already locked by ${current.owner_id}`);
      }
      if (current) {
        this.db
          .prepare("DELETE FROM run_locks WHERE run_id = ?")
          .run(runId);
      }
      this.db
        .prepare(
          `INSERT INTO run_locks(run_id, owner_id, acquired_at, heartbeat_at)
           VALUES (?, ?, ?, ?)`,
        )
        .run(runId, ownerId, now, now);
      this.db.exec("COMMIT");
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  heartbeatLock(runId: string, ownerId: string): void {
    const result = this.db
      .prepare(
        "UPDATE run_locks SET heartbeat_at = ? WHERE run_id = ? AND owner_id = ?",
      )
      .run(timestamp(), runId, ownerId);
    if (Number(result.changes) !== 1) throw new Error("run lock was lost");
  }

  releaseLock(runId: string, ownerId: string): void {
    this.db
      .prepare("DELETE FROM run_locks WHERE run_id = ? AND owner_id = ?")
      .run(runId, ownerId);
  }

  markCompleted(
    runId: string,
    expected: StateBinding,
    finalArtifactId: string,
  ): StateBinding {
    const completedAt = timestamp();
    return this.casTransition(
      runId,
      expected,
      {
        lifecycle: "completed",
        eventType: "run_completed",
        eventPayload: { finalArtifactId },
      },
      (db) => {
        const artifact = db
          .prepare(
            "SELECT artifact_id FROM artifact_manifests WHERE run_id = ? AND artifact_id = ?",
          )
          .get(runId, finalArtifactId);
        if (!artifact) {
          throw new Error("final artifact is not registered");
        }
        db.prepare(
          "UPDATE runs SET completed_at = ? WHERE run_id = ?",
        ).run(completedAt, runId);
      },
    );
  }

  private ensureWritable(runId: string): void {
    if (this.getRun(runId).lifecycle === "completed") {
      throw new Error("completed run is immutable");
    }
  }

  private requireRunRow(runId: string): RunRow {
    const row = this.db
      .prepare("SELECT * FROM runs WHERE run_id = ?")
      .get(runId) as RunRow | undefined;
    if (!row) throw new Error(`unknown run ${runId}`);
    return row;
  }

  private materializeStateFromRow(row: RunRow): WorkflowState {
    const runnableStageIds = this.db
      .prepare(
        `SELECT stage_id FROM workflow_plan_nodes
         WHERE run_id = ? AND plan_revision = ? AND lifecycle = 'runnable'
         ORDER BY stage_id`,
      )
      .all(row.run_id, row.workflow_plan_revision)
      .map((entry) => String((entry as { stage_id: string }).stage_id));
    const taskRows = this.db
      .prepare(
        `SELECT task_id, status FROM tasks WHERE run_id = ? ORDER BY task_id`,
      )
      .all(row.run_id) as Array<{ task_id: string; status: string }>;
    const resultRows = this.db
      .prepare(
        `SELECT r.result_id
         FROM turn_results r
         LEFT JOIN result_consumptions c ON c.result_id = r.result_id
         WHERE r.run_id = ? AND r.status = 'committed' AND c.result_id IS NULL
         ORDER BY r.result_id`,
      )
      .all(row.run_id) as Array<{ result_id: string }>;
    const attempts = this.db
      .prepare(
        `SELECT task_id, COUNT(*) AS count FROM attempts
         WHERE run_id = ? GROUP BY task_id`,
      )
      .all(row.run_id) as Array<{ task_id: string; count: number }>;
    const runConfig = JSON.parse(row.config_json) as {
      maxTurns?: number;
      maxInputTokens?: number;
      maxOutputTokens?: number;
      maxToolCalls?: number;
      maxElapsedMs?: number;
    };
    const usage = this.db
      .prepare(
        `SELECT
           COALESCE(SUM(input_tokens), 0) AS input_tokens,
           COALESCE(SUM(output_tokens), 0) AS output_tokens,
           COALESCE(SUM(tool_calls), 0) AS tool_calls,
           COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms
         FROM usage_records WHERE run_id = ?`,
      )
      .get(row.run_id) as {
      input_tokens: number;
      output_tokens: number;
      tool_calls: number;
      elapsed_ms: number;
    };
    const turnsUsed = attempts.reduce(
      (sum, attempt) => sum + Number(attempt.count),
      0,
    );
    const maxTurns = Number(runConfig.maxTurns ?? 100);
    const maxInputTokens = Number(runConfig.maxInputTokens ?? 1_000_000);
    const maxOutputTokens = Number(runConfig.maxOutputTokens ?? 500_000);
    const maxToolCalls = Number(runConfig.maxToolCalls ?? 1_000);
    const maxElapsedMs = Number(runConfig.maxElapsedMs ?? 86_400_000);
    const exhaustedDimensions: string[] = [];
    if (turnsUsed >= maxTurns) exhaustedDimensions.push("turns");
    if (Number(usage.input_tokens) >= maxInputTokens) {
      exhaustedDimensions.push("input_tokens");
    }
    if (Number(usage.output_tokens) >= maxOutputTokens) {
      exhaustedDimensions.push("output_tokens");
    }
    if (Number(usage.tool_calls) >= maxToolCalls) {
      exhaustedDimensions.push("tool_calls");
    }
    if (Number(usage.elapsed_ms) >= maxElapsedMs) {
      exhaustedDimensions.push("elapsed_ms");
    }
    const semanticNoProgressCount = this.semanticNoProgressCount(row.run_id);
    return {
      workflowId: row.workflow_id,
      runId: row.run_id,
      snapshotVersion: row.snapshot_version,
      canonicalRevision: row.canonical_revision,
      eventCursor: row.event_cursor,
      workflowPlanRevision: row.workflow_plan_revision,
      lifecycle: row.lifecycle as WorkflowLifecycle,
      currentStageId: row.current_stage_id,
      activeFocusRef: row.active_focus_ref_json
        ? JSON.parse(row.active_focus_ref_json)
        : null,
      runnableStageIds,
      pendingTaskIds: taskRows
        .filter((task) => task.status === "pending")
        .map((task) => task.task_id),
      inFlightTaskIds: taskRows
        .filter((task) => ["dispatched", "running"].includes(task.status))
        .map((task) => task.task_id),
      committedUnconsumedResultIds: resultRows.map((row) => row.result_id),
      pendingProposalIds: this.db
        .prepare(
          `SELECT proposal_id FROM decision_proposals
           WHERE run_id = ? AND status = 'pending' ORDER BY proposal_id`,
        )
        .all(row.run_id)
        .map((entry) => String((entry as { proposal_id: string }).proposal_id)),
      retryCounters: Object.fromEntries(
        attempts.map((attempt) => [attempt.task_id, Number(attempt.count)]),
      ),
      noProgressCounters: {
        semanticTransitionsWithoutCanonicalDelta: semanticNoProgressCount,
      },
      budgetState: {
        turnsUsed,
        maxTurns,
        inputTokensUsed: Number(usage.input_tokens),
        maxInputTokens,
        outputTokensUsed: Number(usage.output_tokens),
        maxOutputTokens,
        toolCallsUsed: Number(usage.tool_calls),
        maxToolCalls,
        elapsedMs: Number(usage.elapsed_ms),
        maxElapsedMs,
        exhaustedDimensions,
      },
      pauseOrBlockReason: row.pause_or_block_reason,
    };
  }

  private semanticNoProgressCount(runId: string): number {
    const rows = this.db
      .prepare(
        `SELECT e.event_type, s.state_json
         FROM events e
         JOIN snapshots s
           ON s.run_id = e.run_id
          AND s.snapshot_version = e.snapshot_version
         WHERE e.run_id = ?
         ORDER BY e.event_cursor`,
      )
      .all(runId) as Array<{ event_type: string; state_json: string }>;
    let previousCanonical = 0;
    let count = 0;
    for (const row of rows) {
      const state = JSON.parse(row.state_json) as {
        canonicalRevision: number;
      };
      if (state.canonicalRevision !== previousCanonical) {
        previousCanonical = state.canonicalRevision;
        count = 0;
      }
      if (
        row.event_type === "turn_result_committed" ||
        row.event_type === "turn_result_consumed"
      ) {
        count = 0;
      } else if (
        row.event_type.startsWith("workflow_") ||
        row.event_type === "stage_gate_failed" ||
        row.event_type === "closure_review_rejected" ||
        row.event_type === "closure_preflight_rejected"
      ) {
        count += 1;
      }
    }
    return count;
  }

  private insertSnapshot(
    runId: string,
    snapshotVersion: number,
    state: WorkflowState,
    now: string,
  ): void {
    this.db
      .prepare(
        `INSERT INTO snapshots(run_id, snapshot_version, state_json, created_at)
         VALUES (?, ?, ?, ?)`,
      )
      .run(runId, snapshotVersion, canonicalJson(state), now);
  }
}

interface RunRow {
  run_id: string;
  workflow_id: string;
  objective: string;
  objective_hash: string;
  acceptance_criteria_json: string;
  acceptance_criteria_hash: string;
  lifecycle: string;
  snapshot_version: number;
  canonical_revision: number;
  event_cursor: number;
  workflow_plan_revision: number;
  current_stage_id: string | null;
  active_focus_ref_json: string | null;
  config_json: string;
  pause_or_block_reason: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

function mapRunRow(row: RunRow): RunRecord {
  return {
    runId: row.run_id,
    workflowId: row.workflow_id,
    objective: row.objective,
    objectiveHash: row.objective_hash,
    acceptanceCriteria: JSON.parse(row.acceptance_criteria_json),
    acceptanceCriteriaHash: row.acceptance_criteria_hash,
    lifecycle: row.lifecycle as WorkflowLifecycle,
    snapshotVersion: row.snapshot_version,
    canonicalRevision: row.canonical_revision,
    eventCursor: row.event_cursor,
    workflowPlanRevision: row.workflow_plan_revision,
    currentStageId: row.current_stage_id,
    activeFocusRef: row.active_focus_ref_json
      ? JSON.parse(row.active_focus_ref_json)
      : null,
    config: JSON.parse(row.config_json),
    pauseOrBlockReason: row.pause_or_block_reason,
    completedAt: row.completed_at,
  };
}

function bindingFromRow(row: RunRow): StateBinding {
  return {
    snapshotVersion: row.snapshot_version,
    canonicalRevision: row.canonical_revision,
    eventCursor: row.event_cursor,
    workflowPlanRevision: row.workflow_plan_revision,
  };
}

function sameBinding(left: StateBinding, right: StateBinding): boolean {
  return (
    left.snapshotVersion === right.snapshotVersion &&
    left.canonicalRevision === right.canonicalRevision &&
    left.eventCursor === right.eventCursor &&
    left.workflowPlanRevision === right.workflowPlanRevision
  );
}

function stripContractHash(contract: StageContract): unknown {
  const { sha256: _sha256, ...rest } = contract;
  return rest;
}

function stripGateHash(gate: GateDefinition): unknown {
  const { sha256: _sha256, ...rest } = gate;
  return rest;
}

function timestamp(): string {
  return new Date().toISOString();
}

function nonNegativeInteger(value: number): number {
  if (!Number.isFinite(value) || value < 0) return 0;
  return Math.trunc(value);
}

function lockIsReclaimable(
  lock: { owner_id: string; heartbeat_at: string },
  staleAfterMs: number,
): boolean {
  const match = lock.owner_id.match(/^controller-pid-(\d+)-/);
  if (match) {
    const pid = Number(match[1]);
    if (Number.isSafeInteger(pid) && pid > 0 && !processIsAlive(pid)) {
      return true;
    }
  }
  const heartbeat = Date.parse(lock.heartbeat_at);
  return (
    !Number.isFinite(heartbeat) ||
    Date.now() - heartbeat > Math.max(1, staleAfterMs)
  );
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as { code?: string }).code === "EPERM";
  }
}
