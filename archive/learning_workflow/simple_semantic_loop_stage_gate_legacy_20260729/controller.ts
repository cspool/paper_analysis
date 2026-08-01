import { randomUUID } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import {
  mkdirSync,
  readFileSync,
  statSync,
} from "node:fs";
import { resolve } from "node:path";
import type {
  AnyTurnTask,
  ClosureReviewEnvelope,
  ClosureReviewTaskEnvelope,
  EvidencePacketEnvelope,
  GateActual,
  GateDefinition,
  GateValueType,
  RegisteredRole,
  ReviewDeltaEnvelope,
  SearchNeed,
  SemanticDelta,
  StageContract,
  StageContractDraft,
  StateBinding,
  TurnCorrectionFeedback,
  TurnBudget,
  ValidationReport,
  WorkflowDecisionProposal,
  WorkflowPermissionEnvelope,
  WorkflowTurnTask,
} from "./contracts/index.ts";
import {
  MAX_OUTPUT_ATTEMPTS_PER_TASK,
  MAX_PROVIDER_FAILURES_PER_TASK,
  MAX_TOTAL_ATTEMPTS_PER_TASK,
  ROLE_MESSAGE_TYPES,
  ROLE_REASONING_EFFORT,
  RUBRIC_REGISTRY,
  STAGE_REGISTRY,
  canonicalEqual,
  canonicalJson,
  canonicalSha256,
  sha256Bytes,
} from "./contracts/index.ts";
import type { WorkflowStore } from "./db/workflow_store.ts";
import {
  exportRun,
  atomicWriteJson,
  atomicWriteText,
} from "./exporter.ts";
import { finalizeAcceptedClosure } from "./closure/finalization.ts";
import { runClosurePreflight } from "./closure/preflight.ts";
import {
  evaluateGate,
  GATE_EVALUATOR_VERSION,
  type GateEvaluationContext,
  type GateResolution,
} from "./stages/gate_engine.ts";
import {
  GATE_COMPILER_POLICY_VERSION,
} from "./stages/gate_compiler.ts";
import {
  isForbiddenExecutionCapability,
  validateRuntimeToolEvents,
} from "./security/no_experiment_guard.ts";
import {
  dispatchFreshTurnAttempt,
  validateCapturedTurnOutput,
} from "./turns/dispatcher.ts";
import type {
  FreshTurnRuntime,
  RawTurnResult,
} from "./turns/runtime.ts";
import {
  resolveWireEffort,
  type EffortCapabilityManifest,
} from "./turns/role_profiles.ts";
import {
  TurnTaskFactory,
  defaultWorkflowPermission,
} from "./turns/task_factory.ts";
import {
  bindPayloadTaskHash,
  buildWorkflowTurnPrompt,
} from "./turns/prompt_builder.ts";
import { validateTaskForDispatch } from "./validators/index.ts";
import {
  freezeStageDraft,
  loadCurrentPlan,
} from "./workflow/plan_store.ts";
import {
  calculateRunnableStages,
  chooseDeterministicallyEquivalentStage,
} from "./workflow/runnable_stage.ts";
import {
  buildRegisteredTrigger,
  type TriggerSignals,
} from "./workflow/trigger_engine.ts";
import {
  commitWorkflowProposal,
  preflightWorkflowProposal,
} from "./workflow/proposal_commit.ts";
import {
  commitTurnResult,
  reconcileCommittedReplacementFailures,
} from "./workflow/result_consumer.ts";
import { isQuiescent } from "./workflow/state_machine.ts";

export interface ControllerConfig {
  projectRoot: string;
  workDir: string;
  model: string;
  skillRoot: string;
  schemaManifestPath: string;
  capabilityManifest: EffortCapabilityManifest;
  budgets: {
    workflow: TurnBudget;
    evidence: TurnBudget;
    direction: TurnBudget;
    closure: TurnBudget;
  };
  maxTransitionsPerRun: number;
  noProgressThreshold: number;
}

export interface ControllerRunResult {
  runId: string;
  lifecycle: string;
  transitions: number;
  completed: boolean;
}

interface TaskRow {
  task_id: string;
  stage_id: string;
  role: RegisteredRole;
  input_message_type: string;
  expected_output_message_type: string;
  state_binding_json: string;
  input_hash: string;
  stage_contract_hash: string;
  skill_hash: string;
  schema_manifest_hash: string;
  task_json: string;
  status: string;
  output_attempt_count: number;
}

export class SemanticLoopController {
  private readonly store: WorkflowStore;
  private readonly runtime: FreshTurnRuntime;
  private readonly config: ControllerConfig;
  private readonly taskFactory: TurnTaskFactory;
  private readonly ownerId =
    `controller-pid-${process.pid}-${randomUUID()}`;

  constructor(
    store: WorkflowStore,
    runtime: FreshTurnRuntime,
    config: ControllerConfig,
  ) {
    this.store = store;
    this.runtime = runtime;
    this.config = config;
    this.taskFactory = new TurnTaskFactory(store, {
      projectRoot: config.projectRoot,
      skillRoot: config.skillRoot,
      schemaManifestPath: config.schemaManifestPath,
    });
  }

  async run(runId: string): Promise<ControllerRunResult> {
    this.store.acquireLock(runId, this.ownerId);
    let transitions = 0;
    try {
      this.recoverCapturedRawAttempts(runId);
      this.reconcileInflightAttempts(runId);
      reconcileCommittedReplacementFailures(this.store, runId);
      this.reconcileCommittedClosureReview(runId);
      while (transitions < this.config.maxTransitionsPerRun) {
        this.store.heartbeatLock(runId, this.ownerId);
        const run = this.store.getRun(runId);
        exportRun(this.store, runId, this.config.workDir);
        if (isQuiescent(run.lifecycle)) {
          return result(runId, run.lifecycle, transitions);
        }

        if (run.lifecycle === "finalizing") {
          this.finalize(runId);
          transitions += 1;
          continue;
        }

        const budgetState = this.store.readWorkflowState(runId).budgetState;
        if (budgetState.exhaustedDimensions.length > 0) {
          this.store.casTransition(runId, this.store.stateBinding(runId), {
            lifecycle: "paused_budget",
            pauseOrBlockReason: `global budget exhausted: ${budgetState.exhaustedDimensions.join(", ")}`,
            eventType: "controller_global_budget_exhausted",
            eventPayload: budgetState,
          });
          transitions += 1;
          continue;
        }

        const pendingTask = this.nextPendingTask(runId);
        if (pendingTask) {
          await this.dispatchTask(runId, pendingTask);
          transitions += 1;
          continue;
        }

        if (run.lifecycle === "closure_preflight") {
          this.scheduleClosureReview(runId);
          transitions += 1;
          continue;
        }

        const plan = loadCurrentPlan(this.store, runId);
        const runnable = calculateRunnableStages(plan);
        this.promoteRunnableNodes(runId, run.workflowPlanRevision, runnable);
        const refreshedRunnable = calculateRunnableStages(
          loadCurrentPlan(this.store, runId),
        );
        if (refreshedRunnable.length === 1) {
          this.prepareOrExecuteStage(runId, refreshedRunnable[0]!);
          transitions += 1;
          continue;
        }
        if (refreshedRunnable.length > 1) {
          const equivalent = chooseDeterministicallyEquivalentStage(
            refreshedRunnable,
            (stage) => `${stage.stageType}\0${stage.role ?? ""}`,
          );
          if (equivalent) {
            this.prepareOrExecuteStage(runId, equivalent);
            transitions += 1;
            continue;
          }
        }

        const signals = this.buildTriggerSignals(runId, refreshedRunnable.length);
        const trigger = buildRegisteredTrigger(
          this.store.readWorkflowState(runId),
          signals,
        );
        if (!trigger) {
          this.failRun(
            runId,
            "state_machine_gap",
            "No deterministic transition or registered semantic trigger exists.",
            true,
          );
          transitions += 1;
          continue;
        }
        this.scheduleWorkflowDecision(runId, trigger);
        transitions += 1;
      }
      const current = this.store.getRun(runId);
      if (!isQuiescent(current.lifecycle)) {
        this.store.casTransition(runId, this.store.stateBinding(runId), {
          lifecycle: "paused_budget",
          pauseOrBlockReason: "maxTransitionsPerRun reached",
          eventType: "controller_transition_budget_exhausted",
          eventPayload: { maxTransitionsPerRun: this.config.maxTransitionsPerRun },
        });
      }
      const final = this.store.getRun(runId);
      return result(runId, final.lifecycle, transitions);
    } finally {
      exportRun(this.store, runId, this.config.workDir);
      this.store.releaseLock(runId, this.ownerId);
    }
  }

  private nextPendingTask(runId: string): TaskRow | null {
    return (
      (this.store.db
        .prepare(
          `SELECT * FROM tasks
           WHERE run_id = ? AND status IN ('pending', 'pending_output_retry')
           ORDER BY created_at, task_id LIMIT 1`,
        )
        .get(runId) as TaskRow | undefined) ?? null
    );
  }

  private async dispatchTask(runId: string, row: TaskRow): Promise<void> {
    const contract = this.loadContract(
      runId,
      row.stage_id,
      row.stage_contract_hash,
    );
    const gate = this.loadGate(
      runId,
      row.stage_id,
      row.stage_contract_hash,
    );
    let logicalTask = JSON.parse(row.task_json) as AnyTurnTask;
    let correctionFeedback =
      taskCorrectionFeedback(logicalTask) ??
      this.loadLatestCorrectionFeedback(runId, row.task_id);
    const latestAttempt = this.store.db
      .prepare(
        `SELECT status FROM attempts
         WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1`,
      )
      .get(row.task_id) as { status: string } | undefined;
    if (
      latestAttempt?.status === "output_contract_invalid" &&
      correctionFeedback === null
    ) {
      this.markTaskAndStageFailed(
        runId,
        row.task_id,
        row.stage_id,
      );
      this.failRun(
        runId,
        "correction_feedback_missing",
        "Output correction retry has no verified previous error packet.",
        true,
      );
      return;
    }
    if (
      correctionFeedback !== null &&
      !canonicalEqual(
        getTaskStateBinding(logicalTask),
        this.store.stateBinding(runId),
      )
    ) {
      this.markTaskAndStageFailed(
        runId,
        row.task_id,
        row.stage_id,
      );
      this.failRun(
        runId,
        "correction_task_stale",
        "Canonical state changed before the bounded correction retry.",
        true,
      );
      return;
    }
    let attemptNo = row.output_attempt_count;

    while (attemptNo < MAX_TOTAL_ATTEMPTS_PER_TASK) {
      attemptNo += 1;
      const attemptId =
        attemptNo === 1
          ? logicalTask.attemptId
          : `attempt-${randomUUID()}`;
      logicalTask = this.bindRetryTask(
        runId,
        logicalTask,
        row.role,
        attemptId,
        correctionFeedback,
      );
      this.store.db
        .prepare(
          `UPDATE tasks SET task_json = ?, input_hash = ?,
           state_binding_json = ?, status = 'pending',
           updated_at = ? WHERE task_id = ?`,
        )
        .run(
          canonicalJson(logicalTask),
          getTaskInputHash(logicalTask),
          canonicalJson(getTaskStateBinding(logicalTask)),
          timestamp(),
          row.task_id,
        );
      const wireEffort = resolveWireEffort(
        row.role,
        this.config.capabilityManifest,
      );
      const expectedSchema = this.taskFactory.schemaForMessage(
        row.expected_output_message_type,
      );
      this.store.createAttempt(runId, {
        attemptId,
        taskId: row.task_id,
        attemptNo,
        role: row.role,
        logicalEffort: ROLE_REASONING_EFFORT[row.role],
        providerWireEffort: wireEffort,
      });
      this.store.markAttemptRunning(attemptId);
      const taskContext = {
        role: row.role,
        frozenBudget: contract.budget,
        currentState: this.store.stateBinding(runId),
        stageContractHash: contract.sha256,
        schemaManifestSha256: row.schema_manifest_hash,
        skillSha256: row.skill_hash,
        expectedInputHash: getTaskInputHash(logicalTask),
        expectedOutputSchemaSha256: canonicalSha256(expectedSchema),
        rubricSha256:
          row.role === "direction_reviewer"
            ? RUBRIC_REGISTRY.direction_readiness_v1.sha256
            : row.role === "closure_reviewer"
              ? RUBRIC_REGISTRY.closure_v1.sha256
              : undefined,
      };
      const dispatch = await dispatchFreshTurnAttempt({
        role: row.role,
        task: logicalTask,
        stageContract: contract,
        taskValidationContext: taskContext,
        skillRoot: `${this.config.skillRoot}/${getSkillName(logicalTask)}`,
        expectedSchema,
        runtime: this.runtime,
        model: this.config.model,
        providerWireEffort: wireEffort,
        cwd: this.config.projectRoot,
      });

      if (!dispatch.dispatched) {
        this.store.insertValidationReport(
          runId,
          dispatch.preDispatchReport,
          row.task_id,
          attemptId,
        );
        this.store.finishAttempt(attemptId, "input_contract_invalid", {
          errorCode: "input_contract_invalid",
        });
        this.markTaskAndStageFailed(
          runId,
          row.task_id,
          row.stage_id,
        );
        this.failRun(
          runId,
          "input_contract_invalid",
          "Task failed authoritative validation before provider dispatch.",
          true,
        );
        return;
      }
      this.savePrompt(
        runId,
        attemptId,
        dispatch.promptText!,
        dispatch.promptSha256!,
      );
      const rawArtifactId = this.saveRawTurn(runId, attemptId, dispatch.rawTurn!);
      this.store.recordUsage(runId, attemptId, {
        inputTokens: dispatch.rawTurn!.usage.inputTokens,
        cachedInputTokens: dispatch.rawTurn!.usage.cachedInputTokens,
        outputTokens: dispatch.rawTurn!.usage.outputTokens,
        reasoningOutputTokens:
          dispatch.rawTurn!.usage.reasoningOutputTokens,
        toolCalls: dispatch.rawTurn!.toolEvents.length,
        elapsedMs: dispatch.rawTurn!.elapsedMs,
      });
      this.store.updateAttemptProviderIds(
        attemptId,
        dispatch.rawTurn!.providerThreadId,
        dispatch.rawTurn!.providerTurnId,
      );
      if (!dispatch.securityReport?.valid) {
        const reportId = this.store.insertValidationReport(
          runId,
          dispatch.securityReport!,
          row.task_id,
          attemptId,
        );
        this.store.finishAttempt(attemptId, "security_invalid", {
          rawResponseArtifactId: rawArtifactId,
          errorCode: "security.runtime_event_violation",
        });
        this.markTaskAndStageFailed(
          runId,
          row.task_id,
          row.stage_id,
        );
        this.failRun(
          runId,
          "security.runtime_event_violation",
          `Runtime tool admission failed (${reportId}).`,
          true,
        );
        return;
      }
      const usageReport = validateAttemptUsage(
        dispatch.rawTurn!,
        contract.budget,
      );
      if (!usageReport.valid) {
        this.store.insertValidationReport(
          runId,
          usageReport,
          row.task_id,
          attemptId,
        );
        this.store.finishAttempt(attemptId, "budget_invalid", {
          rawResponseArtifactId: rawArtifactId,
          errorCode: usageReport.errors[0]?.code ?? "budget.turn_exceeded",
        });
        this.markTaskAndStageFailed(
          runId,
          row.task_id,
          row.stage_id,
        );
        this.store.casTransition(runId, this.store.stateBinding(runId), {
          lifecycle: "paused_budget",
          pauseOrBlockReason: "Turn exceeded its frozen Stage budget.",
          eventType: "turn_budget_exceeded",
          eventPayload: {
            taskId: row.task_id,
            attemptId,
            errors: usageReport.errors,
          },
        });
        return;
      }
      if (dispatch.rawTurn!.status !== "completed") {
        const providerCode =
          dispatch.rawTurn!.status === "interrupted"
            ? "provider.interrupted"
            : "provider.failed";
        this.store.finishAttempt(attemptId, "provider_failed", {
          rawResponseArtifactId: rawArtifactId,
          errorCode: providerCode,
        });
        const providerFailures = Number(
          (
            this.store.db
              .prepare(
                `SELECT COUNT(*) AS count FROM attempts
                 WHERE task_id = ? AND status = 'provider_failed'`,
              )
              .get(row.task_id) as { count: number }
          ).count,
        );
        if (
          providerFailures < MAX_PROVIDER_FAILURES_PER_TASK &&
          attemptNo < MAX_TOTAL_ATTEMPTS_PER_TASK
        ) {
          continue;
        }
        this.markTaskAndStageFailed(
          runId,
          row.task_id,
          row.stage_id,
        );
        this.failRun(
          runId,
          providerCode,
          dispatch.rawTurn!.error ?? "Provider Turn failed.",
          false,
        );
        return;
      }
      let report = dispatch.resultValidationReport;
      if (
        report?.valid &&
        dispatch.result &&
        row.role === "workflow_decision"
      ) {
        report = preflightWorkflowProposal(
          this.store,
          logicalTask as WorkflowTurnTask,
          dispatch.result as WorkflowDecisionProposal,
        );
      }
      if (!report?.valid || !dispatch.result) {
        const rejectedReport =
          report ?? normalizationFailureReport("missing validated result");
        const priorOutputFailures = Number(
          (
            this.store.db
              .prepare(
                `SELECT COUNT(*) AS count FROM attempts
                 WHERE task_id = ? AND status = 'output_contract_invalid'`,
              )
              .get(row.task_id) as { count: number }
          ).count,
        );
        const retryable =
          priorOutputFailures <
            MAX_OUTPUT_ATTEMPTS_PER_TASK - 1 &&
          attemptNo < MAX_TOTAL_ATTEMPTS_PER_TASK;
        const validationReportId =
          this.store.recordOutputContractFailure(
          runId,
          rejectedReport,
          row.task_id,
          attemptId,
          rawArtifactId,
          rejectedReport.errors[0]?.code ?? "schema.invalid",
          retryable,
        );
        correctionFeedback = buildCorrectionFeedback(
          attemptId,
          dispatch.rawTurn!.text,
          rejectedReport,
          validationReportId,
        );
        if (retryable) {
          continue;
        }
        this.markTaskAndStageFailed(
          runId,
          row.task_id,
          row.stage_id,
        );
        this.failRun(
          runId,
          "output_attempts_exhausted",
          `The producing role exhausted ${MAX_OUTPUT_ATTEMPTS_PER_TASK} fresh output attempts.`,
          false,
        );
        return;
      }

      this.store.insertValidationReport(
        runId,
        report,
        row.task_id,
        attemptId,
      );
      if (row.role === "workflow_decision") {
        const workflowGateEvaluation = evaluateGate(
          gate,
          this.gateContext(
            runId,
            gate,
            dispatch.result,
            report,
            dispatch.rawTurn!,
            row.role,
            logicalTask,
            dispatch.securityReport!,
          ),
        );
        if (!workflowGateEvaluation.passed) {
          this.recordGateFailure(
            runId,
            row,
            attemptId,
            gate,
            workflowGateEvaluation,
            rawArtifactId,
          );
          return;
        }
        const commit = commitWorkflowProposal(
          this.store,
          logicalTask as WorkflowTurnTask,
          dispatch.result as WorkflowDecisionProposal,
          {
            gateId: gate.gateId,
            gateEvaluation: workflowGateEvaluation,
            rawResponseArtifactId: rawArtifactId,
          },
        );
        if (!commit.accepted) {
          this.store.finishAttempt(
            attemptId,
            "proposal_rejected",
            {
              rawResponseArtifactId: rawArtifactId,
              errorCode: commit.rejectionCode ?? undefined,
            },
          );
          this.store.db
            .prepare(
              "UPDATE tasks SET status = 'failed', updated_at = ? WHERE task_id = ?",
            )
            .run(timestamp(), row.task_id);
        }
        return;
      }

      const gateEvaluation = evaluateGate(
        gate,
        this.gateContext(
          runId,
          gate,
          dispatch.result,
          report,
          dispatch.rawTurn!,
          row.role,
          logicalTask,
          dispatch.securityReport!,
        ),
      );
      if (!gateEvaluation.passed) {
        this.recordGateFailure(
          runId,
          row,
          attemptId,
          gate,
          gateEvaluation,
          rawArtifactId,
        );
        return;
      }
      const resultId = getResultId(dispatch.result);
      const committed = commitTurnResult(
        this.store,
        runId,
        getTaskStateBinding(logicalTask),
        {
          resultId,
          taskId: row.task_id,
          attemptId,
          stageId: row.stage_id,
          gateId: gate.gateId,
          role: row.role,
          messageType: row.expected_output_message_type,
          result: dispatch.result,
          validationReport: report,
          gateEvaluation,
          rawResponseArtifactId: rawArtifactId,
          committedLifecycle:
            row.role === "closure_reviewer"
              ? "waiting_closure_review"
              : "running",
        },
      );
      if (row.role === "closure_reviewer") {
        const closure = dispatch.result as ClosureReviewEnvelope;
        this.afterClosureReview(
          runId,
          committed.nextState,
          closure,
          resultId,
        );
      }
      return;
    }
  }

  private bindRetryTask(
    runId: string,
    task: AnyTurnTask,
    role: RegisteredRole,
    attemptId: string,
    correctionFeedback: TurnCorrectionFeedback | null,
  ): AnyTurnTask {
    const clone = structuredClone(task);
    clone.attemptId = attemptId;
    const currentState = this.store.stateBinding(runId);
    if (role === "workflow_decision") {
      const workflow = clone as WorkflowTurnTask;
      workflow.stateSnapshot = currentState;
      workflow.decisionInputHash = "";
      workflow.correctionFeedback = correctionFeedback;
      const skill = this.taskFactory.skillPackage(role);
      return buildWorkflowTurnPrompt({
        task: workflow,
        skillMarkdown: skill.skillMarkdown,
        expectedSchema: this.taskFactory.schemaForMessage(
          "WORKFLOW_DECISION_PROPOSAL",
        ),
      }).task;
    }
    const payload = clone as Exclude<
      AnyTurnTask,
      WorkflowTurnTask
    >;
    payload.stateBinding = currentState;
    payload.inputHash = "";
    payload.payload.correctionFeedback = correctionFeedback;
    return bindPayloadTaskHash(payload);
  }

  private loadLatestCorrectionFeedback(
    runId: string,
    taskId: string,
  ): TurnCorrectionFeedback | null {
    const row = this.store.db
      .prepare(
        `SELECT a.attempt_id, a.attempt_no,
                vr.validation_report_id, vr.report_json
         FROM attempts a
         JOIN validation_reports vr
           ON vr.run_id = a.run_id
          AND vr.task_id = a.task_id
          AND vr.attempt_id = a.attempt_id
          AND vr.valid = 0
         WHERE a.run_id = ? AND a.task_id = ?
           AND a.status = 'output_contract_invalid'
         ORDER BY a.attempt_no DESC, vr.created_at DESC,
                  vr.validation_report_id DESC
         LIMIT 1`,
      )
      .get(runId, taskId) as
      | {
          attempt_id: string;
          attempt_no: number;
          validation_report_id: string;
          report_json: string;
        }
      | undefined;
    if (!row) return null;
    try {
      const raw = JSON.parse(
        this.readVerifiedArtifact(
          runId,
          `raw-${row.attempt_id}`,
        ).toString("utf8"),
      ) as { text?: unknown };
      if (typeof raw.text !== "string") return null;
      return buildCorrectionFeedback(
        row.attempt_id,
        raw.text,
        JSON.parse(row.report_json) as ValidationReport,
        row.validation_report_id,
      );
    } catch {
      return null;
    }
  }

  private prepareOrExecuteStage(
    runId: string,
    node: ReturnType<typeof calculateRunnableStages>[number],
  ): void {
    const contract = this.loadContract(runId, node.stageId);
    if (node.executionKind === "SCRIPT_TRANSITION") {
      this.executeScriptStage(runId, node.stageId, contract);
      return;
    }
    if (node.stageType === "EVIDENCE_READ") {
      if (this.ensureSearchNeedCanonical(runId, contract)) return;
    }
    const existing = this.store.db
      .prepare(
        `SELECT task_id FROM tasks WHERE run_id = ? AND stage_id = ?
         AND status NOT IN ('failed', 'cancelled') LIMIT 1`,
      )
      .get(runId, node.stageId);
    if (existing) return;
    if (!node.role) throw new Error("Agent Stage has no registered role");
    const gate = this.loadGate(runId, node.stageId);
    const task = this.taskFactory.buildStageTask(runId, {
      stageId: node.stageId,
      stageType: node.stageType,
      role: node.role,
      contract,
      gate,
    });
    this.insertTask(runId, task, contract);
    this.store.db
      .prepare(
        `UPDATE workflow_plan_nodes SET lifecycle = 'dispatched'
         WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
      )
      .run(
        runId,
        this.store.getRun(runId).workflowPlanRevision,
        node.stageId,
      );
  }

  private executeScriptStage(
    runId: string,
    stageId: string,
    contract: StageContract,
  ): void {
    const proposal = this.proposalForContract(runId, contract);
    const gate = this.loadGate(runId, stageId);
    const domain = proposal.domainProposal;
    if (!domain) throw new Error("script transition has no DomainProposal");
    const expected = this.store.stateBinding(runId);
    if (
      contract.stageType === "SCRIPT_APPLY_TOPIC_FRAME" &&
      domain.kind !== "topic_frame"
    ) {
      throw new Error("Topic script Stage/domain mismatch");
    }
    if (
      contract.stageType === "SCRIPT_APPLY_SEMANTIC_DELTA" &&
      domain.kind !== "semantic_delta"
    ) {
      throw new Error("SemanticDelta script Stage/domain mismatch");
    }
    const semanticObject =
      domain.kind === "topic_frame"
        ? {
            objectType: "topic",
            objectId: domain.value.topicId,
            revision: domain.value.revision,
            value: domain.value,
          }
        : domain.kind === "semantic_delta" && domain.value.proposedObject
          ? {
              objectType: domain.value.target.type,
              objectId: domain.value.target.id,
              revision:
                (domain.value.proposedObject as { revision: number }).revision,
              value: domain.value.proposedObject,
            }
          : null;
    if (domain.kind === "semantic_delta" && semanticObject) {
      const active = this.store.db
        .prepare(
          `SELECT revision FROM canonical_objects
           WHERE run_id = ? AND object_type = ? AND object_id = ? AND active = 1`,
        )
        .get(runId, semanticObject.objectType, semanticObject.objectId) as
        | { revision: number }
        | undefined;
      if (domain.value.action === "create") {
        if (active || domain.value.expectedTargetRevision !== 0) {
          throw new Error("SemanticDelta create target already exists");
        }
      } else if (
        !active ||
        active.revision !== domain.value.expectedTargetRevision
      ) {
        throw new Error("SemanticDelta expectedTargetRevision is stale");
      }
    }
    const basisResults =
      domain.kind === "semantic_delta"
        ? this.loadBasisTurnResults(runId, domain.value)
        : [];
    const scriptValidation: ValidationReport = {
      validatorVersion: "simple-semantic-loop-validator/1",
      valid: true,
      errors: [],
      checkedArtifactHashes: contract.requiredInputs.map(
        (artifact) => artifact.sha256,
      ),
      checkedObjectRefs: structuredClone(contract.scope),
    };
    const gateEvaluation = evaluateGate(
      gate,
      this.scriptGateContext(runId, scriptValidation),
    );
    if (!gateEvaluation.passed) {
      this.recordScriptGateFailure(
        runId,
        stageId,
        gate,
        gateEvaluation,
      );
      return;
    }
    this.store.casTransition(
      runId,
      expected,
      {
        canonicalRevisionDelta:
          domain.kind === "semantic_delta"
            ? 1
            : semanticObject
              ? 1
              : 0,
        lifecycle: "running",
        currentStageId: stageId,
        eventType: "script_stage_committed",
        eventPayload: {
          stageId,
          stageType: contract.stageType,
          proposalId: proposal.proposalId,
          semanticObject: semanticObject
            ? {
                objectType: semanticObject.objectType,
                objectId: semanticObject.objectId,
                revision: semanticObject.revision,
              }
            : null,
        },
      },
      (db, nextSnapshotVersion) => {
        if (semanticObject) {
          this.store.saveCanonicalObject(
            db,
            runId,
            semanticObject.objectType,
            semanticObject.objectId,
            semanticObject.revision,
            semanticObject.value,
            basisResults[0]?.resultId ?? null,
          );
        }
        if (domain.kind === "semantic_delta") {
          this.store.saveCanonicalObject(
            db,
            runId,
            "semantic_delta",
            domain.value.deltaId,
            1,
            { ...domain.value, status: "committed" },
            basisResults[0]?.resultId ?? null,
          );
          for (const resultRef of domain.value.basisResultRefs) {
            if (resultRef.objectType !== "turn_result") continue;
            const existing = db
              .prepare(
                "SELECT commit_id FROM result_consumptions WHERE result_id = ?",
              )
              .get(resultRef.objectId);
            if (!existing) {
              db.prepare(
                `INSERT INTO result_consumptions(
                   result_id, run_id, commit_id, delta_id,
                   consumed_snapshot_version, consumed_at
                 ) VALUES (?, ?, ?, ?, ?, ?)`,
              ).run(
                resultRef.objectId,
                runId,
                `consume-${randomUUID()}`,
                domain.value.deltaId,
                nextSnapshotVersion,
                timestamp(),
              );
            }
          }
          this.applySemanticResultSideEffects(
            db,
            runId,
            domain.value,
            basisResults,
          );
        }
        db.prepare(
          `UPDATE workflow_plan_nodes SET lifecycle = 'consumed'
           WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
        ).run(runId, expected.workflowPlanRevision, stageId);
        db.prepare(
          `INSERT INTO gate_results(
             gate_result_id, run_id, stage_id, gate_id, result_id, passed,
             report_json, created_at
           ) VALUES (?, ?, ?, ?, NULL, 1, ?, ?)`,
        ).run(
          `gate-result-${randomUUID()}`,
          runId,
          stageId,
          gate.gateId,
          canonicalJson(gateEvaluation),
          timestamp(),
        );
      },
    );
  }

  private loadBasisTurnResults(
    runId: string,
    delta: SemanticDelta,
  ): Array<{
    resultId: string;
    messageType: string;
    envelope: EvidencePacketEnvelope | ReviewDeltaEnvelope;
  }> {
    return delta.basisResultRefs.map((ref) => {
      if (ref.objectType !== "turn_result" || ref.revision !== 1) {
        throw new Error("SemanticDelta basis must use turn_result revision 1");
      }
      const row = this.store.db
        .prepare(
          `SELECT r.result_id, r.message_type, r.payload_json, c.result_id AS consumed
           FROM turn_results r
           LEFT JOIN result_consumptions c ON c.result_id = r.result_id
           WHERE r.run_id = ? AND r.result_id = ? AND r.status = 'committed'`,
        )
        .get(runId, ref.objectId) as
        | {
            result_id: string;
            message_type: string;
            payload_json: string;
            consumed: string | null;
          }
        | undefined;
      if (!row || row.consumed) {
        throw new Error(`basis result is absent or already consumed: ${ref.objectId}`);
      }
      if (!["EVIDENCE_PACKET", "REVIEW_DELTA"].includes(row.message_type)) {
        throw new Error(
          `SemanticDelta cannot consume ${row.message_type}`,
        );
      }
      return {
        resultId: row.result_id,
        messageType: row.message_type,
        envelope: JSON.parse(row.payload_json),
      };
    });
  }

  private applySemanticResultSideEffects(
    db: DatabaseSync,
    runId: string,
    delta: SemanticDelta,
    basisResults: Array<{
      resultId: string;
      messageType: string;
      envelope: EvidencePacketEnvelope | ReviewDeltaEnvelope;
    }>,
  ): void {
    for (const basis of basisResults) {
      if (basis.messageType === "EVIDENCE_PACKET") {
        const packet = basis.envelope as EvidencePacketEnvelope;
        const row = db
          .prepare(
            `SELECT object_json FROM canonical_objects
             WHERE run_id = ? AND object_type = 'search_need'
             AND object_id = ? AND active = 1`,
          )
          .get(runId, packet.payload.needId) as
          | { object_json: string }
          | undefined;
        if (!row) {
          throw new Error(
            `EvidencePacket SearchNeed is not canonical: ${packet.payload.needId}`,
          );
        }
        const need = JSON.parse(row.object_json) as SearchNeed;
        if (need.revision !== packet.payload.needRevision) {
          throw new Error("EvidencePacket Need revision became stale");
        }
        const nextNeed: SearchNeed = {
          ...need,
          revision: need.revision + 1,
          previousAttemptIds: [
            ...new Set([
              ...need.previousAttemptIds,
              packet.attemptId,
            ]),
          ],
          status:
            packet.payload.conclusion === "answered"
              ? "answered"
              : packet.payload.conclusion === "not_found"
                ? "no_delta"
                : "pending",
        };
        this.store.saveCanonicalObject(
          db,
          runId,
          "search_need",
          nextNeed.needId,
          nextNeed.revision,
          nextNeed,
          basis.resultId,
        );
        for (const contradiction of packet.payload.contradictions) {
          const existing = db
            .prepare(
              `SELECT object_json FROM canonical_objects
               WHERE run_id = ? AND object_type = 'contradiction'
               AND object_id = ? AND active = 1`,
            )
            .get(runId, contradiction.contradictionId);
          if (existing) {
            throw new Error(
              `duplicate active contradiction ${contradiction.contradictionId}`,
            );
          }
          const objectRef = {
            objectType: "contradiction",
            objectId: contradiction.contradictionId,
            revision: 1,
          };
          this.store.saveCanonicalObject(
            db,
            runId,
            "contradiction",
            contradiction.contradictionId,
            1,
            {
              ...contradiction,
              objectRef,
              ownerDirectionId: need.owner.directionId,
              targetObjectRef:
                contradiction.target.kind === "object"
                  ? contradiction.target.objectRef
                  : null,
              dispositionReviewId: null,
              disposition: null,
            },
            basis.resultId,
          );
        }
        if (
          need.intent === "discover_anchor" &&
          packet.payload.conclusion === "not_found" &&
          delta.action === "no_semantic_delta"
        ) {
          const noDeltaId = `no-delta-${delta.deltaId}`;
          this.store.saveCanonicalObject(
            db,
            runId,
            "no_delta",
            noDeltaId,
            1,
            {
              noDeltaRecordId: noDeltaId,
              needId: need.needId,
              needRevision: need.revision,
              resultId: basis.resultId,
              reason: packet.payload.conclusionRationale,
            },
            basis.resultId,
          );
          const expansionId =
            `topic-expansion-${need.needId}-${need.revision}`;
          this.store.saveCanonicalObject(
            db,
            runId,
            "topic_expansion",
            expansionId,
            1,
            {
              needId: need.needId,
              needRevision: need.revision,
              intent: "discover_anchor",
              ownerTopicId: need.owner.topicId,
              completed: true,
              outcome: "no_new_anchor_no_critical_delta",
              noDeltaRecordId: noDeltaId,
              semanticDeltaId: null,
            },
            basis.resultId,
          );
        }
      } else {
        const review = basis.envelope as ReviewDeltaEnvelope;
        if (review.payload.experimentHandoff) {
          const handoff = review.payload.experimentHandoff;
          this.store.saveCanonicalObject(
            db,
            runId,
            "experiment_handoff",
            handoff.handoffId,
            1,
            { ...handoff, complete: true },
            basis.resultId,
          );
        }
        const contradictions = db
          .prepare(
            `SELECT object_id, revision, object_json
             FROM canonical_objects
             WHERE run_id = ? AND object_type = 'contradiction'
             AND active = 1`,
          )
          .all(runId) as Array<{
          object_id: string;
          revision: number;
          object_json: string;
        }>;
        for (const row of contradictions) {
          const contradiction = JSON.parse(
            row.object_json,
          ) as Record<string, unknown>;
          if (
            contradiction.ownerDirectionId !==
              review.payload.directionId ||
            contradiction.dispositionReviewId
          ) {
            continue;
          }
          this.store.saveCanonicalObject(
            db,
            runId,
            "contradiction",
            row.object_id,
            row.revision + 1,
            {
              ...contradiction,
              objectRef: {
                objectType: "contradiction",
                objectId: row.object_id,
                revision: row.revision + 1,
              },
              dispositionReviewId: review.payload.reviewId,
              disposition: review.payload.decision,
            },
            basis.resultId,
          );
        }
      }
    }
  }

  /**
   * Returns true when it committed the SearchNeed as this loop's one state
   * transition, so dispatch must wait for the next loop.
   */
  private ensureSearchNeedCanonical(
    runId: string,
    contract: StageContract,
  ): boolean {
    const proposal = this.proposalForContract(runId, contract);
    if (proposal.domainProposal?.kind !== "search_need") {
      const needRefs = contract.scope.filter(
        (ref) => ref.objectType === "search_need",
      );
      if (needRefs.length !== 1) {
        throw new Error(
          "Evidence Stage must own a new SearchNeed or scope exactly one current SearchNeed",
        );
      }
      const needRef = needRefs[0]!;
      const existing = this.store.db
        .prepare(
          `SELECT revision, object_json FROM canonical_objects
           WHERE run_id = ? AND object_type = 'search_need'
             AND object_id = ? AND active = 1`,
        )
        .get(runId, needRef.objectId) as
        | { revision: number; object_json: string }
        | undefined;
      const status = existing
        ? (JSON.parse(existing.object_json) as { status?: string }).status
        : null;
      if (
        !existing ||
        existing.revision !== needRef.revision ||
        status !== "pending"
      ) {
        throw new Error(
          "Evidence Stage scoped SearchNeed is absent, stale, or non-pending",
        );
      }
      return false;
    }
    const need = proposal.domainProposal.value;
    const existing = this.store.db
      .prepare(
        `SELECT revision FROM canonical_objects WHERE run_id = ?
         AND object_type = 'search_need' AND object_id = ? AND active = 1`,
      )
      .get(runId, need.needId) as { revision: number } | undefined;
    if (existing) {
      if (existing.revision !== need.revision) {
        throw new Error("active SearchNeed revision differs from frozen Stage");
      }
      return false;
    }
    const expected = this.store.stateBinding(runId);
    this.store.casTransition(
      runId,
      expected,
      {
        canonicalRevisionDelta: 1,
        lifecycle: "running",
        currentStageId: contract.stageId,
        eventType: "search_need_committed",
        eventPayload: { needId: need.needId, revision: need.revision },
      },
      (db) =>
        this.store.saveCanonicalObject(
          db,
          runId,
          "search_need",
          need.needId,
          need.revision,
          need,
          null,
        ),
    );
    return true;
  }

  private scheduleWorkflowDecision(
    runId: string,
    trigger: NonNullable<ReturnType<typeof buildRegisteredTrigger>>,
  ): void {
    const pair = this.createControllerStage(
      runId,
      "WORKFLOW_DECISION",
      "workflow_decision",
      this.config.budgets.workflow,
    );
    const projection = this.workflowDomainProjection(runId);
    const permission = defaultWorkflowPermission(trigger, {
      workflow: this.config.budgets.workflow,
      evidence: this.config.budgets.evidence,
      direction: this.config.budgets.direction,
    });
    permission.suppliedObjectRefs = this.activeObjectRefs(runId);
    permission.suppliedResultRefs =
      this.committedResultRefs(runId);
    const task = this.taskFactory.buildWorkflowTask(runId, trigger, {
      stageId: pair.contract.stageId,
      contract: pair.contract,
      permission,
      domainProjection: projection,
    });
    this.insertControllerStageAndTask(
      runId,
      pair.contract,
      pair.gate,
      task,
    );
  }

  private scheduleClosureReview(runId: string): void {
    const existing = this.store.db
      .prepare(
        `SELECT task_id FROM tasks WHERE run_id = ? AND role = 'closure_reviewer'
         AND status IN ('pending', 'pending_output_retry', 'dispatched') LIMIT 1`,
      )
      .get(runId);
    if (existing) return;
    const pair = this.createControllerStage(
      runId,
      "CLOSURE_REVIEW",
      "closure_reviewer",
      this.config.budgets.closure,
    );
    const task = this.taskFactory.buildStageTask(runId, {
      stageId: pair.contract.stageId,
      stageType: "CLOSURE_REVIEW",
      role: "closure_reviewer",
      contract: pair.contract,
      gate: pair.gate,
    }) as ClosureReviewTaskEnvelope;
    const preflight = runClosurePreflight(task);
    if (!preflight.validation.valid) {
      this.store.casTransition(runId, this.store.stateBinding(runId), {
        lifecycle: "running",
        eventType: "closure_preflight_rejected",
        eventPayload: preflight.report,
      });
      return;
    }
    task.payload.mechanicalPreflight = preflight.report;
    const bound = bindPayloadTaskHash(task);
    this.insertControllerStageAndTask(
      runId,
      pair.contract,
      pair.gate,
      bound,
    );
  }

  private createControllerStage(
    runId: string,
    stageType: "WORKFLOW_DECISION" | "CLOSURE_REVIEW",
    role: "workflow_decision" | "closure_reviewer",
    budget: TurnBudget,
  ) {
    const registered = STAGE_REGISTRY[stageType];
    const draft: StageContractDraft = {
      proposalLocalStageKey: `controller-${stageType.toLowerCase()}-${randomUUID()}`,
      stageType,
      objective:
        stageType === "WORKFLOW_DECISION"
          ? "Propose one bounded workflow action for the registered trigger."
          : "Independently review one StopCandidate against canonical closure facts.",
      scope: [],
      executionKind: registered.executionKind,
      role,
      requiredInputs: [],
      expectedOutputMessageType: registered.output,
      requestedTools: [],
      requestedPaths: [],
      prohibitedActions: [
        "state write",
        "agent delegation",
        "Goal continuation",
        "experiment execution",
      ],
      budget,
    };
    return freezeStageDraft(
      draft,
      {
        proposalLocalStageKey: draft.proposalLocalStageKey,
        mechanicalChecks: [],
        semanticEvaluation: {
          required: false,
          evaluatorRole: null,
          rubricId: null,
          inputProjection: [],
          expectedOutputMessageType: null,
        },
      },
      this.store.getRun(runId).snapshotVersion,
    );
  }

  private insertControllerStageRows(
    runId: string,
    contract: StageContract,
    gate: GateDefinition,
  ): void {
    if (
      contract.definedAtSnapshotVersion !==
        this.store.getRun(runId).snapshotVersion ||
      gate.definedAtSnapshotVersion !==
        this.store.getRun(runId).snapshotVersion
    ) {
      throw new Error("Controller Stage/Gate were not frozen against current state");
    }
    const now = timestamp();
    this.store.db
      .prepare(
        `INSERT INTO stage_contracts(
           contract_id, run_id, stage_id, revision, stage_type, role,
           defined_at_snapshot_version, sha256, contract_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
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
    this.store.db
      .prepare(
        `INSERT INTO gate_definitions(
           gate_id, run_id, stage_id, revision, defined_at_snapshot_version,
           sha256, gate_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        gate.gateId,
        runId,
        gate.stageId,
        gate.revision,
        gate.definedAtSnapshotVersion,
        gate.sha256,
        canonicalJson(gate),
        now,
      );
  }

  private insertControllerStageAndTask(
    runId: string,
    contract: StageContract,
    gate: GateDefinition,
    task: AnyTurnTask,
  ): void {
    this.store.db.exec("BEGIN IMMEDIATE");
    try {
      this.insertControllerStageRows(runId, contract, gate);
      this.insertTask(runId, task, contract);
      this.store.db.exec("COMMIT");
    } catch (error) {
      this.store.db.exec("ROLLBACK");
      throw error;
    }
  }

  private insertTask(
    runId: string,
    task: AnyTurnTask,
    contract: StageContract,
  ): void {
    const role = contract.role;
    if (!role) throw new Error("cannot create Agent task for role=null");
    this.store.createTask(runId, {
      taskId: task.taskId,
      stageId: task.stageId,
      role,
      inputMessageType: ROLE_MESSAGE_TYPES[role].input,
      expectedOutputMessageType: ROLE_MESSAGE_TYPES[role].output,
      stateBinding: getTaskStateBinding(task),
      inputHash: getTaskInputHash(task),
      stageContractHash: contract.sha256,
      skillHash: getSkillHash(task),
      schemaManifestHash: getSchemaManifestHash(task),
      task,
    });
  }

  private scriptGateContext(
    runId: string,
    validationReport: ValidationReport,
  ): GateEvaluationContext {
    return {
      resolve: (actual) => {
        switch (actual.source) {
          case "result":
          case "task":
            return unresolved(
              "gate.operand_unavailable_for_script",
              `${actual.source} operand is unavailable for script transitions`,
            );
          case "canonical":
            return this.resolveCanonicalGateActual(runId, actual);
          case "artifact":
            return this.resolveArtifactGateActual(runId, actual);
          case "runtime":
            switch (actual.fact) {
              case "allowed_tool_events_only":
              case "allowed_paths_only":
              case "turn_budget_within_contract":
                return resolvedValue(
                  true,
                  actual.valueType,
                  `script runtime fact ${actual.fact}`,
                );
              case "experiment_execution_count":
                return resolvedValue(
                  0,
                  actual.valueType,
                  "script transition executes no experiment capability",
                );
              case "external_evidence_used":
                return resolvedValue(
                  false,
                  actual.valueType,
                  "script transition reads no external evidence",
                );
            }
          case "validator":
            if (actual.fact !== "script_transition_valid") {
              return unresolved(
                "gate.validator_fact_unavailable_for_script",
                `${actual.fact} is not defined for script transitions`,
              );
            }
            return resolvedValue(
              validationReport.valid,
              actual.valueType,
              "Controller script-transition validator",
            );
        }
      },
    };
  }

  private recordScriptGateFailure(
    runId: string,
    stageId: string,
    gate: GateDefinition,
    evaluation: ReturnType<typeof evaluateGate>,
  ): void {
    const expected = this.store.stateBinding(runId);
    this.store.casTransition(
      runId,
      expected,
      {
        lifecycle: "running",
        currentStageId: stageId,
        eventType: "stage_gate_failed",
        eventPayload: { stageId, evaluation, executionKind: "SCRIPT_TRANSITION" },
      },
      (db) => {
        db.prepare(
          `INSERT INTO gate_results(
             gate_result_id, run_id, stage_id, gate_id, result_id, passed,
             report_json, created_at
           ) VALUES (?, ?, ?, ?, NULL, 0, ?, ?)`,
        ).run(
          `gate-result-${randomUUID()}`,
          runId,
          stageId,
          gate.gateId,
          canonicalJson(evaluation),
          timestamp(),
        );
        db.prepare(
          `UPDATE workflow_plan_nodes SET lifecycle = 'failed'
           WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
        ).run(runId, expected.workflowPlanRevision, stageId);
      },
    );
  }

  private gateContext(
    runId: string,
    gate: GateDefinition,
    result: unknown,
    report: ValidationReport,
    rawTurn: NonNullable<
      Awaited<ReturnType<typeof dispatchFreshTurnAttempt>>["rawTurn"]
    >,
    role: RegisteredRole,
    task: AnyTurnTask,
    securityReport: ValidationReport,
  ): GateEvaluationContext {
    void gate;
    return {
      resolve: (actual) =>
        this.resolveGateActual(
          runId,
          actual,
          result,
          report,
          rawTurn,
          role,
          task,
          securityReport,
        ),
    };
  }

  private resolveGateActual(
    runId: string,
    actual: GateActual,
    result: unknown,
    report: ValidationReport,
    rawTurn: NonNullable<
      Awaited<ReturnType<typeof dispatchFreshTurnAttempt>>["rawTurn"]
    >,
    role: RegisteredRole,
    task: AnyTurnTask,
    securityReport: ValidationReport,
  ): GateResolution {
    switch (actual.source) {
      case "result":
        return resolvedPointer(
          result,
          actual.pointer,
          actual.valueType,
          "result",
        );
      case "task":
        return resolvedPointer(
          task,
          actual.pointer,
          actual.valueType,
          "task",
        );
      case "canonical": {
        return this.resolveCanonicalGateActual(runId, actual);
      }
      case "runtime": {
        const admitted =
          securityReport.valid &&
          validateRuntimeToolEvents(role, rawTurn.toolEvents).valid;
        switch (actual.fact) {
          case "allowed_tool_events_only":
          case "allowed_paths_only":
            return resolvedValue(
              admitted,
              actual.valueType,
              `runtime fact ${actual.fact}`,
            );
          case "turn_budget_within_contract":
            return resolvedValue(
              true,
              actual.valueType,
              "Turn budget was validated before Gate evaluation",
            );
          case "experiment_execution_count":
            return resolvedValue(
              rawTurn.toolEvents.filter((event) =>
                isForbiddenExecutionCapability(event.toolName),
              ).length,
              actual.valueType,
              "runtime forbidden-execution event count",
            );
          case "external_evidence_used":
            return resolvedValue(
              rawTurn.toolEvents.some(
                (event) =>
                  !event.toolName.startsWith("mcp__obsidian__"),
              ),
              actual.valueType,
              "runtime non-Obsidian tool observation",
            );
        }
      }
      case "validator": {
        switch (actual.fact) {
          case "schema_valid":
          case "message_binding_matches":
          case "registered_validator_passes":
            return resolvedValue(
              report.valid,
              actual.valueType,
              `validator fact ${actual.fact}`,
            );
          case "references_resolve": {
            if (actual.pointer === null) {
              return unresolved(
                "gate.validator_pointer_missing",
                "references_resolve requires a result pointer",
              );
            }
            const pointer = resolveJsonPointerValue(
              result,
              actual.pointer,
            );
            return resolvedValue(
              report.valid && pointer.found,
              actual.valueType,
              "domain validator reference resolution",
            );
          }
          case "source_context_present": {
            if (actual.pointer === null) {
              return unresolved(
                "gate.validator_pointer_missing",
                "source_context_present requires a result pointer",
              );
            }
            return resolvedValue(
              report.valid &&
                evidenceSourceContextsPresent(result, actual.pointer),
              actual.valueType,
              "finding-to-context provenance",
            );
          }
          case "duplicate_commit": {
            const duplicate = Boolean(
              this.store.db
                .prepare(
                  `SELECT 1 FROM turn_results
                   WHERE task_id = ? AND payload_hash = ? LIMIT 1`,
                )
                .get(task.taskId, canonicalSha256(result)),
            );
            return resolvedValue(
              duplicate,
              actual.valueType,
              "committed result payload lookup",
            );
          }
          case "script_transition_valid":
            return resolvedValue(
              true,
              actual.valueType,
              "script transition validation is enforced before commit",
            );
        }
      }
      case "artifact": {
        return this.resolveArtifactGateActual(runId, actual);
      }
    }
  }

  private resolveCanonicalGateActual(
    runId: string,
    actual: Extract<GateActual, { source: "canonical" }>,
  ): GateResolution {
    const row = this.store.db
      .prepare(
        `SELECT revision, object_json FROM canonical_objects
         WHERE run_id = ? AND object_type = ? AND object_id = ?
           AND active = 1`,
      )
      .get(
        runId,
        actual.objectRef.objectType,
        actual.objectRef.objectId,
      ) as { revision: number; object_json: string } | undefined;
    if (!row) {
      return unresolved(
        "gate.canonical_object_missing",
        `canonical ${actual.objectRef.objectType}:${actual.objectRef.objectId} is absent`,
      );
    }
    if (row.revision !== actual.objectRef.revision) {
      return unresolved(
        "gate.canonical_revision_stale",
        `canonical revision ${row.revision} differs from frozen ${actual.objectRef.revision}`,
      );
    }
    return resolvedPointer(
      JSON.parse(row.object_json),
      actual.pointer,
      actual.valueType,
      "canonical object",
    );
  }

  private resolveArtifactGateActual(
    runId: string,
    actual: Extract<GateActual, { source: "artifact" }>,
  ): GateResolution {
    const artifact = this.store.db
      .prepare(
        `SELECT sha256 FROM artifact_manifests
         WHERE run_id = ? AND artifact_id = ?`,
      )
      .get(runId, actual.artifactId) as
      | { sha256: string }
      | undefined;
    if (!artifact) {
      return actual.fact === "exists"
        ? resolvedValue(
            false,
            actual.valueType,
            `artifact ${actual.artifactId} is absent`,
          )
        : unresolved(
            "gate.artifact_missing",
            `artifact ${actual.artifactId} is absent`,
          );
    }
    try {
      this.readVerifiedArtifact(runId, actual.artifactId);
    } catch (error) {
      return unresolved(
        "gate.artifact_integrity_failed",
        error instanceof Error ? error.message : String(error),
      );
    }
    return resolvedValue(
      actual.fact === "exists" ? true : artifact.sha256,
      actual.valueType,
      `verified artifact fact ${actual.fact}`,
    );
  }

  private recordGateFailure(
    runId: string,
    row: TaskRow,
    attemptId: string,
    gate: GateDefinition,
    evaluation: ReturnType<typeof evaluateGate>,
    rawArtifactId: string,
  ): void {
    const expected = this.store.stateBinding(runId);
    this.store.casTransition(
      runId,
      expected,
      {
        lifecycle: "running",
        currentStageId: row.stage_id,
        eventType: "stage_gate_failed",
        eventPayload: {
          taskId: row.task_id,
          stageId: row.stage_id,
          attemptId,
          evaluation,
        },
      },
      (db) => {
        db.prepare(
          `INSERT INTO gate_results(
             gate_result_id, run_id, stage_id, gate_id, result_id, passed,
             report_json, created_at
           ) VALUES (?, ?, ?, ?, NULL, 0, ?, ?)`,
        ).run(
          `gate-result-${randomUUID()}`,
          runId,
          row.stage_id,
          gate.gateId,
          canonicalJson(evaluation),
          timestamp(),
        );
        db.prepare(
          `UPDATE workflow_plan_nodes SET lifecycle = 'failed'
           WHERE run_id = ? AND plan_revision = ? AND stage_id = ?`,
        ).run(runId, expected.workflowPlanRevision, row.stage_id);
        db.prepare(
          "UPDATE tasks SET status = 'failed', updated_at = ? WHERE task_id = ?",
        ).run(timestamp(), row.task_id);
        db.prepare(
          `UPDATE attempts
           SET status = 'gate_failed', raw_response_artifact_id = ?,
               error_code = 'gate.failed', finished_at = ?
           WHERE attempt_id = ?`,
        ).run(rawArtifactId, timestamp(), attemptId);
      },
    );
  }

  private afterClosureReview(
    runId: string,
    expected: StateBinding,
    review: ClosureReviewEnvelope,
    resultId: string,
  ): void {
    const lifecycle =
      review.payload.decision === "accept" ? "finalizing" : "running";
    const eventType =
      review.payload.decision === "accept"
        ? "closure_review_accepted"
        : "closure_review_rejected";
    this.store.casTransition(
      runId,
      expected,
      {
        lifecycle,
        eventType,
        eventPayload:
          review.payload.decision === "accept"
            ? {
                reviewId: review.payload.reviewId,
                canonicalRevision: review.payload.canonicalRevision,
              }
            : {
                reviewId: review.payload.reviewId,
                reopenScopes: review.payload.reopenScopes,
              },
      },
      (db, nextSnapshotVersion) => {
        db.prepare(
          `INSERT OR IGNORE INTO result_consumptions(
             result_id, run_id, commit_id, delta_id,
             consumed_snapshot_version, consumed_at
           ) VALUES (?, ?, ?, NULL, ?, ?)`,
        ).run(
          resultId,
          runId,
          `consume-closure-${review.payload.reviewId}`,
          nextSnapshotVersion,
          timestamp(),
        );
        if (review.payload.decision === "reject") {
          db.prepare(
            `UPDATE canonical_objects SET active = 0
             WHERE run_id = ?
               AND object_type IN ('stop_candidate', 'stop_proof')
               AND active = 1`,
          ).run(runId);
        }
      },
    );
  }

  private finalize(runId: string): void {
    const row = this.store.db
      .prepare(
        `SELECT r.payload_json, t.task_json
         FROM turn_results r JOIN tasks t ON t.task_id = r.task_id
         WHERE r.run_id = ? AND r.message_type = 'CLOSURE_REVIEW'
         AND r.status = 'committed'
         ORDER BY r.committed_at DESC LIMIT 1`,
      )
      .get(runId) as
      | { payload_json: string; task_json: string }
      | undefined;
    if (!row) {
      this.failRun(
        runId,
        "finalization.closure_result_missing",
        "Finalizing lifecycle has no committed ClosureReview.",
        true,
      );
      return;
    }
    const closure = JSON.parse(row.payload_json) as ClosureReviewEnvelope;
    const task = JSON.parse(row.task_json) as ClosureReviewTaskEnvelope;
    const final = finalizeAcceptedClosure(
      this.store,
      this.config.workDir,
      task,
      closure,
    );
    if (!final.completed) {
      this.failRun(
        runId,
        "finalization.coverage_failed",
        final.coverageErrors.join("; "),
        true,
      );
    }
  }

  private buildTriggerSignals(
    runId: string,
    runnableCount: number,
  ): TriggerSignals {
    const hasTopicFrame = Boolean(
      this.store.db
        .prepare(
          `SELECT 1 FROM canonical_objects WHERE run_id = ?
           AND object_type = 'topic' AND active = 1 LIMIT 1`,
        )
        .get(runId),
    );
    const state = this.store.readWorkflowState(runId);
    const gateFailed = Boolean(
      this.store.db
        .prepare(
          `SELECT 1
           FROM gate_results g
           JOIN workflow_plan_nodes n
             ON n.run_id = g.run_id
            AND n.stage_id = g.stage_id
           WHERE g.run_id = ?
             AND n.plan_revision = ?
             AND n.lifecycle IN ('failed', 'blocked')
             AND g.passed = 0
           LIMIT 1`,
        )
        .get(runId, state.workflowPlanRevision),
    );
    const criticalContradiction =
      this.hasUnresolvedCriticalContradiction(runId);
    const closureRejected =
      this.closureReviewRejectedUnresolved(runId);
    const frontierCount = this.store
      .query(
        `SELECT object_type, object_json FROM canonical_objects
         WHERE run_id = ? AND active = 1
         AND object_type IN ('anchor', 'search_need')`,
        runId,
      )
      .filter((row) => {
        const value = JSON.parse(String(row.object_json)) as {
          status?: string;
        };
        return row.object_type === "anchor"
          ? value.status === "active"
          : value.status === "pending";
      }).length;
    const pending = state.pendingTaskIds.length + state.inFlightTaskIds.length > 0;
    const plan = loadCurrentPlan(this.store, runId);
    const activePlanWork = plan.stageNodes.some((stage) =>
      ["frozen", "runnable", "dispatched", "gate_running"].includes(
        stage.lifecycle,
      ),
    );
    return {
      hasTopicFrame,
      committedResultRequiresIntegration:
        state.committedUnconsumedResultIds.length > 0,
      nonEquivalentFrontierCount: frontierCount,
      nonEquivalentRunnableStageCount: runnableCount,
      gateFailedWithoutRecovery: gateFailed,
      planExhaustedObjectiveOpen:
        hasTopicFrame && !activePlanWork && !pending,
      criticalEvidenceContradiction: criticalContradiction,
      noProgressThresholdReached:
        (state.noProgressCounters
          .semanticTransitionsWithoutCanonicalDelta ?? 0) >=
        this.config.noProgressThreshold,
      closureRejected,
      userDecisionRequired: false,
      hasPendingOrInFlight: pending,
      hasRunnableStage: runnableCount > 0,
    };
  }

  private workflowDomainProjection(
    runId: string,
  ): WorkflowTurnTask["domainProjection"] {
    const active = <T>(type: string): T[] =>
      this.store
        .query(
          `SELECT object_json FROM canonical_objects
           WHERE run_id = ? AND object_type = ? AND active = 1
           ORDER BY object_id LIMIT 20`,
          runId,
          type,
        )
        .map((row) => JSON.parse(String(row.object_json)) as T);
    const evidencePackets = this.store
      .query(
        `SELECT payload_json FROM turn_results WHERE run_id = ?
         AND message_type = 'EVIDENCE_PACKET' AND status = 'committed'
         ORDER BY committed_at DESC LIMIT 10`,
        runId,
      )
      .map((row) => {
        const envelope = JSON.parse(String(row.payload_json));
        return envelope.payload ?? envelope;
      });
    const reviews = this.store
      .query(
        `SELECT payload_json FROM turn_results WHERE run_id = ?
         AND message_type = 'REVIEW_DELTA' AND status = 'committed'
         ORDER BY committed_at DESC LIMIT 10`,
        runId,
      )
      .map((row) => {
        const envelope = JSON.parse(String(row.payload_json));
        return envelope.payload ?? envelope;
      });
    const stopCandidate = active<Record<string, unknown>>("stop_candidate")[0];
    const stopProof = active<Record<string, unknown>>("stop_proof")[0];
    return {
      topic: active("topic")[0] ?? null,
      focusAnchor: active("anchor")[0] ?? null,
      focusDirection: active("direction")[0] ?? null,
      searchNeeds: active("search_need"),
      evidencePackets,
      directionReviews: reviews,
      stopCandidateBundle:
        stopCandidate && stopProof
          ? ({ candidate: stopCandidate, proof: stopProof } as never)
          : null,
      completionProjection:
        this.taskFactory.buildCompletionProjection(runId),
    };
  }

  private activeObjectRefs(runId: string) {
    return this.store
      .query(
        `SELECT object_type, object_id, revision FROM canonical_objects
         WHERE run_id = ? AND active = 1 ORDER BY object_type, object_id`,
        runId,
      )
      .map((row) => ({
        objectType: String(row.object_type),
        objectId: String(row.object_id),
        revision: Number(row.revision),
      }));
  }

  private committedResultRefs(runId: string) {
    return this.store
      .query(
        `SELECT result_id FROM turn_results
         WHERE run_id = ? AND status = 'committed'
         ORDER BY result_id`,
        runId,
      )
      .map((row) => ({
        objectType: "turn_result",
        objectId: String(row.result_id),
        revision: 1,
      }));
  }

  private closureReviewRejectedUnresolved(runId: string): boolean {
    const rejection = this.store.db
      .prepare(
        `SELECT event_cursor FROM events
         WHERE run_id = ? AND event_type = 'closure_review_rejected'
         ORDER BY event_cursor DESC LIMIT 1`,
      )
      .get(runId) as { event_cursor: number } | undefined;
    if (!rejection) return false;
    return !this.store
      .query(
        `SELECT p.expected_state_json, t.task_json
         FROM decision_proposals p
         JOIN tasks t ON t.task_id = p.task_id
         WHERE p.run_id = ? AND p.status = 'accepted'
         ORDER BY p.created_at DESC`,
        runId,
      )
      .some((row) => {
        const task = JSON.parse(String(row.task_json)) as {
          trigger?: string;
        };
        const expected = JSON.parse(
          String(row.expected_state_json),
        ) as StateBinding;
        return (
          task.trigger === "CLOSURE_REJECTED" &&
          expected.eventCursor >= rejection.event_cursor
        );
      });
  }

  private hasUnresolvedCriticalContradiction(runId: string): boolean {
    const canonical = this.store
      .query(
        `SELECT object_json FROM canonical_objects
         WHERE run_id = ? AND object_type = 'contradiction' AND active = 1`,
        runId,
      )
      .some((row) => {
        const value = JSON.parse(String(row.object_json)) as {
          critical?: boolean;
          dispositionReviewId?: string | null;
        };
        return value.critical === true && !value.dispositionReviewId;
      });
    if (canonical) return true;
    return this.store
      .query(
        `SELECT r.payload_json
         FROM turn_results r
         LEFT JOIN result_consumptions c ON c.result_id = r.result_id
         WHERE r.run_id = ?
           AND r.message_type = 'EVIDENCE_PACKET'
           AND r.status = 'committed'
           AND c.result_id IS NULL`,
        runId,
      )
      .some((row) => {
        const envelope = JSON.parse(String(row.payload_json));
        return (envelope.payload?.contradictions ?? []).some(
          (item: { critical?: boolean }) => item.critical === true,
        );
      });
  }

  private promoteRunnableNodes(
    runId: string,
    planRevision: number,
    runnable: ReturnType<typeof calculateRunnableStages>,
  ): void {
    for (const stage of runnable) {
      if (stage.lifecycle === "frozen") {
        this.store.db
          .prepare(
            `UPDATE workflow_plan_nodes SET lifecycle = 'runnable'
             WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
             AND lifecycle = 'frozen'`,
          )
          .run(runId, planRevision, stage.stageId);
      }
    }
  }

  private proposalForContract(
    runId: string,
    contract: StageContract,
  ): WorkflowDecisionProposal {
    for (const row of this.store.query(
      `SELECT proposal_json FROM decision_proposals
       WHERE run_id = ? AND status = 'accepted' ORDER BY created_at DESC`,
      runId,
    )) {
      const proposal = JSON.parse(
        String(row.proposal_json),
      ) as WorkflowDecisionProposal;
      if (
        proposal.proposedStageContract?.proposalLocalStageKey ===
        contract.proposalLocalStageKey
      ) {
        return proposal;
      }
      if (
        proposal.proposedPlanPatch?.operations.some(
          (operation) =>
            operation.op === "add_stage" &&
            operation.stage.proposalLocalStageKey ===
              contract.proposalLocalStageKey,
        )
      ) {
        return proposal;
      }
    }
    throw new Error("Stage has no accepted owning proposal");
  }

  private loadContract(
    runId: string,
    stageId: string,
    expectedContractHash?: string,
  ): StageContract {
    type ContractRow = {
      contract_id: string;
      sha256: string;
      contract_json: string;
    };
    let row = this.store.db
      .prepare(
        `SELECT sc.contract_id, sc.sha256, sc.contract_json
         FROM workflow_plan_nodes node
         JOIN stage_contracts sc
           ON sc.run_id = node.run_id
          AND sc.contract_id = node.contract_id
         WHERE node.run_id = ? AND node.plan_revision = ?
           AND node.stage_id = ?
         LIMIT 1`,
      )
      .get(
        runId,
        this.store.getRun(runId).workflowPlanRevision,
        stageId,
      ) as ContractRow | undefined;
    if (!row && expectedContractHash !== undefined) {
      row = this.store.db
        .prepare(
          `SELECT contract_id, sha256, contract_json
           FROM stage_contracts
           WHERE run_id = ? AND stage_id = ? AND sha256 = ?
           LIMIT 1`,
        )
        .get(runId, stageId, expectedContractHash) as
        | ContractRow
        | undefined;
    }
    if (!row) throw new Error(`missing frozen StageContract for ${stageId}`);
    const contract = JSON.parse(row.contract_json) as StageContract;
    if (
      contract.contractId !== row.contract_id ||
      contract.stageId !== stageId ||
      contract.sha256 !== row.sha256 ||
      (expectedContractHash !== undefined &&
        contract.sha256 !== expectedContractHash) ||
      contract.sha256 !== canonicalSha256(withoutSha256(contract))
    ) {
      throw new Error(`frozen StageContract integrity failure for ${stageId}`);
    }
    return contract;
  }

  private loadGate(
    runId: string,
    stageId: string,
    expectedContractHash?: string,
  ): GateDefinition {
    type GateRow = {
      gate_id: string;
      sha256: string;
      gate_json: string;
      contract_sha256: string;
    };
    let row = this.store.db
      .prepare(
        `SELECT gd.gate_id, gd.sha256, gd.gate_json,
                sc.sha256 AS contract_sha256
         FROM workflow_plan_nodes node
         JOIN gate_definitions gd
          ON gd.run_id = node.run_id
          AND gd.gate_id = node.gate_id
         JOIN stage_contracts sc
           ON sc.run_id = node.run_id
          AND sc.contract_id = node.contract_id
         WHERE node.run_id = ? AND node.plan_revision = ?
           AND node.stage_id = ?
         LIMIT 1`,
      )
      .get(
        runId,
        this.store.getRun(runId).workflowPlanRevision,
        stageId,
      ) as GateRow | undefined;
    if (!row && expectedContractHash !== undefined) {
      row = this.store.db
        .prepare(
          `SELECT gd.gate_id, gd.sha256, gd.gate_json,
                  sc.sha256 AS contract_sha256
           FROM gate_definitions gd
           JOIN stage_contracts sc
             ON sc.run_id = gd.run_id
            AND sc.stage_id = gd.stage_id
            AND sc.revision = gd.revision
           WHERE gd.run_id = ? AND gd.stage_id = ?
             AND sc.sha256 = ?
           LIMIT 1`,
        )
        .get(runId, stageId, expectedContractHash) as
        | GateRow
        | undefined;
    }
    if (!row) throw new Error(`missing frozen GateDefinition for ${stageId}`);
    const gate = JSON.parse(row.gate_json) as GateDefinition;
    const legacyGate =
      !Object.hasOwn(gate, "stageContractHash") &&
      gate.mechanicalChecks.every(
        (check) => !Object.hasOwn(check, "actual"),
      );
    const currentGateBindingValid =
      legacyGate ||
      (gate.stageContractHash === row.contract_sha256 &&
        gate.compilerPolicyVersion ===
          GATE_COMPILER_POLICY_VERSION &&
        gate.evaluatorVersion === GATE_EVALUATOR_VERSION &&
        /^[a-f0-9]{64}$/.test(gate.proposedCriteriaSha256));
    if (
      gate.gateId !== row.gate_id ||
      gate.stageId !== stageId ||
      gate.sha256 !== row.sha256 ||
      gate.sha256 !== canonicalSha256(withoutSha256(gate)) ||
      (expectedContractHash !== undefined &&
        row.contract_sha256 !== expectedContractHash) ||
      !currentGateBindingValid
    ) {
      throw new Error(`frozen GateDefinition integrity failure for ${stageId}`);
    }
    return gate;
  }

  private insertControllerFailureEvent(
    runId: string,
    code: string,
    message: string,
    terminal: boolean,
  ): void {
    const expected = this.store.stateBinding(runId);
    this.store.casTransition(runId, expected, {
      lifecycle: terminal ? "failed_terminal" : "failed_retriable",
      pauseOrBlockReason: `${code}: ${message}`,
      eventType: "controller_failure",
      eventPayload: { code, message, terminal },
    });
  }

  private failRun(
    runId: string,
    code: string,
    message: string,
    terminal: boolean,
  ): void {
    this.insertControllerFailureEvent(runId, code, message, terminal);
  }

  private markTaskAndStageFailed(
    runId: string,
    taskId: string,
    stageId: string,
  ): void {
    this.store.db
      .prepare(
        `UPDATE tasks SET status = 'failed', updated_at = ?
         WHERE task_id = ?`,
      )
      .run(timestamp(), taskId);
    this.store.db
      .prepare(
        `UPDATE workflow_plan_nodes SET lifecycle = 'failed'
         WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
           AND lifecycle NOT IN ('completed', 'superseded')`,
      )
      .run(
        runId,
        this.store.getRun(runId).workflowPlanRevision,
        stageId,
      );
  }

  private saveRawTurn(
    runId: string,
    attemptId: string,
    raw: NonNullable<
      Awaited<ReturnType<typeof dispatchFreshTurnAttempt>>["rawTurn"]
    >,
  ): string {
    const relativePath = `raw_turns/${attemptId}.json`;
    const absolutePath = resolve(this.config.workDir, relativePath);
    mkdirSync(resolve(this.config.workDir, "raw_turns"), { recursive: true });
    atomicWriteJson(absolutePath, raw);
    const bytes = readFileSync(absolutePath);
    const artifactId = `raw-${attemptId}`;
    this.store.registerArtifact(runId, {
      artifactId,
      kind: "raw_turn",
      relativePath,
      sha256: sha256Bytes(bytes),
      sizeBytes: statSync(absolutePath).size,
      trustClass: "untrusted_log",
    });
    return artifactId;
  }

  private savePrompt(
    runId: string,
    attemptId: string,
    prompt: string,
    expectedSha256: string,
  ): string {
    const relativePath = `prompts/${attemptId}.txt`;
    const absolutePath = resolve(this.config.workDir, relativePath);
    mkdirSync(resolve(this.config.workDir, "prompts"), { recursive: true });
    const actualSha256 = sha256Bytes(prompt);
    if (actualSha256 !== expectedSha256) {
      throw new Error("prompt hash changed between build and persistence");
    }
    atomicWriteText(absolutePath, prompt);
    const artifactId = `prompt-${attemptId}`;
    this.store.registerArtifact(runId, {
      artifactId,
      kind: "turn_prompt",
      relativePath,
      sha256: actualSha256,
      sizeBytes: statSync(absolutePath).size,
      trustClass: "controller_generated",
    });
    return artifactId;
  }

  /**
   * Recover a provider-completed raw Turn captured before the Controller could
   * finish validation/Gate commit. Recovery is local-only: it never resumes a
   * provider thread and never creates a replacement attempt.
   */
  private recoverCapturedRawAttempts(runId: string): void {
    const rows = this.store.query(
      `SELECT a.attempt_id, a.attempt_no, a.provider_thread_id,
              a.provider_turn_id, t.*
       FROM attempts a
       JOIN tasks t ON t.task_id = a.task_id
       WHERE a.run_id = ?
         AND a.status IN ('created', 'running')
         AND EXISTS (
           SELECT 1 FROM artifact_manifests am
           WHERE am.run_id = a.run_id
             AND am.artifact_id = ('raw-' || a.attempt_id)
         )
       ORDER BY a.attempt_no, a.attempt_id`,
      runId,
    );
    for (const value of rows) {
      const row = value as unknown as TaskRow & {
        attempt_id: string;
        attempt_no: number;
        provider_thread_id: string | null;
        provider_turn_id: string | null;
      };
      const rawArtifactId = `raw-${row.attempt_id}`;
      try {
        const logicalTask = JSON.parse(row.task_json) as AnyTurnTask;
        const contract = this.loadContract(
          runId,
          row.stage_id,
          row.stage_contract_hash,
        );
        const gate = this.loadGate(
          runId,
          row.stage_id,
          row.stage_contract_hash,
        );
        this.verifyFrozenRecoveryBindings(
          runId,
          row,
          logicalTask,
          contract,
          gate,
        );
        const raw = JSON.parse(
          this.readVerifiedArtifact(runId, rawArtifactId).toString(
            "utf8",
          ),
        ) as RawTurnResult;
        if (
          raw.attemptId !== row.attempt_id ||
          raw.providerThreadId !== row.provider_thread_id ||
          raw.providerTurnId !== row.provider_turn_id
        ) {
          throw new Error(
            "captured raw Turn identity differs from durable attempt",
          );
        }
        this.verifyCapturedUsage(runId, row.attempt_id, raw);
        const currentState = this.store.stateBinding(runId);
        if (
          !canonicalEqual(
            getTaskStateBinding(logicalTask),
            currentState,
          )
        ) {
          throw new Error(
            "captured raw Turn is stale against current StateBinding",
          );
        }
        const taskContext = {
          role: row.role,
          frozenBudget: contract.budget,
          currentState,
          stageContractHash: contract.sha256,
          schemaManifestSha256: row.schema_manifest_hash,
          skillSha256: row.skill_hash,
          expectedInputHash: row.input_hash,
          expectedOutputSchemaSha256:
            getExpectedOutputSchemaHash(logicalTask),
          rubricSha256:
            row.role === "direction_reviewer"
              ? RUBRIC_REGISTRY.direction_readiness_v1.sha256
              : row.role === "closure_reviewer"
                ? RUBRIC_REGISTRY.closure_v1.sha256
                : undefined,
        };
        const taskReport = this.validateCapturedTaskForRecovery(
          logicalTask,
          taskContext,
        );
        if (!taskReport.valid) {
          throw new Error(
            `captured task binding invalid: ${canonicalJson(
              taskReport.errors,
            )}`,
          );
        }
        const usageReport = validateAttemptUsage(raw, contract.budget);
        if (!usageReport.valid) {
          this.persistRecoveredFailure(
            runId,
            row,
            rawArtifactId,
            usageReport,
            "budget_invalid",
            "budget.turn_exceeded",
            true,
          );
          continue;
        }
        const captured = validateCapturedTurnOutput(
          row.role,
          logicalTask,
          raw,
        );
        if (!captured.securityReport.valid) {
          this.persistRecoveredFailure(
            runId,
            row,
            rawArtifactId,
            captured.securityReport,
            "security_invalid",
            "security.runtime_event_violation",
            true,
          );
          continue;
        }
        if (raw.status !== "completed") {
          throw new Error(
            `captured raw Turn is not completed: ${raw.status}`,
          );
        }
        let report = captured.resultValidationReport;
        if (
          report?.valid &&
          captured.result &&
          row.role === "workflow_decision"
        ) {
          report = preflightWorkflowProposal(
            this.store,
            logicalTask as WorkflowTurnTask,
            captured.result as WorkflowDecisionProposal,
          );
        }
        if (!report?.valid || !captured.result) {
          const rejected =
            report ??
            normalizationFailureReport(
              "captured Turn has no validated result",
            );
          const priorOutputFailures = Number(
            (
              this.store.db
                .prepare(
                  `SELECT COUNT(*) AS count FROM attempts
                   WHERE task_id = ?
                     AND status = 'output_contract_invalid'`,
                )
                .get(row.task_id) as { count: number }
            ).count,
          );
          const retryable =
            priorOutputFailures <
              MAX_OUTPUT_ATTEMPTS_PER_TASK - 1 &&
            Number(row.attempt_no) <
              MAX_TOTAL_ATTEMPTS_PER_TASK;
          this.store.recordOutputContractFailure(
            runId,
            rejected,
            row.task_id,
            row.attempt_id,
            rawArtifactId,
            rejected.errors[0]?.code ?? "schema.invalid",
            retryable,
          );
          if (!retryable) {
            this.markTaskAndStageFailed(
              runId,
              row.task_id,
              row.stage_id,
            );
            this.failRun(
              runId,
              "output_attempts_exhausted",
              "Captured Turn exhausted bounded output attempts.",
              false,
            );
          }
          continue;
        }
        this.ensureValidationReport(
          runId,
          row.task_id,
          row.attempt_id,
          report,
        );
        if (row.role === "workflow_decision") {
          const workflowGateEvaluation = evaluateGate(
            gate,
            this.gateContext(
              runId,
              gate,
              captured.result,
              report,
              raw,
              row.role,
              logicalTask,
              captured.securityReport,
            ),
          );
          if (!workflowGateEvaluation.passed) {
            this.recordGateFailure(
              runId,
              row,
              row.attempt_id,
              gate,
              workflowGateEvaluation,
              rawArtifactId,
            );
            continue;
          }
          const commit = commitWorkflowProposal(
            this.store,
            logicalTask as WorkflowTurnTask,
            captured.result as WorkflowDecisionProposal,
            {
              gateId: gate.gateId,
              gateEvaluation: workflowGateEvaluation,
              rawResponseArtifactId: rawArtifactId,
            },
          );
          if (!commit.accepted) {
            this.store.finishAttempt(
              row.attempt_id,
              "proposal_rejected",
              {
                rawResponseArtifactId: rawArtifactId,
                errorCode: commit.rejectionCode ?? undefined,
              },
            );
            this.store.db
              .prepare(
                `UPDATE tasks SET status = 'failed', updated_at = ?
                 WHERE task_id = ?`,
              )
              .run(timestamp(), row.task_id);
          }
          continue;
        }
        const evaluation = evaluateGate(
          gate,
          this.gateContext(
            runId,
            gate,
            captured.result,
            report,
            raw,
            row.role,
            logicalTask,
            captured.securityReport,
          ),
        );
        if (!evaluation.passed) {
          this.recordGateFailure(
            runId,
            row,
            row.attempt_id,
            gate,
            evaluation,
            rawArtifactId,
          );
          continue;
        }
        const resultId = getResultId(captured.result);
        const committed = commitTurnResult(
          this.store,
          runId,
          getTaskStateBinding(logicalTask),
          {
            resultId,
            taskId: row.task_id,
            attemptId: row.attempt_id,
            stageId: row.stage_id,
            gateId: gate.gateId,
            role: row.role as Exclude<
              RegisteredRole,
              "workflow_decision"
            >,
            messageType: row.expected_output_message_type,
            result: captured.result,
            validationReport: report,
            gateEvaluation: evaluation,
            rawResponseArtifactId: rawArtifactId,
            committedLifecycle:
              row.role === "closure_reviewer"
                ? "waiting_closure_review"
                : "running",
          },
        );
        if (row.role === "closure_reviewer") {
          this.afterClosureReview(
            runId,
            committed.nextState,
            captured.result as ClosureReviewEnvelope,
            resultId,
          );
        }
      } catch (error) {
        const report = normalizationFailureReport(
          `captured raw recovery failed closed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
        report.errors[0]!.code = "recovery.captured_raw_invalid";
        this.persistRecoveredFailure(
          runId,
          row,
          rawArtifactId,
          report,
          "recovery_invalid",
          "recovery.captured_raw_invalid",
          true,
        );
      }
    }
  }

  private verifyFrozenRecoveryBindings(
    runId: string,
    row: TaskRow,
    task: AnyTurnTask,
    contract: StageContract,
    gate: GateDefinition,
  ): void {
    if (
      task.taskId !== row.task_id ||
      task.stageId !== row.stage_id ||
      task.stageContractHash !== row.stage_contract_hash ||
      getTaskInputHash(task) !== row.input_hash ||
      recomputeTaskInputHash(task) !== row.input_hash ||
      contract.sha256 !== row.stage_contract_hash ||
      contract.sha256 !== canonicalSha256(withoutSha256(contract)) ||
      gate.sha256 !== canonicalSha256(withoutSha256(gate)) ||
      contract.stageId !== gate.stageId ||
      contract.stageId !== row.stage_id
    ) {
      throw new Error(
        "task, StageContract, GateDefinition, or hash binding mismatch",
      );
    }
    this.readVerifiedArtifact(runId, `prompt-${row.attempt_id}`);
  }

  /**
   * A captured Turn may have been created by the immediately preceding
   * protocol revision, where correctionFeedback did not yet exist. Preserve
   * its original hash binding, add only the missing nullable field for current
   * semantic validation, and ignore the expected hash change caused by that
   * compatibility projection. No newly dispatched task uses this path.
   */
  private validateCapturedTaskForRecovery(
    task: AnyTurnTask,
    context: Parameters<typeof validateTaskForDispatch>[1],
  ): ValidationReport {
    const current = validateTaskForDispatch(task, context);
    if (current.valid) return current;

    const projected = structuredClone(task);
    let projectedLegacyField = false;
    if ("decisionInputHash" in projected) {
      if (!Object.hasOwn(projected, "correctionFeedback")) {
        projected.correctionFeedback = null;
        projectedLegacyField = true;
      }
    } else if (!Object.hasOwn(projected.payload, "correctionFeedback")) {
      projected.payload.correctionFeedback = null;
      projectedLegacyField = true;
    }
    if (!projectedLegacyField) return current;

    const compatible = validateTaskForDispatch(projected, context);
    compatible.errors = compatible.errors.filter(
      (error) =>
        error.code !== "binding.input_hash" &&
        error.code !== "binding.decision_input_hash",
    );
    compatible.valid = compatible.errors.length === 0;
    return compatible;
  }

  private verifyCapturedUsage(
    runId: string,
    attemptId: string,
    raw: RawTurnResult,
  ): void {
    const rows = this.store.query(
      `SELECT input_tokens, cached_input_tokens, output_tokens,
              reasoning_output_tokens, tool_calls, elapsed_ms
       FROM usage_records
       WHERE run_id = ? AND attempt_id = ?`,
      runId,
      attemptId,
    );
    if (rows.length !== 1) {
      throw new Error(
        "captured Turn must have exactly one durable usage record",
      );
    }
    const usage = rows[0]!;
    const expected = {
      input_tokens: raw.usage.inputTokens,
      cached_input_tokens: raw.usage.cachedInputTokens,
      output_tokens: raw.usage.outputTokens,
      reasoning_output_tokens: raw.usage.reasoningOutputTokens,
      tool_calls: raw.toolEvents.length,
      elapsed_ms: raw.elapsedMs,
    };
    if (
      Object.entries(expected).some(
        ([key, value]) => Number(usage[key]) !== value,
      )
    ) {
      throw new Error(
        "captured raw Turn usage differs from durable usage record",
      );
    }
  }

  private readVerifiedArtifact(
    runId: string,
    artifactId: string,
  ): Buffer {
    const row = this.store.db
      .prepare(
        `SELECT relative_path, sha256, size_bytes
         FROM artifact_manifests
         WHERE run_id = ? AND artifact_id = ?`,
      )
      .get(runId, artifactId) as
      | {
          relative_path: string;
          sha256: string;
          size_bytes: number;
        }
      | undefined;
    if (!row) throw new Error(`missing artifact ${artifactId}`);
    const workRoot = resolve(this.config.workDir);
    const absolutePath = resolve(workRoot, row.relative_path);
    if (
      absolutePath === workRoot ||
      !absolutePath.startsWith(`${workRoot}/`)
    ) {
      throw new Error(`artifact ${artifactId} escapes work directory`);
    }
    const bytes = readFileSync(absolutePath);
    if (
      bytes.byteLength !== Number(row.size_bytes) ||
      sha256Bytes(bytes) !== row.sha256
    ) {
      throw new Error(`artifact ${artifactId} hash/size mismatch`);
    }
    return bytes;
  }

  private ensureValidationReport(
    runId: string,
    taskId: string,
    attemptId: string,
    report: ValidationReport,
  ): void {
    const existing = this.store.db
      .prepare(
        `SELECT 1 FROM validation_reports
         WHERE run_id = ? AND task_id = ? AND attempt_id = ?
           AND valid = ? AND report_json = ?
         LIMIT 1`,
      )
      .get(
        runId,
        taskId,
        attemptId,
        report.valid ? 1 : 0,
        canonicalJson(report),
      );
    if (!existing) {
      this.store.insertValidationReport(
        runId,
        report,
        taskId,
        attemptId,
      );
    }
  }

  private persistRecoveredFailure(
    runId: string,
    row: TaskRow & { attempt_id: string },
    rawArtifactId: string,
    report: ValidationReport,
    attemptStatus: string,
    errorCode: string,
    terminal: boolean,
  ): void {
    this.ensureValidationReport(
      runId,
      row.task_id,
      row.attempt_id,
      report,
    );
    this.store.finishAttempt(row.attempt_id, attemptStatus, {
      rawResponseArtifactId: rawArtifactId,
      errorCode,
    });
    this.markTaskAndStageFailed(
      runId,
      row.task_id,
      row.stage_id,
    );
    this.failRun(
      runId,
      errorCode,
      report.errors[0]?.message ?? "captured raw recovery failed",
      terminal,
    );
  }

  private reconcileInflightAttempts(runId: string): void {
    const latestEvent = this.store.db
      .prepare(
        `SELECT event_type FROM events
         WHERE run_id = ? ORDER BY event_cursor DESC LIMIT 1`,
      )
      .get(runId) as { event_type: string } | undefined;
    const explicitlyResumed =
      latestEvent?.event_type === "operator_resumed";
    const rows = this.store.query(
      `SELECT a.attempt_id, a.task_id, a.attempt_no, a.status,
              t.stage_id,
              (
                SELECT COUNT(*) FROM attempts prior
                WHERE prior.task_id = a.task_id
                  AND prior.status IN (
                    'provider_failed',
                    'interrupted_reconciled'
                  )
              ) AS prior_provider_failure_count,
              (
                SELECT COUNT(*) FROM attempts prior
                WHERE prior.task_id = a.task_id
                  AND prior.status = 'output_contract_invalid'
              ) AS prior_output_failure_count
       FROM attempts a
       JOIN tasks t ON t.task_id = a.task_id
       WHERE a.run_id = ? AND a.status IN ('created', 'running')
       ORDER BY attempt_no, attempt_id`,
      runId,
    );
    for (const row of rows) {
      const attemptId = String(row.attempt_id);
      const taskId = String(row.task_id);
      const exhausted =
        Number(row.prior_provider_failure_count) + 1 >=
          MAX_PROVIDER_FAILURES_PER_TASK ||
        Number(row.prior_output_failure_count) >=
          MAX_OUTPUT_ATTEMPTS_PER_TASK ||
        Number(row.attempt_no) >= MAX_TOTAL_ATTEMPTS_PER_TASK;
      const expected = this.store.stateBinding(runId);
      this.store.casTransition(
        runId,
        expected,
        {
          lifecycle:
            exhausted && !explicitlyResumed
              ? "failed_retriable"
              : undefined,
          pauseOrBlockReason: exhausted && !explicitlyResumed
            ? `provider history unavailable after final attempt ${attemptId}`
            : undefined,
          eventType: exhausted
            ? "attempt_reconcile_exhausted"
            : "attempt_reconciled_for_fresh_retry",
          eventPayload: {
            attemptId,
            taskId,
            attemptNo: Number(row.attempt_no),
            providerHistoryResumed: false,
          },
        },
        (db) => {
          db.prepare(
            `UPDATE attempts
             SET status = 'interrupted_reconciled',
                 error_code = 'provider_history_not_resumed',
                 finished_at = ?
             WHERE attempt_id = ?`,
          ).run(timestamp(), attemptId);
          db.prepare(
            `UPDATE tasks SET status = ?, updated_at = ?
             WHERE task_id = ?`,
          ).run(
            exhausted ? "failed" : "pending_output_retry",
            timestamp(),
            taskId,
          );
          if (exhausted) {
            db.prepare(
              `UPDATE workflow_plan_nodes SET lifecycle = 'failed'
               WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
                 AND lifecycle NOT IN ('completed', 'superseded')`,
            ).run(
              runId,
              expected.workflowPlanRevision,
              String(row.stage_id),
            );
          }
        },
      );
    }
    const orphaned = this.store.query(
      `SELECT t.task_id, t.stage_id, t.output_attempt_count,
              (
                SELECT a.status FROM attempts a
                WHERE a.task_id = t.task_id
                ORDER BY a.attempt_no DESC LIMIT 1
              ) AS latest_attempt_status,
              (
                SELECT COUNT(*) FROM attempts a
                WHERE a.task_id = t.task_id
                  AND a.status IN (
                    'provider_failed',
                    'interrupted_reconciled'
                  )
              ) AS provider_failure_count,
              (
                SELECT COUNT(*) FROM attempts a
                WHERE a.task_id = t.task_id
                  AND a.status = 'output_contract_invalid'
              ) AS output_failure_count,
              (
                SELECT COUNT(*) FROM attempts a
                WHERE a.task_id = t.task_id
              ) AS attempt_count
       FROM tasks t
       WHERE t.run_id = ?
         AND t.status IN ('dispatched', 'running')
         AND NOT EXISTS (
           SELECT 1 FROM attempts a
           WHERE a.task_id = t.task_id
             AND a.status IN ('created', 'running')
         )
       ORDER BY t.task_id`,
      runId,
    );
    for (const row of orphaned) {
      const attemptCount = Number(row.attempt_count);
      const exhausted =
        Number(row.provider_failure_count) >=
          MAX_PROVIDER_FAILURES_PER_TASK ||
        Number(row.output_failure_count) >=
          MAX_OUTPUT_ATTEMPTS_PER_TASK ||
        attemptCount >= MAX_TOTAL_ATTEMPTS_PER_TASK;
      this.store.casTransition(
        runId,
        this.store.stateBinding(runId),
        {
          lifecycle:
            exhausted && !explicitlyResumed
              ? "failed_retriable"
              : undefined,
          pauseOrBlockReason: exhausted && !explicitlyResumed
            ? `orphaned task exhausted attempts: ${String(row.task_id)}`
            : undefined,
          eventType: exhausted
            ? "orphaned_task_reconcile_exhausted"
            : "orphaned_task_reconciled_for_fresh_retry",
          eventPayload: {
            taskId: String(row.task_id),
            attemptCount,
            latestAttemptStatus:
              row.latest_attempt_status === null
                ? null
                : String(row.latest_attempt_status),
            providerHistoryResumed: false,
          },
        },
        (db) => {
          db.prepare(
            `UPDATE tasks SET status = ?, updated_at = ?
             WHERE task_id = ?`,
          ).run(
            exhausted ? "failed" : "pending_output_retry",
            timestamp(),
            String(row.task_id),
          );
          if (exhausted) {
            db.prepare(
              `UPDATE workflow_plan_nodes SET lifecycle = 'failed'
               WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
                 AND lifecycle NOT IN ('completed', 'superseded')`,
            ).run(
              runId,
              this.store.getRun(runId).workflowPlanRevision,
              String(row.stage_id),
            );
          }
        },
      );
    }
  }

  private reconcileCommittedClosureReview(runId: string): void {
    if (this.store.getRun(runId).lifecycle !== "waiting_closure_review") {
      return;
    }
    const row = this.store.db
      .prepare(
        `SELECT r.result_id, r.payload_json
         FROM turn_results r
         LEFT JOIN result_consumptions c ON c.result_id = r.result_id
         WHERE r.run_id = ?
           AND r.message_type = 'CLOSURE_REVIEW'
           AND r.status = 'committed'
           AND c.result_id IS NULL
         ORDER BY r.committed_snapshot_version DESC
         LIMIT 1`,
      )
      .get(runId) as
      | { result_id: string; payload_json: string }
      | undefined;
    if (!row) {
      throw new Error(
        "waiting_closure_review has no committed, unconsumed ClosureReview",
      );
    }
    this.afterClosureReview(
      runId,
      this.store.stateBinding(runId),
      JSON.parse(row.payload_json) as ClosureReviewEnvelope,
      row.result_id,
    );
  }
}

function result(
  runId: string,
  lifecycle: string,
  transitions: number,
): ControllerRunResult {
  return {
    runId,
    lifecycle,
    transitions,
    completed: lifecycle === "completed",
  };
}

function getTaskStateBinding(task: AnyTurnTask): StateBinding {
  return "stateSnapshot" in task ? task.stateSnapshot : task.stateBinding;
}

function getTaskInputHash(task: AnyTurnTask): string {
  return "decisionInputHash" in task ? task.decisionInputHash : task.inputHash;
}

function recomputeTaskInputHash(task: AnyTurnTask): string {
  const preimage = structuredClone(task);
  if ("decisionInputHash" in preimage) {
    preimage.decisionInputHash = "";
    return canonicalSha256({
      task: preimage,
      skillSha256: preimage.skill.sha256,
      expectedSchemaSha256:
        preimage.schema.expectedOutputSchemaSha256,
      inlineArtifactHashes: [],
    });
  }
  preimage.inputHash = "";
  return canonicalSha256(preimage);
}

function getSkillName(task: AnyTurnTask): string {
  return "payload" in task ? task.payload.skill.name : task.skill.name;
}

function getSkillHash(task: AnyTurnTask): string {
  return "payload" in task ? task.payload.skill.sha256 : task.skill.sha256;
}

function getSchemaManifestHash(task: AnyTurnTask): string {
  return "payload" in task
    ? task.payload.schema.manifestSha256
    : task.schema.manifestSha256;
}

function getExpectedOutputSchemaHash(task: AnyTurnTask): string {
  return "payload" in task
    ? task.payload.schema.expectedOutputSchemaSha256
    : task.schema.expectedOutputSchemaSha256;
}

function taskCorrectionFeedback(
  task: AnyTurnTask,
): TurnCorrectionFeedback | null {
  if ("payload" in task) {
    return task.payload.correctionFeedback ?? null;
  }
  return task.correctionFeedback ?? null;
}

function buildCorrectionFeedback(
  previousAttemptId: string,
  rawText: string,
  report: ValidationReport,
  validationReportId: string,
): TurnCorrectionFeedback {
  const failureClass = classifyCorrectionFailure(report);
  const errors = report.errors.slice(0, 32).map((error) => ({
    code: error.code.slice(0, 128),
    jsonPointer:
      error.jsonPointer === null
        ? null
        : error.jsonPointer.slice(0, 512),
    message: error.message.slice(0, 4_096),
    ...correctionGuidance(error),
  }));
  return {
    previousAttemptId,
    previousOutputSha256: sha256Bytes(rawText),
    validationReportId,
    validationReportSha256: canonicalSha256(report),
    failureClass,
    errors,
    retryInstruction:
      failureClass === "STRUCTURE_INVALID"
        ? "Return exactly one complete JSON value that satisfies the attached output Schema; do not include prose, fences, or extra fields."
        : failureClass === "BINDING_INVALID"
          ? "Regenerate the complete output for this fresh attempt and exactly echo every current task, attempt, state, contract, and input-hash binding."
          : "Regenerate the complete output without changing the immutable objective. Correct every listed semantic, authority, permission, reference, Stage, and Gate error using the required form stated in each message, then revalidate the entire replacement—especially every typed Gate operand—not only the listed pointers.",
  };
}

function correctionGuidance(
  error: ValidationReport["errors"][number],
): {
  requiredRule: string;
  validExamples: string[];
} {
  switch (error.code) {
    case "gate.result_pointer_not_stable":
      return {
        requiredRule:
          "Use exactly /payload for the domain payload object, or a schema-valid descendant beginning /payload/. Never address envelope identity or binding fields.",
        validExamples: [
          canonicalJson({
            source: "result",
            pointer: "/payload",
            valueType: "object",
          }),
          canonicalJson({
            source: "result",
            pointer: "/payload/needId",
            valueType: "string",
          }),
        ],
      };
    case "gate.validator_pointer_contract": {
      const fact = error.message.split(/\s+/, 1)[0] ?? "validator_fact";
      if (
        error.message.includes("requires pointer:null") ||
        error.message.includes("forbids a result pointer")
      ) {
        return {
          requiredRule:
            "Whole-result validator facts must use pointer:null. Only references_resolve and source_context_present accept a result pointer.",
          validExamples: [
            canonicalJson({
              source: "validator",
              fact,
              pointer: null,
              valueType: "boolean",
            }),
          ],
        };
      }
      return {
        requiredRule:
          "references_resolve and source_context_present require a schema-valid pointer into the result domain payload.",
        validExamples: [
          canonicalJson({
            source: "validator",
            fact,
            pointer: "/payload/findings",
            valueType: "boolean",
          }),
        ],
      };
    }
    case "gate.pointer_not_found":
      return {
        requiredRule:
          "Choose a JSON Pointer that resolves against the frozen input or output Schema named by the error.",
        validExamples: [
          "/payload",
          "/payload/needId",
          "/payload/findings",
        ],
      };
    case "gate.pointer_type_mismatch":
      return {
        requiredRule:
          "Declare the valueType defined by the frozen Schema at the selected pointer, and make expected exactly match that type.",
        validExamples: [],
      };
    default:
      if (
        error.code.startsWith("normalization.") ||
        error.code.startsWith("schema.")
      ) {
        return {
          requiredRule:
            "Return one complete strict JSON value satisfying the attached expected output Schema at this pointer.",
          validExamples: [],
        };
      }
      if (
        error.code.includes("binding") ||
        error.code.includes("state") ||
        error.code.includes("hash")
      ) {
        return {
          requiredRule:
            "Copy the corresponding authoritative value from the current task packet exactly; never reuse the previous attempt binding.",
          validExamples: [],
        };
      }
      return {
        requiredRule:
          "Satisfy the Controller validation rule stated in message while preserving every immutable task constraint.",
        validExamples: [],
      };
  }
}

function classifyCorrectionFailure(
  report: ValidationReport,
): TurnCorrectionFeedback["failureClass"] {
  const codes = report.errors.map((error) => error.code);
  if (
    codes.some(
      (code) =>
        code.startsWith("normalization.") ||
        code.startsWith("schema."),
    )
  ) {
    return "STRUCTURE_INVALID";
  }
  if (
    codes.some(
      (code) =>
        code.startsWith("envelope.") ||
        code.startsWith("binding.") ||
        code.includes("input_hash") ||
        code.includes("decision_input_hash") ||
        code.includes("stale_expected_state") ||
        code.includes("identity"),
    )
  ) {
    return "BINDING_INVALID";
  }
  return "SEMANTIC_INVALID";
}

function getResultId(result: unknown): string {
  const payload = (result as { payload?: Record<string, unknown> }).payload;
  for (const key of ["packetId", "reviewId"]) {
    if (typeof payload?.[key] === "string") return String(payload[key]);
  }
  return `result-${canonicalSha256(result).slice(0, 24)}`;
}

function resolveJsonPointerValue(
  value: unknown,
  pointer: string,
): { found: true; value: unknown } | { found: false } {
  if (!pointer.startsWith("/") || pointer === "/") {
    return { found: false };
  }
  let current = value;
  for (const encoded of pointer.slice(1).split("/")) {
    const segment = encoded
      .replaceAll("~1", "/")
      .replaceAll("~0", "~");
    if (Array.isArray(current)) {
      if (!/^\d+$/.test(segment)) return { found: false };
      const index = Number(segment);
      if (index >= current.length) return { found: false };
      current = current[index];
      continue;
    }
    if (
      current === null ||
      typeof current !== "object" ||
      !Object.hasOwn(current, segment)
    ) {
      return { found: false };
    }
    current = (current as Record<string, unknown>)[segment];
  }
  return current === undefined
    ? { found: false }
    : { found: true, value: current };
}

function resolvedPointer(
  value: unknown,
  pointer: string,
  valueType: GateValueType,
  label: string,
): GateResolution {
  const resolution = resolveJsonPointerValue(value, pointer);
  if (!resolution.found) {
    return unresolved(
      "gate.subject_missing",
      `${label} pointer ${pointer} does not resolve`,
    );
  }
  return resolvedValue(
    resolution.value,
    valueType,
    `${label} pointer ${pointer}`,
  );
}

function resolvedValue(
  value: unknown,
  valueType: GateValueType,
  detail: string,
): GateResolution {
  if (!matchesGateValueType(value, valueType)) {
    return unresolved(
      "gate.runtime_type_mismatch",
      `${detail} does not match declared ${valueType}`,
    );
  }
  return { resolved: true, value, detail };
}

function unresolved(
  errorCode: string,
  detail: string,
): GateResolution {
  return { resolved: false, errorCode, detail };
}

function matchesGateValueType(
  value: unknown,
  valueType: GateValueType,
): boolean {
  switch (valueType) {
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    case "string_array":
      return (
        Array.isArray(value) &&
        value.every((item) => typeof item === "string")
      );
    case "object":
      return (
        value !== null &&
        typeof value === "object" &&
        !Array.isArray(value)
      );
  }
}

function evidenceSourceContextsPresent(
  result: unknown,
  findingsPointer: string,
): boolean {
  const findings = resolveJsonPointerValue(result, findingsPointer);
  const contexts = resolveJsonPointerValue(
    result,
    "/payload/contextsRead",
  );
  if (
    !findings.found ||
    !Array.isArray(findings.value) ||
    !contexts.found ||
    !Array.isArray(contexts.value)
  ) {
    return false;
  }
  if (findings.value.length === 0) return true;
  const sourceUnits = new Set(
    contexts.value.flatMap((context) => {
      if (
        context !== null &&
        typeof context === "object" &&
        typeof (context as Record<string, unknown>).sourceUnitId ===
          "string"
      ) {
        return [
          String(
            (context as Record<string, unknown>).sourceUnitId,
          ),
        ];
      }
      return [];
    }),
  );
  return findings.value.every(
    (finding) =>
      finding !== null &&
      typeof finding === "object" &&
      typeof (finding as Record<string, unknown>).sourceUnitId ===
        "string" &&
      sourceUnits.has(
        String(
          (finding as Record<string, unknown>).sourceUnitId,
        ),
      ),
  );
}

function normalizationFailureReport(message: string): ValidationReport {
  return {
    validatorVersion: "simple-semantic-loop-validator/1",
    valid: false,
    errors: [
      {
        code: "normalization.invalid",
        jsonPointer: null,
        message,
      },
    ],
    checkedArtifactHashes: [],
    checkedObjectRefs: [],
  };
}

function validateAttemptUsage(
  raw: NonNullable<
    Awaited<ReturnType<typeof dispatchFreshTurnAttempt>>["rawTurn"]
  >,
  budget: TurnBudget,
): ValidationReport {
  const report: ValidationReport = {
    validatorVersion: "simple-semantic-loop-validator/1",
    valid: true,
    errors: [],
    checkedArtifactHashes: [],
    checkedObjectRefs: [],
  };
  const checks = [
    ["input_tokens", raw.usage.inputTokens, budget.maxInputTokens],
    ["output_tokens", raw.usage.outputTokens, budget.maxOutputTokens],
    ["tool_calls", raw.toolEvents.length, budget.maxToolCalls],
    ["elapsed_ms", raw.elapsedMs, budget.timeoutMs],
  ] as const;
  for (const [name, actual, maximum] of checks) {
    if (actual > maximum) {
      report.valid = false;
      report.errors.push({
        code: `budget.${name}_exceeded`,
        jsonPointer: null,
        message: `${name}=${actual} exceeds frozen maximum ${maximum}`,
      });
    }
  }
  return report;
}

function stripFrozenContract(contract: StageContract): StageContractDraft {
  const {
    contractId: _contractId,
    stageId: _stageId,
    revision: _revision,
    definedAtSnapshotVersion: _defined,
    sha256: _hash,
    ...draft
  } = contract;
  return draft;
}

function stripFrozenGate(gate: GateDefinition) {
  const {
    gateId: _gateId,
    stageId: _stageId,
    revision: _revision,
    definedAtSnapshotVersion: _defined,
    sha256: _hash,
    stageContractHash: _contractHash,
    proposedCriteriaSha256: _criteriaHash,
    compilerPolicyVersion: _compilerVersion,
    evaluatorVersion: _evaluatorVersion,
    ...draft
  } = gate;
  return draft;
}

function withoutSha256<T extends { sha256: string }>(
  value: T,
): Omit<T, "sha256"> {
  const { sha256: _sha256, ...rest } = value;
  return rest;
}

function timestamp(): string {
  return new Date().toISOString();
}
