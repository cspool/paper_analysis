import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type {
  GoalDispatch,
  GoalRuntimePersistenceEvent,
  RawGoalResult,
  RawTurnResult,
  TurnRuntime,
} from "../simple_semantic_loop/refactor/types.ts";
import {
  buildDecisionPrompt,
  buildJudgePrompt,
  buildLabPrompt,
  decisionDeveloperInstructions,
  judgeDeveloperInstructions,
  labDeveloperInstructions,
} from "./prompts.ts";
import {
  decisionCorrectionSuffix,
  judgeCorrectionSuffix,
  parseDecisionResult,
  parseJudgeResult,
} from "./protocol.ts";
import { verifyFrozenFiles } from "./setup.ts";
import { DirectionExperimentStore } from "./store.ts";
import {
  DIRECTION_EXPERIMENT_FORMAT_VERSION,
  type DecisionHistoryEntry,
  type DirectionGoalRecord,
  type DirectionHandoff,
  type DirectionHistoryEntry,
  type DirectionLoopOutcome,
  type DirectionRunFile,
  type DirectionStateFile,
  type ExperimentContractRecord,
  type ExperimentDecisionResult,
  type ExperimentHistoryEntry,
  type FreshTurnRecord,
  type GoalInterruptionKind,
  type JudgeResult,
  type JudgmentHistoryEntry,
  type LabCheckpoint,
  type LabCycleBinding,
  type LabGoalInvocationRecord,
  type LabRuntimeEnvelope,
  type TerminalExperimentDecision,
} from "./types.ts";

interface LabArtifactInspection {
  resultRef: string | null;
  checkpointRef: string | null;
  errors: string[];
}

export class DirectionExperimentController {
  private readonly store: DirectionExperimentStore;
  private readonly runtime: TurnRuntime;

  constructor(store: DirectionExperimentStore, runtime: TurnRuntime) {
    this.store = store;
    this.runtime = runtime;
  }

  async run(resume = false, additionalCycles?: number): Promise<DirectionLoopOutcome> {
    this.store.acquireLock();
    try {
      this.assertWritableAndFrozen();
      this.prepareLifecycle(resume, additionalCycles);
      while (this.store.readState().lifecycle === "RUNNING") {
        this.assertWritableAndFrozen();
        if (this.honorPendingPauseRequest()) break;
        const node = this.store.readState().node;
        if (node === "DECISION") await this.executeDecision();
        else if (node === "LAB_GOAL") await this.executeLabGoal();
        else if (node === "JUDGE") await this.executeJudge();
        else throw new Error("RUNNING state has no executable node");
      }
      return this.currentOutcome();
    } catch (error) {
      const message = errorMessage(error);
      let evidenceScope: DirectionStateFile["evidenceScope"] = null;
      if (
        this.store.exists("run.json") &&
        this.store.exists("state.json") &&
        this.isCurrentFormat()
      ) {
        const state = this.store.readState();
        evidenceScope = state.evidenceScope;
        if (state.lifecycle === "RUNNING") {
          this.store.writeState({
            ...state,
            revision: state.revision + 1,
            lifecycle: "FAILED",
            node: null,
            reason: message,
            pauseKind: null,
            activeLabInvocationRef: null,
          }, "RUN_FAILED");
        }
        this.writeOutcome("FAILED", message);
      }
      return {
        workflowOutcome: "FAILED",
        reportRef: null,
        handoffRef: null,
        evidenceScope,
        reason: message,
      };
    } finally {
      this.store.releaseLock();
    }
  }

  private prepareLifecycle(resume: boolean, additionalCycles?: number): void {
    const state = this.store.readState();
    if (state.lifecycle === "FINISHED") return;
    if (state.lifecycle === "FAILED") {
      throw new Error("failed run is not resumable; inspect status and initialize a new run");
    }
    if (
      additionalCycles !== undefined &&
      (!Number.isInteger(additionalCycles) || additionalCycles < 1)
    ) throw new Error("additionalCycles must be a positive integer");
    if (!resume) {
      if (state.lifecycle === "PAUSED") throw new Error("run is paused; use resume");
      if (additionalCycles !== undefined) {
        throw new Error("--additional-cycles is only valid with resume");
      }
      return;
    }
    this.store.clearPauseRequest();
    if (state.lifecycle === "RUNNING") {
      if (additionalCycles !== undefined) {
        this.store.writeState({
          ...state,
          revision: state.revision + 1,
          authorizedLabCycles: state.authorizedLabCycles + additionalCycles,
        }, "RUNNING_RECOVERY_AUTHORIZED");
      }
      return;
    }
    if (state.pauseKind === "CYCLE_BUDGET_EXHAUSTED" && additionalCycles === undefined) {
      throw new Error("cycle budget is exhausted; resume requires --additional-cycles N");
    }
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "RUNNING",
      node: state.node ?? "DECISION",
      authorizedLabCycles: state.authorizedLabCycles + (additionalCycles ?? 0),
      reason: null,
      pauseKind: null,
      activeLabInvocationRef: null,
    }, "RUN_RESUMED");
  }

  private honorPendingPauseRequest(): boolean {
    const request = this.store.readPauseRequest();
    if (!request) return false;
    const state = this.store.readState();
    if (state.lifecycle !== "RUNNING") return false;
    const reason = request.reason || "operator requested pause";
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "PAUSED",
      reason,
      pauseKind: "OPERATOR_REQUESTED",
      activeLabInvocationRef: null,
    }, "RUN_PAUSED");
    this.store.appendEvent("OPERATOR_PAUSE_APPLIED", ["control/pause-request.json"], request);
    this.store.clearPauseRequest();
    this.writeOutcome("PAUSED", reason);
    return true;
  }

  private async executeDecision(): Promise<void> {
    const run = this.store.readRun();
    const sourceState = this.store.readState();
    const sourceHistory = this.history();
    const ordinal = sourceHistory.filter((entry) => entry.kind === "DECISION").length + 1;
    const snapshotRoot = `snapshots/decision-${ordinal}-${this.store.newId("snapshot")}`;
    const stateSnapshotRef = `${snapshotRoot}/state.json`;
    const historySnapshotRef = `${snapshotRoot}/trajectory.json`;
    const runtimeEnvelopeRef = `${snapshotRoot}/lab_runtime_envelope.json`;
    const runtimeEnvelope = this.labRuntimeEnvelope(run, sourceState);
    this.store.writeJson(stateSnapshotRef, sourceState);
    this.store.writeJson(historySnapshotRef, this.trajectorySnapshot(run, sourceState));
    this.store.writeJson(runtimeEnvelopeRef, runtimeEnvelope);

    let correction = "";
    let outputFailures = 0;
    let runtimeFailures = 0;
    let attempt = 0;
    while (true) {
      attempt += 1;
      const turnId = this.store.newId("experiment-decision");
      const root = `decisions/decision-${ordinal}/attempt-${attempt}-${turnId}`;
      const recordRef = `${root}/turn.json`;
      const promptRef = `${root}/prompt.md`;
      const runtimeRef = `${root}/runtime.jsonl`;
      const prompt = buildDecisionPrompt(
        this.store,
        run,
        stateSnapshotRef,
        historySnapshotRef,
        runtimeEnvelopeRef,
        correction,
      );
      this.store.writeText(promptRef, prompt);
      const record = freshRecord(
        turnId,
        "EXPERIMENT_DECISION",
        ordinal,
        attempt,
        promptRef,
        stateSnapshotRef,
        historySnapshotRef,
        runtimeEnvelopeRef,
        runtimeRef,
      );
      this.store.writeJson(recordRef, record);
      this.store.appendEvent("EXPERIMENT_DECISION_STARTED", [recordRef, promptRef]);
      const raw = await this.runFreshTurn(run, {
        turnId,
        role: "EXPERIMENT_DECISION",
        prompt,
        effort: "max",
        timeoutProfile: run.budgets.decision,
        developerInstructions: decisionDeveloperInstructions(),
        runtimeRef,
      });
      this.assertAuthorityUnchanged(sourceState, sourceHistory);
      const rawOutputRef = `${root}/raw_output.txt`;
      this.store.writeText(rawOutputRef, raw.text || raw.partialText || "");
      const finished = finishFreshRecord(record, raw, rawOutputRef);
      if (raw.status !== "completed") {
        runtimeFailures += 1;
        const errorRef = `${root}/error.json`;
        this.store.writeJson(errorRef, runtimeError(raw));
        this.store.writeJson(recordRef, { ...finished, errorRef });
        this.store.appendEvent("EXPERIMENT_DECISION_RUNTIME_FAILED", [recordRef, errorRef]);
        if (runtimeFailures <= run.budgets.maxRuntimeRetries) continue;
        this.pauseFreshTurn(
          "DECISION",
          "DECISION_RETRY_EXHAUSTED",
          "Experiment Decision exceeded runtime retry budget",
        );
        return;
      }
      const parsed = parseDecisionResult(raw.text, {
        maxEstimatedMinutes: runtimeEnvelope.maxContractMinutes,
      });
      if (!parsed.value) {
        outputFailures += 1;
        const errorRef = `${root}/error.json`;
        this.store.writeJson(errorRef, { kind: "CORE_PROTOCOL", errors: parsed.errors });
        this.store.writeJson(recordRef, { ...finished, errorRef });
        this.store.appendEvent("EXPERIMENT_DECISION_OUTPUT_INVALID", [recordRef, errorRef]);
        if (outputFailures <= run.budgets.maxOutputRetries) {
          correction = decisionCorrectionSuffix(parsed.errors);
          continue;
        }
        this.pauseFreshTurn(
          "DECISION",
          "DECISION_RETRY_EXHAUSTED",
          "Experiment Decision exceeded output-correction retry budget",
        );
        return;
      }
      const resultRef = `${root}/decision.json`;
      this.store.writeJson(resultRef, parsed.value);
      this.store.writeJson(recordRef, { ...finished, resultRef });
      this.store.appendEvent("EXPERIMENT_DECISION_ACCEPTED", [recordRef, resultRef]);
      this.applyDecision(parsed.value, resultRef, ordinal);
      return;
    }
  }

  private applyDecision(
    value: ExperimentDecisionResult,
    decisionRef: string,
    ordinal: number,
  ): void {
    const state = this.store.readState();
    if (value.decision === "RUN_LAB") {
      const contract = this.freezeContract(value, decisionRef);
      const nextCycle = state.cycle + 1;
      const exhausted = nextCycle > state.authorizedLabCycles;
      const history: DecisionHistoryEntry = {
        kind: "DECISION",
        ordinal,
        decisionRef,
        decision: value.decision,
        evidenceScope: value.evidenceScope,
        reason: value.reason,
        contractRevision: contract.contractRevision,
        contractRef: contract.ref,
        contractHash: contract.hash,
        reviewFocusRef: null,
        completedAt: nowIso(),
      };
      this.store.appendJsonLine("history.jsonl", history);
      const reason = exhausted
        ? `authorized Lab cycle budget through cycle ${state.authorizedLabCycles} exhausted; cycle ${nextCycle} and contract ${contract.contractRevision} are prepared`
        : null;
      this.store.writeState({
        ...state,
        revision: state.revision + 1,
        lifecycle: exhausted ? "PAUSED" : "RUNNING",
        node: "LAB_GOAL",
        cycle: nextCycle,
        transitions: state.transitions + 1,
        reason,
        pauseKind: exhausted ? "CYCLE_BUDGET_EXHAUSTED" : null,
        activeContractRevision: contract.contractRevision,
        activeContractRef: contract.ref,
        activeContractHash: contract.hash,
        activeGoalRecordRef: null,
        activeLabInvocationRef: null,
        activeJudgeRequestRef: null,
        latestLabResultRef: null,
        latestCheckpointRef: null,
        latestDecisionRef: decisionRef,
        evidenceScope: value.evidenceScope,
      }, exhausted ? "CYCLE_BUDGET_EXHAUSTED" : "EXPERIMENT_CONTRACT_FROZEN");
      if (exhausted) this.writeOutcome("PAUSED", reason);
      return;
    }
    if (value.decision === "RUN_JUDGE") {
      const requestRef = this.createJudgeRequest(
        "DECISION_REQUESTED_REVIEW",
        value.reviewFocus!,
        decisionRef,
      );
      const history: DecisionHistoryEntry = {
        kind: "DECISION",
        ordinal,
        decisionRef,
        decision: value.decision,
        evidenceScope: value.evidenceScope,
        reason: value.reason,
        contractRevision: state.activeContractRef ? state.activeContractRevision : null,
        contractRef: state.activeContractRef,
        contractHash: state.activeContractHash,
        reviewFocusRef: requestRef,
        completedAt: nowIso(),
      };
      this.store.appendJsonLine("history.jsonl", history);
      this.store.writeState({
        ...state,
        revision: state.revision + 1,
        lifecycle: "RUNNING",
        node: "JUDGE",
        transitions: state.transitions + 1,
        reason: null,
        pauseKind: null,
        activeJudgeRequestRef: requestRef,
        latestDecisionRef: decisionRef,
        evidenceScope: value.evidenceScope,
      }, "JUDGE_REVIEW_REQUESTED");
      return;
    }

    const history: DecisionHistoryEntry = {
      kind: "DECISION",
      ordinal,
      decisionRef,
      decision: value.decision,
      evidenceScope: value.evidenceScope,
      reason: value.reason,
      contractRevision: state.activeContractRef ? state.activeContractRevision : null,
      contractRef: state.activeContractRef,
      contractHash: state.activeContractHash,
      reviewFocusRef: null,
      completedAt: nowIso(),
    };
    this.store.appendJsonLine("history.jsonl", history);
    if (value.decision === "BLOCKED") {
      this.store.writeState({
        ...state,
        revision: state.revision + 1,
        lifecycle: "PAUSED",
        node: "DECISION",
        transitions: state.transitions + 1,
        reason: value.reason,
        pauseKind: "DECISION_BLOCKED",
        latestDecisionRef: decisionRef,
        evidenceScope: value.evidenceScope,
      }, "EXPERIMENT_DECISION_BLOCKED");
      this.writeOutcome("PAUSED", value.reason);
      return;
    }
    this.finish(value, decisionRef);
  }

  private freezeContract(
    value: ExperimentDecisionResult,
    decisionRef: string,
  ): { contractRevision: number; ref: string; hash: string } {
    const run = this.store.readRun();
    const state = this.store.readState();
    const contractRevision = state.activeContractRevision + 1;
    const ref = `contracts/contract-${contractRevision}/contract.json`;
    const record: ExperimentContractRecord = {
      formatVersion: 2,
      contractRevision,
      directionId: run.source.directionId,
      directionRevision: run.source.directionRevision,
      sourceDirectionRef: run.inputs.directionResult.path,
      decisionRef,
      targetEvidenceScope: value.evidenceScope,
      ...value.experimentContract!,
      createdAt: nowIso(),
    };
    this.store.writeJson(ref, record);
    const hash = this.store.sha256(ref);
    this.store.writeJson(`contracts/contract-${contractRevision}/binding.json`, {
      contractRevision,
      contractRef: ref,
      contractHash: hash,
      decisionRef,
      directionId: run.source.directionId,
      directionRevision: run.source.directionRevision,
    });
    this.store.appendEvent("EXPERIMENT_CONTRACT_WRITTEN", [ref, decisionRef], {
      contractRevision,
      contractHash: hash,
      estimatedMinutes: record.estimatedMinutes,
    });
    return { contractRevision, ref, hash };
  }

  private async executeLabGoal(): Promise<void> {
    const run = this.store.readRun();
    const state = this.store.readState();
    if (!state.activeContractRef || !state.activeContractHash || state.activeContractRevision < 1) {
      throw new Error("LAB_GOAL node lacks one frozen experiment contract");
    }
    if (state.cycle > state.authorizedLabCycles) {
      const reason = `authorized Lab cycle budget through cycle ${state.authorizedLabCycles} exhausted`;
      this.store.writeState({
        ...state,
        revision: state.revision + 1,
        lifecycle: "PAUSED",
        reason,
        pauseKind: "CYCLE_BUDGET_EXHAUSTED",
      }, "CYCLE_BUDGET_EXHAUSTED");
      this.writeOutcome("PAUSED", reason);
      return;
    }
    const goalRef = state.activeGoalRecordRef ?? this.createGoalRecord(state);
    const invocation = this.prepareLabInvocation(run, goalRef);
    const executionState = this.store.readState();
    const sourceHistory = this.history();
    const execution = await this.runPersistentGoal(run, goalRef, invocation);
    this.assertAuthorityUnchanged(executionState, sourceHistory);
    const inspection = this.inspectLabArtifacts(execution.goal);
    if (inspection.resultRef) {
      this.adoptLabResult(execution.goal, execution.invocation, inspection);
      return;
    }
    this.pauseLabGoal(execution.goal, execution.invocation, inspection);
  }

  private createGoalRecord(state: DirectionStateFile): string {
    const goalId = this.store.newId("direction-lab-goal");
    const cycleRef = `workspace/cycles/${state.cycle}`;
    const cycleSourceRef = `${cycleRef}/source`;
    for (const directory of [
      cycleRef,
      cycleSourceRef,
      `${cycleRef}/shards`,
      `${cycleRef}/raw`,
      `${cycleRef}/analysis`,
      `${cycleRef}/freeze`,
    ]) this.store.ensureDir(directory);
    const root = `lab_goals/cycle-${state.cycle}/${goalId}`;
    const goalRef = `${root}/goal.json`;
    const outputRef = `${cycleRef}/result.md`;
    const checkpointRef = `${cycleRef}/checkpoint.json`;
    const bindingRef = `${cycleRef}/cycle_binding.json`;
    const binding: LabCycleBinding = {
      formatVersion: 1,
      cycle: state.cycle,
      goalId,
      contractRevision: state.activeContractRevision,
      contractRef: state.activeContractRef!,
      contractHash: state.activeContractHash!,
      sourceDecisionRef: state.latestDecisionRef!,
      resultRef: outputRef,
      checkpointRef,
      cycleSourceRef,
      createdAt: nowIso(),
    };
    this.store.writeJson(bindingRef, binding);
    const record: DirectionGoalRecord = {
      goalId,
      cycle: state.cycle,
      contractRevision: state.activeContractRevision,
      contractRef: state.activeContractRef!,
      contractHash: state.activeContractHash!,
      sourceDecisionRef: state.latestDecisionRef!,
      workspaceRef: "workspace",
      cycleRef,
      cycleSourceRef,
      outputRef,
      checkpointRef,
      bindingRef,
      bindingHash: this.store.sha256(bindingRef),
      providerThreadId: null,
      providerTurnIds: [],
      goalStatus: "pending",
      invocationRefs: [],
      activeInvocationRef: null,
      startedAt: null,
      completedAt: null,
      tokensUsed: 0,
      timeUsedSeconds: 0,
      error: null,
    };
    this.store.writeJson(goalRef, record);
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      activeGoalRecordRef: goalRef,
    }, "LAB_GOAL_RECORD_CREATED");
    this.store.appendEvent("LAB_GOAL_RECORD_CREATED", [goalRef, record.contractRef, bindingRef]);
    return goalRef;
  }

  private prepareLabInvocation(
    run: DirectionRunFile,
    goalRef: string,
  ): LabGoalInvocationRecord {
    const goal = this.store.readGoal(goalRef);
    const ordinal = goal.invocationRefs.length + 1;
    const invocationId = this.store.newId("lab-invocation");
    const root = goalRef.replace(/goal\.json$/, `invocations/${String(ordinal).padStart(3, "0")}-${invocationId}`);
    const invocationRef = `${root}/invocation.json`;
    const startedAt = nowIso();
    const deadlineAt = new Date(Date.now() + run.budgets.lab.hardTimeoutMs).toISOString();
    const invocation: LabGoalInvocationRecord = {
      invocationId,
      ordinal,
      goalRef,
      cycle: goal.cycle,
      contractRevision: goal.contractRevision,
      contractHash: goal.contractHash,
      resumed: Boolean(goal.providerThreadId),
      resumeThreadId: goal.providerThreadId,
      startedAt,
      deadlineAt,
      completedAt: null,
      promptRef: `${root}/prompt.md`,
      runtimeRef: `${root}/runtime.jsonl`,
      providerRawRef: `${root}/provider_raw.jsonl`,
      providerFinalRef: null,
      providerThreadId: goal.providerThreadId,
      providerTurnIds: [],
      providerStatus: null,
      interruptionKind: null,
      operatorPauseRequested: false,
      checkpointRef: null,
      resultRef: null,
      tokensUsed: 0,
      timeUsedSeconds: 0,
      error: null,
    };
    this.store.writeText(invocation.promptRef, buildLabPrompt(this.store, run, goal, invocation));
    this.store.writeJson(invocationRef, invocation);
    this.store.writeJson(goalRef, {
      ...goal,
      goalStatus: "active",
      activeInvocationRef: invocationRef,
      invocationRefs: [...goal.invocationRefs, invocationRef],
      startedAt: goal.startedAt ?? startedAt,
      completedAt: null,
    });
    const state = this.store.readState();
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      activeLabInvocationRef: invocationRef,
    }, invocation.resumed ? "LAB_GOAL_RESUMED" : "LAB_GOAL_INVOCATION_STARTED");
    this.store.appendEvent(
      invocation.resumed ? "LAB_GOAL_RESUMED" : "LAB_GOAL_INVOCATION_STARTED",
      [goalRef, invocationRef, invocation.promptRef],
      { ordinal, deadlineAt, resumeThreadId: invocation.resumeThreadId },
    );
    return invocation;
  }

  private async runPersistentGoal(
    run: DirectionRunFile,
    goalRef: string,
    prepared: LabGoalInvocationRecord,
  ): Promise<{
    goal: DirectionGoalRecord;
    invocation: LabGoalInvocationRecord;
    result: RawGoalResult;
  }> {
    const goal = this.store.readGoal(goalRef);
    const prompt = this.store.readText(prepared.promptRef);
    let operatorPauseRequested = false;
    let interruptRequested = false;
    let result: RawGoalResult;
    const dispatch: GoalDispatch = {
      experimentId: prepared.invocationId,
      role: "DIRECTION_LAB_GOAL",
      prompt,
      objective: compactGoalObjective(
        this.store.readJson<ExperimentContractRecord>(goal.contractRef),
        goal.cycle,
      ),
      cwd: run.projectRoot,
      model: run.model,
      effort: "high",
      tokenBudget: null,
      timeoutProfile: run.budgets.lab,
      resumeThreadId: goal.providerThreadId,
      developerInstructions: labDeveloperInstructions(),
      onRuntimeEvent: (event) => this.persistGoalRuntimeEvent(
        goalRef,
        goal.activeInvocationRef!,
        prepared.runtimeRef,
        prepared.providerRawRef,
        event,
      ),
    };
    const monitor = setInterval(() => {
      const request = this.store.readPauseRequest();
      if (!request || interruptRequested) return;
      operatorPauseRequested = true;
      interruptRequested = true;
      this.store.appendEvent("LAB_GOAL_OPERATOR_INTERRUPT_REQUESTED", [goalRef], request);
      void Promise.resolve(this.runtime.interruptGoal?.(request.reason)).catch((error) => {
        this.store.appendEvent("LAB_GOAL_OPERATOR_INTERRUPT_FAILED", [goalRef], {
          error: errorMessage(error),
        });
      });
    }, run.budgets.controlPollMs);
    try {
      if (!this.runtime.runGoal) throw new Error("runtime does not support persistent Goals");
      result = await this.runtime.runGoal(dispatch);
    } catch (error) {
      const current = this.store.readGoal(goalRef);
      result = {
        goalStatus: "runtimeFailed",
        finalText: "",
        providerThreadId: current.providerThreadId,
        providerTurnIds: current.providerTurnIds,
        tokensUsed: current.tokensUsed,
        timeUsedSeconds: current.timeUsedSeconds,
        failureKind: "PROVIDER_ERROR",
        error: errorMessage(error),
      };
    } finally {
      clearInterval(monitor);
    }

    const providerFinalRef = prepared.promptRef.replace(/prompt\.md$/, "provider_final.md");
    this.store.writeText(providerFinalRef, result.finalText || "NO_PROVIDER_FINAL_OUTPUT\n");
    const persistedGoal = this.store.readGoal(goalRef);
    const persistedInvocation = this.store.readJson<LabGoalInvocationRecord>(
      persistedGoal.activeInvocationRef!,
    );
    const interruptionKind = normalizeGoalFailure(result);
    const invocation: LabGoalInvocationRecord = {
      ...persistedInvocation,
      completedAt: nowIso(),
      providerFinalRef,
      providerThreadId: result.providerThreadId ?? persistedGoal.providerThreadId,
      providerTurnIds: result.providerTurnIds.length > 0
        ? result.providerTurnIds
        : persistedInvocation.providerTurnIds,
      providerStatus: result.goalStatus,
      interruptionKind,
      operatorPauseRequested,
      tokensUsed: result.tokensUsed,
      timeUsedSeconds: result.timeUsedSeconds,
      error: result.error,
    };
    const completedGoal: DirectionGoalRecord = {
      ...persistedGoal,
      providerThreadId: result.providerThreadId ?? persistedGoal.providerThreadId,
      providerTurnIds: unique([...persistedGoal.providerTurnIds, ...result.providerTurnIds]),
      goalStatus: result.goalStatus,
      activeInvocationRef: null,
      tokensUsed: result.tokensUsed,
      timeUsedSeconds: result.timeUsedSeconds,
      error: result.error,
    };
    this.store.writeJson(persistedGoal.activeInvocationRef!, invocation);
    this.store.writeJson(goalRef, completedGoal);
    this.store.appendEvent("LAB_GOAL_INVOCATION_COMPLETED", [goalRef, persistedGoal.activeInvocationRef!], {
      providerStatus: result.goalStatus,
      interruptionKind,
      elapsedSeconds: result.timeUsedSeconds,
    });
    return { goal: completedGoal, invocation, result };
  }

  private inspectLabArtifacts(goal: DirectionGoalRecord): LabArtifactInspection {
    const errors: string[] = [];
    if (!this.store.exists(goal.bindingRef) || this.store.sha256(goal.bindingRef) !== goal.bindingHash) {
      errors.push(`cycle binding hash mismatch: ${goal.bindingRef}`);
    } else {
      const binding = this.store.readJson<LabCycleBinding>(goal.bindingRef);
      if (
        binding.cycle !== goal.cycle ||
        binding.goalId !== goal.goalId ||
        binding.contractRevision !== goal.contractRevision ||
        binding.contractHash !== goal.contractHash ||
        binding.resultRef !== goal.outputRef ||
        binding.checkpointRef !== goal.checkpointRef
      ) errors.push(`cycle binding identity mismatch: ${goal.bindingRef}`);
    }
    let checkpointRef: string | null = null;
    if (this.store.exists(goal.checkpointRef)) {
      const checkpointErrors = this.validateCheckpoint(goal);
      if (checkpointErrors.length === 0) checkpointRef = goal.checkpointRef;
      else errors.push(...checkpointErrors);
    }
    const resultRef = errors.some((error) => error.includes("cycle binding"))
      ? null
      : this.store.isNonEmptyFile(goal.outputRef)
      ? goal.outputRef
      : null;
    return { resultRef, checkpointRef, errors };
  }

  private validateCheckpoint(goal: DirectionGoalRecord): string[] {
    const errors: string[] = [];
    let checkpoint: LabCheckpoint;
    try {
      checkpoint = this.store.readJson<LabCheckpoint>(goal.checkpointRef);
    } catch (error) {
      return [`checkpoint is not valid JSON: ${errorMessage(error)}`];
    }
    if (
      checkpoint.cycle !== goal.cycle ||
      checkpoint.contractRevision !== goal.contractRevision ||
      checkpoint.contractHash !== goal.contractHash
    ) errors.push(`checkpoint contract binding mismatch: ${goal.checkpointRef}`);
    for (const [name, value] of [
      ["phase", checkpoint.phase],
      ["lastProgressAt", checkpoint.lastProgressAt],
      ["resumeAction", checkpoint.resumeAction],
    ] as const) {
      if (typeof value !== "string" || !value.trim()) {
        errors.push(`checkpoint.${name} must be a non-empty string`);
      }
    }
    for (const [name, value] of [
      ["completedUnits", checkpoint.completedUnits],
      ["validatedArtifacts", checkpoint.validatedArtifacts],
      ["partialExcludedRefs", checkpoint.partialExcludedRefs],
    ] as const) {
      if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
        errors.push(`checkpoint.${name} must be a string array`);
      }
    }
    if (checkpoint.activeCommand !== null && typeof checkpoint.activeCommand !== "string") {
      errors.push("checkpoint.activeCommand must be a string or null");
    }
    return errors;
  }

  private adoptLabResult(
    goal: DirectionGoalRecord,
    invocation: LabGoalInvocationRecord,
    inspection: LabArtifactInspection,
  ): void {
    const invocationRef = goal.invocationRefs.at(-1)!;
    const adoptedAfterInterruption =
      invocation.providerStatus !== "complete" || invocation.interruptionKind !== null;
    this.store.writeJson(invocationRef, {
      ...invocation,
      resultRef: inspection.resultRef,
      checkpointRef: inspection.checkpointRef,
    });
    const completedGoal = {
      ...goal,
      completedAt: nowIso(),
      error: invocation.error,
    };
    this.store.writeJson(invocation.goalRef, completedGoal);
    const history: ExperimentHistoryEntry = {
      kind: "EXPERIMENT",
      cycle: goal.cycle,
      contractRevision: goal.contractRevision,
      contractRef: goal.contractRef,
      contractHash: goal.contractHash,
      sourceDecisionRef: goal.sourceDecisionRef,
      goalRecordRef: invocation.goalRef,
      invocationRefs: [...goal.invocationRefs],
      providerStatus: goal.goalStatus,
      interruptionKind: invocation.interruptionKind,
      resultRef: inspection.resultRef!,
      checkpointRef: inspection.checkpointRef,
      adoptedAfterInterruption,
      tokensUsed: goal.tokensUsed,
      timeUsedSeconds: goal.timeUsedSeconds,
      completedAt: nowIso(),
    };
    this.store.appendJsonLine("history.jsonl", history);
    const requestRef = this.createJudgeRequest(
      "POST_LAB_REVIEW",
      "独立审阅当前原子合同的结果、明文 stop condition、证据有效性、实际范围和未决项。",
      goal.sourceDecisionRef,
      inspection.resultRef!,
    );
    const state = this.store.readState();
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "RUNNING",
      node: "JUDGE",
      transitions: state.transitions + 1,
      reason: null,
      pauseKind: null,
      activeGoalRecordRef: null,
      activeLabInvocationRef: null,
      activeJudgeRequestRef: requestRef,
      latestLabResultRef: inspection.resultRef,
      latestCheckpointRef: inspection.checkpointRef,
    }, adoptedAfterInterruption ? "LAB_RESULT_ADOPTED_AFTER_INTERRUPTION" : "LAB_RESULT_INDEXED");
    this.store.appendEvent(
      adoptedAfterInterruption ? "LAB_RESULT_ADOPTED_AFTER_INTERRUPTION" : "LAB_RESULT_INDEXED",
      [invocation.goalRef, invocationRef, inspection.resultRef!, requestRef],
      { providerStatus: invocation.providerStatus, interruptionKind: invocation.interruptionKind },
    );
    this.store.clearPauseRequest();
  }

  private pauseLabGoal(
    goal: DirectionGoalRecord,
    invocation: LabGoalInvocationRecord,
    inspection: LabArtifactInspection,
  ): void {
    const invocationRef = goal.invocationRefs.at(-1)!;
    this.store.writeJson(invocationRef, {
      ...invocation,
      checkpointRef: inspection.checkpointRef,
      resultRef: null,
    });
    if (inspection.checkpointRef) {
      this.store.appendEvent("LAB_GOAL_CHECKPOINT_INDEXED", [invocation.goalRef, inspection.checkpointRef], {
        invocationOrdinal: invocation.ordinal,
      });
    }
    if (invocation.interruptionKind === "IDLE_TIMEOUT" || invocation.interruptionKind === "HARD_TIMEOUT") {
      this.store.appendEvent("LAB_GOAL_TIMEOUT", [invocation.goalRef, invocationRef], {
        timeoutKind: invocation.interruptionKind,
        checkpointRef: inspection.checkpointRef,
      });
    }
    const operatorPause = invocation.operatorPauseRequested ||
      invocation.interruptionKind === "OPERATOR_INTERRUPT";
    const artifactReason = inspection.errors.length > 0
      ? `; artifact validation: ${inspection.errors.join("; ")}`
      : "";
    const reason = operatorPause
      ? `operator requested pause${inspection.checkpointRef ? "; checkpoint indexed" : "; no valid checkpoint"}${artifactReason}`
      : inspection.checkpointRef
      ? `Lab invocation ended with ${invocation.providerStatus}; checkpoint indexed for same-Goal resume${artifactReason}`
      : `Lab invocation ended with ${invocation.providerStatus} and produced neither an adoptable result nor a valid checkpoint${artifactReason}`;
    const state = this.store.readState();
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "PAUSED",
      node: "LAB_GOAL",
      reason,
      pauseKind: operatorPause ? "OPERATOR_REQUESTED" : "LAB_GOAL_PAUSED",
      activeLabInvocationRef: null,
      latestCheckpointRef: inspection.checkpointRef,
    }, operatorPause ? "RUN_PAUSED" : "LAB_GOAL_PAUSED");
    this.store.appendEvent("LAB_GOAL_PAUSED", [invocation.goalRef, invocationRef], {
      providerStatus: invocation.providerStatus,
      interruptionKind: invocation.interruptionKind,
      checkpointRef: inspection.checkpointRef,
      errors: inspection.errors,
    });
    this.store.clearPauseRequest();
    this.writeOutcome("PAUSED", reason);
  }

  private createJudgeRequest(
    kind: "POST_LAB_REVIEW" | "DECISION_REQUESTED_REVIEW",
    focus: string,
    sourceDecisionRef: string,
    labResultRef?: string,
  ): string {
    const state = this.store.readState();
    const requestId = this.store.newId("judge-request");
    const ref = `judge_requests/${requestId}.json`;
    this.store.writeJson(ref, {
      requestId,
      kind,
      focus,
      sourceDecisionRef,
      contractRevision: state.activeContractRef ? state.activeContractRevision : null,
      contractRef: state.activeContractRef,
      contractHash: state.activeContractHash,
      labResultRef: labResultRef ?? state.latestLabResultRef,
      checkpointRef: state.latestCheckpointRef,
      createdAt: nowIso(),
    });
    return ref;
  }

  private async executeJudge(): Promise<void> {
    const run = this.store.readRun();
    const sourceState = this.store.readState();
    const sourceHistory = this.history();
    if (!sourceState.activeJudgeRequestRef) {
      throw new Error("JUDGE node lacks an active review request");
    }
    const ordinal = sourceHistory.filter((entry) => entry.kind === "JUDGMENT").length + 1;
    const snapshotRoot = `snapshots/judge-${ordinal}-${this.store.newId("snapshot")}`;
    const stateSnapshotRef = `${snapshotRoot}/state.json`;
    const historySnapshotRef = `${snapshotRoot}/trajectory.json`;
    this.store.writeJson(stateSnapshotRef, sourceState);
    this.store.writeJson(historySnapshotRef, this.trajectorySnapshot(run, sourceState));

    let correction = "";
    let outputFailures = 0;
    let runtimeFailures = 0;
    let attempt = 0;
    while (true) {
      attempt += 1;
      const turnId = this.store.newId("evidence-judge");
      const root = `judges/judgment-${ordinal}/attempt-${attempt}-${turnId}`;
      const recordRef = `${root}/turn.json`;
      const promptRef = `${root}/prompt.md`;
      const runtimeRef = `${root}/runtime.jsonl`;
      const prompt = buildJudgePrompt(
        this.store,
        run,
        stateSnapshotRef,
        historySnapshotRef,
        sourceState.activeJudgeRequestRef,
        correction,
      );
      this.store.writeText(promptRef, prompt);
      const record = freshRecord(
        turnId,
        "EVIDENCE_JUDGE",
        ordinal,
        attempt,
        promptRef,
        stateSnapshotRef,
        historySnapshotRef,
        null,
        runtimeRef,
      );
      this.store.writeJson(recordRef, record);
      this.store.appendEvent("EVIDENCE_JUDGE_STARTED", [recordRef, promptRef]);
      const raw = await this.runFreshTurn(run, {
        turnId,
        role: "EVIDENCE_JUDGE",
        prompt,
        effort: "high",
        timeoutProfile: run.budgets.judge,
        developerInstructions: judgeDeveloperInstructions(),
        runtimeRef,
      });
      this.assertAuthorityUnchanged(sourceState, sourceHistory);
      const rawOutputRef = `${root}/raw_output.txt`;
      this.store.writeText(rawOutputRef, raw.text || raw.partialText || "");
      const finished = finishFreshRecord(record, raw, rawOutputRef);
      if (raw.status !== "completed") {
        runtimeFailures += 1;
        const errorRef = `${root}/error.json`;
        this.store.writeJson(errorRef, runtimeError(raw));
        this.store.writeJson(recordRef, { ...finished, errorRef });
        this.store.appendEvent("EVIDENCE_JUDGE_RUNTIME_FAILED", [recordRef, errorRef]);
        if (runtimeFailures <= run.budgets.maxRuntimeRetries) continue;
        this.pauseFreshTurn(
          "JUDGE",
          "JUDGE_RETRY_EXHAUSTED",
          "Evidence Judge exceeded runtime retry budget",
        );
        return;
      }
      const parsed = parseJudgeResult(raw.text);
      if (!parsed.value) {
        outputFailures += 1;
        const errorRef = `${root}/error.json`;
        this.store.writeJson(errorRef, { kind: "CORE_PROTOCOL", errors: parsed.errors });
        this.store.writeJson(recordRef, { ...finished, errorRef });
        this.store.appendEvent("EVIDENCE_JUDGE_OUTPUT_INVALID", [recordRef, errorRef]);
        if (outputFailures <= run.budgets.maxOutputRetries) {
          correction = judgeCorrectionSuffix(parsed.errors);
          continue;
        }
        this.pauseFreshTurn(
          "JUDGE",
          "JUDGE_RETRY_EXHAUSTED",
          "Evidence Judge exceeded output-correction retry budget",
        );
        return;
      }
      const resultRef = `${root}/judgment.json`;
      this.store.writeJson(resultRef, parsed.value);
      this.store.writeJson(recordRef, { ...finished, resultRef });
      this.store.appendEvent("EVIDENCE_JUDGE_ACCEPTED", [recordRef, resultRef]);
      this.completeJudgment(parsed.value, resultRef, ordinal, sourceState);
      return;
    }
  }

  private completeJudgment(
    value: JudgeResult,
    judgmentRef: string,
    ordinal: number,
    sourceState: DirectionStateFile,
  ): void {
    const history: JudgmentHistoryEntry = {
      kind: "JUDGMENT",
      ordinal,
      contractRevision: sourceState.activeContractRef
        ? sourceState.activeContractRevision
        : null,
      contractRef: sourceState.activeContractRef,
      labResultRef: sourceState.latestLabResultRef,
      requestRef: sourceState.activeJudgeRequestRef!,
      judgmentRef,
      assessment: value.assessment,
      evidenceScope: value.evidenceScope,
      reason: value.reason,
      completedAt: nowIso(),
    };
    this.store.appendJsonLine("history.jsonl", history);
    const state = this.store.readState();
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "RUNNING",
      node: "DECISION",
      transitions: state.transitions + 1,
      reason: null,
      pauseKind: null,
      activeJudgeRequestRef: null,
      latestJudgeRef: judgmentRef,
      evidenceScope: value.evidenceScope,
    }, "EVIDENCE_JUDGMENT_INDEXED");
  }

  private async runFreshTurn(
    run: DirectionRunFile,
    options: {
      turnId: string;
      role: "EXPERIMENT_DECISION" | "EVIDENCE_JUDGE";
      prompt: string;
      effort: "high" | "max";
      timeoutProfile: DirectionRunFile["budgets"]["decision"];
      developerInstructions: string;
      runtimeRef: string;
    },
  ): Promise<RawTurnResult> {
    try {
      return await this.runtime.run({
        turnId: options.turnId,
        role: options.role,
        prompt: options.prompt,
        outputSchema: null,
        cwd: run.projectRoot,
        model: run.model,
        effort: options.effort,
        timeoutProfile: options.timeoutProfile,
        maxInputTokens: run.budgets.maxInputTokens,
        maxOutputTokens: run.budgets.maxOutputTokens,
        developerInstructions: options.developerInstructions,
        onRuntimeEvent: (event) => this.store.runtimeEvent(options.runtimeRef, event),
      });
    } catch (error) {
      return failedTurn(error);
    }
  }

  private persistGoalRuntimeEvent(
    goalRef: string,
    invocationRef: string,
    runtimeRef: string,
    providerRawRef: string,
    event: GoalRuntimePersistenceEvent,
  ): void {
    if (event.type === "goal_raw_event") {
      this.store.runtimeEvent(providerRawRef, event);
    } else if (event.type === "goal_message_completed") {
      this.store.runtimeEvent(providerRawRef, event);
      this.store.runtimeEvent(runtimeRef, {
        ...event,
        text: undefined,
        textChars: event.text.length,
        textSha256: sha256Text(event.text),
      });
    } else if (event.type === "goal_tool") {
      this.store.runtimeEvent(providerRawRef, event);
      this.store.runtimeEvent(runtimeRef, {
        type: event.type,
        at: event.at,
        phase: event.phase,
        providerTurnId: event.providerTurnId,
        toolName: event.event.toolName,
        status: event.event.status,
      });
    } else {
      this.store.runtimeEvent(runtimeRef, event);
    }
    if (!this.store.exists(goalRef) || !this.store.exists(invocationRef)) return;
    const goal = this.store.readGoal(goalRef);
    const invocation = this.store.readJson<LabGoalInvocationRecord>(invocationRef);
    if (event.type === "goal_provider_started") {
      this.store.writeJson(goalRef, { ...goal, providerThreadId: event.threadId });
      this.store.writeJson(invocationRef, { ...invocation, providerThreadId: event.threadId });
    } else if (event.type === "goal_turn_started") {
      this.store.writeJson(goalRef, {
        ...goal,
        providerThreadId: event.threadId,
        providerTurnIds: unique([...goal.providerTurnIds, event.providerTurnId]),
      });
      this.store.writeJson(invocationRef, {
        ...invocation,
        providerThreadId: event.threadId,
        providerTurnIds: unique([...invocation.providerTurnIds, event.providerTurnId]),
      });
    } else if (event.type === "goal_status") {
      this.store.writeJson(goalRef, { ...goal, goalStatus: event.status });
      this.store.writeJson(invocationRef, { ...invocation, providerStatus: event.status });
    }
  }

  private labRuntimeEnvelope(
    run: DirectionRunFile,
    state: DirectionStateFile,
  ): LabRuntimeEnvelope {
    const activeGoal = state.activeGoalRecordRef && this.store.exists(state.activeGoalRecordRef)
      ? this.store.readGoal(state.activeGoalRecordRef)
      : null;
    const priorExperiment = [...this.history()].reverse().find(
      (entry): entry is ExperimentHistoryEntry => entry.kind === "EXPERIMENT",
    );
    return {
      idleTimeoutMs: run.budgets.lab.idleTimeoutMs,
      hardTimeoutMs: run.budgets.lab.hardTimeoutMs,
      resultReserveMs: run.budgets.labResultReserveMs,
      maxContractMinutes: maxContractMinutes(run),
      currentCycle: state.cycle,
      remainingAuthorizedCycles: Math.max(0, state.authorizedLabCycles - state.cycle),
      activeGoalTimeUsedSeconds: activeGoal?.timeUsedSeconds ?? 0,
      priorLabTimeUsedSeconds: priorExperiment?.timeUsedSeconds ?? 0,
      latestCheckpointRef: state.latestCheckpointRef,
      latestLabResultRef: state.latestLabResultRef,
    };
  }

  private trajectorySnapshot(
    run: DirectionRunFile,
    state: DirectionStateFile,
  ): Record<string, unknown> {
    const activeGoal = state.activeGoalRecordRef && this.store.exists(state.activeGoalRecordRef)
      ? this.store.readGoal(state.activeGoalRecordRef)
      : null;
    return {
      sourceDirectionRef: run.inputs.directionResult.path,
      sourceDirectionId: run.source.directionId,
      sourceDirectionRevision: run.source.directionRevision,
      experimentPolicyRef: run.inputs.experimentPolicy.path,
      activeContractRevision: state.activeContractRevision,
      activeContractRef: state.activeContractRef,
      activeContractHash: state.activeContractHash,
      activeGoalRecordRef: state.activeGoalRecordRef,
      activeLabInvocationRef: state.activeLabInvocationRef,
      latestLabResultRef: state.latestLabResultRef,
      latestCheckpointRef: state.latestCheckpointRef,
      latestJudgeRef: state.latestJudgeRef,
      activeGoalSummary: activeGoal
        ? {
          goalId: activeGoal.goalId,
          cycle: activeGoal.cycle,
          goalStatus: activeGoal.goalStatus,
          providerThreadId: activeGoal.providerThreadId,
          invocationRefs: activeGoal.invocationRefs,
          timeUsedSeconds: activeGoal.timeUsedSeconds,
        }
        : null,
      history: this.history(),
    };
  }

  private finish(value: ExperimentDecisionResult, decisionRef: string): void {
    if (!isTerminalDecision(value.decision)) {
      throw new Error(`non-terminal decision passed to finish: ${value.decision}`);
    }
    const run = this.store.readRun();
    const state = this.store.readState();
    const history = this.history();
    const contractRefs = history
      .filter((entry): entry is DecisionHistoryEntry =>
        entry.kind === "DECISION" && entry.decision === "RUN_LAB")
      .map((entry) => entry.contractRef!);
    const experimentResultRefs = history
      .filter((entry): entry is ExperimentHistoryEntry => entry.kind === "EXPERIMENT")
      .map((entry) => entry.resultRef);
    const judgmentRefs = history
      .filter((entry): entry is JudgmentHistoryEntry => entry.kind === "JUDGMENT")
      .map((entry) => entry.judgmentRef);
    const outcome = value.decision === "COMPLETE_SUPPORT"
      ? "SUPPORTED"
      : value.decision === "COMPLETE_REJECT"
      ? "NOT_SUPPORTED"
      : "RETURN_TO_LEARNING";
    const handoff: DirectionHandoff = {
      sourceDirectionRef: "inputs/direction_result.json",
      directionId: run.source.directionId,
      directionRevision: run.source.directionRevision,
      directionTargetRef: "inputs/direction_target.md",
      sourceEvidenceManifestRef: "inputs/evidence_manifest.json",
      experimentPolicyRef: "inputs/experiment_policy.md",
      outcome,
      finalDecision: value.decision,
      evidenceScope: value.evidenceScope,
      summary: value.reason,
      activeContractRevision: state.activeContractRevision,
      activeContractRef: state.activeContractRef,
      activeContractHash: state.activeContractHash,
      contractRefs: unique(contractRefs),
      experimentResultRefs,
      judgmentRefs,
      finalDecisionRef: decisionRef,
      reportRef: "final/report.md",
    };
    this.store.writeJson("final/handoff.json", handoff);
    this.store.writeText("final/report.md", renderFinalReport(run, value, decisionRef, history));
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "FINISHED",
      node: null,
      transitions: state.transitions + 1,
      reason: null,
      pauseKind: null,
      activeGoalRecordRef: null,
      activeLabInvocationRef: null,
      activeJudgeRequestRef: null,
      latestDecisionRef: decisionRef,
      finalDecision: value.decision,
      evidenceScope: value.evidenceScope,
    }, "RUN_FINISHED");
    this.writeOutcome("FINISHED", null);
  }

  private pauseFreshTurn(
    node: "DECISION" | "JUDGE",
    pauseKind: "DECISION_RETRY_EXHAUSTED" | "JUDGE_RETRY_EXHAUSTED",
    reason: string,
  ): void {
    const state = this.store.readState();
    this.store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "PAUSED",
      node,
      reason,
      pauseKind,
    }, pauseKind);
    this.writeOutcome("PAUSED", reason);
  }

  private assertWritableAndFrozen(): void {
    const run = this.store.readRun();
    const state = this.store.readState();
    if (!this.isCurrentFormat()) {
      throw new Error(
        `format v${run.formatVersion} is audit-only; initialize a v${DIRECTION_EXPERIMENT_FORMAT_VERSION} run`,
      );
    }
    const errors = verifyFrozenFiles(this.store);
    if (state.activeContractRef || state.activeContractHash) {
      if (!state.activeContractRef || !state.activeContractHash) {
        errors.push("active contract ref/hash must be both present or both null");
      } else if (!this.store.exists(state.activeContractRef)) {
        errors.push(`active experiment contract is missing: ${state.activeContractRef}`);
      } else if (this.store.sha256(state.activeContractRef) !== state.activeContractHash) {
        errors.push(`active experiment contract hash mismatch: ${state.activeContractRef}`);
      }
    }
    if (errors.length > 0) throw new Error(errors.join("; "));
  }

  private assertAuthorityUnchanged(
    expectedState: DirectionStateFile,
    expectedHistory: DirectionHistoryEntry[],
  ): void {
    if (!isDeepStrictEqual(this.store.readState(), expectedState)) {
      throw new Error("Script authority changed during Agent execution: state.json");
    }
    if (!isDeepStrictEqual(this.history(), expectedHistory)) {
      throw new Error("Script authority changed during Agent execution: history.jsonl");
    }
    this.assertWritableAndFrozen();
  }

  private isCurrentFormat(): boolean {
    const run = this.store.readRun();
    const state = this.store.readState();
    return run.formatVersion === DIRECTION_EXPERIMENT_FORMAT_VERSION &&
      state.formatVersion === DIRECTION_EXPERIMENT_FORMAT_VERSION &&
      run.workflow === "DIRECTION_EXPERIMENT_LOOP";
  }

  private history(): DirectionHistoryEntry[] {
    if (!this.store.exists("history.jsonl")) return [];
    return this.store.readText("history.jsonl")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as DirectionHistoryEntry);
  }

  private writeOutcome(
    workflowOutcome: DirectionLoopOutcome["workflowOutcome"],
    reason: string | null,
  ): void {
    const evidenceScope = this.store.readState().evidenceScope;
    this.store.writeJson("final/outcome.json", {
      workflowOutcome,
      reportRef: workflowOutcome === "FINISHED" ? "final/report.md" : null,
      handoffRef: workflowOutcome === "FINISHED" ? "final/handoff.json" : null,
      evidenceScope,
      reason,
    });
  }

  private currentOutcome(): DirectionLoopOutcome {
    const state = this.store.readState();
    if (state.lifecycle === "FINISHED") {
      return {
        workflowOutcome: "FINISHED",
        reportRef: "final/report.md",
        handoffRef: "final/handoff.json",
        evidenceScope: state.evidenceScope,
        reason: null,
      };
    }
    return {
      workflowOutcome: state.lifecycle === "FAILED" ? "FAILED" : "PAUSED",
      reportRef: null,
      handoffRef: null,
      evidenceScope: state.evidenceScope,
      reason: state.reason,
    };
  }
}

function freshRecord(
  turnId: string,
  role: FreshTurnRecord["role"],
  ordinal: number,
  attempt: number,
  promptRef: string,
  stateSnapshotRef: string,
  historySnapshotRef: string,
  runtimeEnvelopeRef: string | null,
  runtimeRef: string,
): FreshTurnRecord {
  return {
    turnId,
    role,
    ordinal,
    attempt,
    startedAt: nowIso(),
    completedAt: null,
    promptRef,
    stateSnapshotRef,
    historySnapshotRef,
    runtimeEnvelopeRef,
    rawOutputRef: null,
    resultRef: null,
    errorRef: null,
    runtimeRef,
    providerThreadId: null,
    providerTurnId: null,
    providerStatus: null,
  };
}

function finishFreshRecord(
  record: FreshTurnRecord,
  raw: RawTurnResult,
  rawOutputRef: string,
): FreshTurnRecord {
  return {
    ...record,
    completedAt: nowIso(),
    rawOutputRef,
    providerThreadId: raw.providerThreadId,
    providerTurnId: raw.providerTurnId,
    providerStatus: raw.status,
  };
}

function runtimeError(raw: RawTurnResult): Record<string, unknown> {
  return {
    kind: "RUNTIME",
    status: raw.status,
    failureKind: raw.failureKind,
    error: raw.error,
    outputCapture: raw.outputCapture,
  };
}

function compactGoalObjective(contract: ExperimentContractRecord, cycle: number): string {
  return `Atomic experiment cycle ${cycle}: ${contract.objective} Comparison: ${contract.comparison}`
    .replace(/\s+/g, " ")
    .slice(0, 3_500);
}

function maxContractMinutes(run: DirectionRunFile): number {
  return Math.max(
    1,
    Math.floor((run.budgets.lab.hardTimeoutMs - run.budgets.labResultReserveMs) / 60_000),
  );
}

function normalizeGoalFailure(result: RawGoalResult): GoalInterruptionKind {
  if (result.failureKind) return result.failureKind;
  if (result.goalStatus === "runtimeFailed") return "PROVIDER_ERROR";
  return null;
}

function renderFinalReport(
  run: DirectionRunFile,
  decision: ExperimentDecisionResult,
  decisionRef: string,
  history: DirectionHistoryEntry[],
): string {
  const contracts = history.filter((entry): entry is DecisionHistoryEntry =>
    entry.kind === "DECISION" && entry.decision === "RUN_LAB"
  );
  const experiments = history.filter((entry): entry is ExperimentHistoryEntry =>
    entry.kind === "EXPERIMENT"
  );
  const judgments = history.filter((entry): entry is JudgmentHistoryEntry =>
    entry.kind === "JUDGMENT"
  );
  return [
    "# Direction Experiment Report",
    "",
    `- Direction: ${run.source.directionId} revision ${run.source.directionRevision}`,
    `- Parent Anchor: ${run.source.parentAnchorId} revision ${run.source.parentAnchorRevision}`,
    `- Final Decision: ${decision.decision}`,
    `- Evidence scope: ${decision.evidenceScope}`,
    `- Final Decision record: ${decisionRef}`,
    "",
    "## Narrowest conclusion",
    "",
    decision.reason,
    "",
    "## Frozen source",
    "",
    "- [Direction](../inputs/direction_result.json)",
    "- [Readable target](../inputs/direction_target.md)",
    "- [Parent Anchor](../inputs/parent_anchor_result.json)",
    "- [Source review](../inputs/source_review_result.json)",
    "- [Evidence manifest](../inputs/evidence_manifest.json)",
    "- [Experiment policy](../inputs/experiment_policy.md)",
    "",
    "## Atomic experiment contracts",
    "",
    ...(contracts.length > 0
      ? contracts.map((entry) =>
        `- Contract ${entry.contractRevision}: [contract](../${entry.contractRef}); [Decision](../${entry.decisionRef}) — ${entry.reason}`
      )
      : ["No Lab contract was required."]),
    "",
    "## Lab evidence",
    "",
    ...(experiments.length > 0
      ? experiments.map((entry) =>
        `- Cycle ${entry.cycle}, contract ${entry.contractRevision}: ${entry.providerStatus}; invocations ${entry.invocationRefs.length}; interrupted adoption ${entry.adoptedAfterInterruption}; [goal](../${entry.goalRecordRef}); [result](../${entry.resultRef})`
      )
      : ["No Lab result was indexed."]),
    "",
    "## Independent judgments",
    "",
    ...(judgments.length > 0
      ? judgments.map((entry) =>
        `- Judgment ${entry.ordinal}: ${entry.assessment}, ${entry.evidenceScope}; [record](../${entry.judgmentRef}) — ${entry.reason}`
      )
      : ["No Evidence Judge Turn was required."]),
    "",
    "## Learning handoff",
    "",
    "See [handoff.json](handoff.json). The Script stores bindings and trajectory; the final semantic conclusion belongs to Experiment Decision.",
    "",
  ].join("\n");
}

function isTerminalDecision(value: string): value is TerminalExperimentDecision {
  return value === "COMPLETE_SUPPORT" ||
    value === "COMPLETE_REJECT" ||
    value === "RETURN_TO_LEARNING";
}

function failedTurn(error: unknown): RawTurnResult {
  return {
    status: "failed",
    text: "",
    providerThreadId: null,
    providerTurnId: null,
    usage: {
      inputTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      reasoningOutputTokens: 0,
      totalTokens: 0,
    },
    toolEvents: [],
    rawEvents: [],
    compacted: false,
    outputCapture: "NONE",
    partialText: "",
    failureKind: "PROVIDER_ERROR",
    interruptError: null,
    lastActivityAt: nowIso(),
    incrementalEventsPersisted: false,
    error: errorMessage(error),
    elapsedMs: 0,
  };
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function nowIso(): string {
  return new Date().toISOString();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
