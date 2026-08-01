import {
  buildContentPrompt,
  buildDecisionPrompt,
  type OutputCorrectionPrompt,
  type RuntimeRetryPrompt,
} from "./prompts.ts";
import {
  parseDecisionProtocol,
  parseStrictJsonObject,
} from "./protocol.ts";
import { renderFinalReport } from "./renderer.ts";
import { selectProtocolAgentMessage } from "./runtime.ts";
import {
  appendDecisionTrajectory,
  appendTerminalRuntimeTrajectory,
  rebuildResearchMemory,
  writeCheckpoint,
  writeDecisionObservation,
} from "./observations.ts";
import {
  validateCoreOutputForAction,
  validateReferenceTemplateForAction,
} from "./schemas.ts";
import { FileLoopStore } from "./store.ts";
import type {
  DecisionProtocolResult,
  CoreControlProjection,
  LoopRole,
  OutputError,
  OutputErrorReport,
  RawTurnResult,
  RoundAuthorizationRecord,
  RuntimePersistenceEvent,
  RuntimeToolEvent,
  ReviewResult,
  RunOutcome,
  SequenceStep,
  StateFile,
  TaskBinding,
  TurnFile,
  TurnTimeoutProfile,
  TokenUsage,
  ValidationAudit,
  WorkflowFailureKind,
  WorkResult,
} from "./types.ts";
import type { TurnRuntime } from "./types.ts";
import { CURRENT_FORMAT_VERSION } from "./types.ts";
import {
  allowedDecisions,
  commitPending,
  commitPreReview,
  computeRemainingRequirements,
  createDecisionContext,
  createReviewerBinding,
  createWorkerBinding,
  sequenceAfterDecision,
} from "./workflow.ts";

interface JsonTurnExecution {
  state: StateFile;
  turnRef: string;
  bindingRef: string;
  binding: TaskBinding;
  resultRef: string | null;
  result: WorkResult | ReviewResult | null;
  control: CoreControlProjection | null;
}

interface RawExecution {
  state: StateFile;
  turnRef: string;
  turn: TurnFile;
  raw: RawTurnResult;
}

export class RefactoredSemanticLoopController {
  private readonly store: FileLoopStore;
  private readonly runtime: TurnRuntime;

  constructor(
    store: FileLoopStore,
    runtime: TurnRuntime,
  ) {
    this.store = store;
    this.runtime = runtime;
  }

  async run(
    resume = false,
    additionalRounds?: number,
  ): Promise<RunOutcome> {
    this.store.acquireLock();
    try {
      const run = this.store.readRun();
      if (run.formatVersion !== CURRENT_FORMAT_VERSION) {
        throw new Error(
          `run formatVersion ${run.formatVersion} is read-only; initialize a formatVersion ${CURRENT_FORMAT_VERSION} work directory`,
        );
      }
      let state = this.store.readState();
      if (state.formatVersion !== CURRENT_FORMAT_VERSION) {
        throw new Error(
          `state.json is not formatVersion ${CURRENT_FORMAT_VERSION}`,
        );
      }
      if (
        resume &&
        additionalRounds !== undefined &&
        state.lifecycle !== "PAUSED"
      ) {
        throw new Error(
          "additionalRounds can only extend a PAUSED workflow",
        );
      }
      if (state.lifecycle === "FINISHED") return this.outcomeFromState(state);
      if (state.lifecycle === "FAILED") return this.outcomeFromState(state);
      if (state.lifecycle === "PAUSED") {
        if (!resume) return this.outcomeFromState(state);
        state = this.resumeState(state, additionalRounds);
      }

      state = this.reconcileInterruptedTurn(state);
      while (state.lifecycle === "RUNNING") {
        const step = state.sequence[0];
        if (!step) {
          state = this.failState(
            state,
            "Controller sequence is empty before FINISH_WORKFLOW",
          );
          break;
        }
        if (state.node !== step.role) {
          state = this.saveState(
            { ...state, node: step.role },
            "CONTROLLER_NODE_SELECTED",
          );
        }
        try {
          if (step.role === "WORKER") {
            state = await this.executeWorker(state, step);
          } else if (step.role === "REVIEWER") {
            state = await this.executeReviewer(state, step);
          } else {
            state = await this.executeDecision(state);
          }
        } catch (error) {
          state = this.failState(
            state,
            error instanceof Error ? error.message : String(error),
          );
        }
      }
      return this.outcomeFromState(state);
    } finally {
      await this.runtime.close?.();
      this.store.releaseLock();
    }
  }

  private async executeWorker(
    state: StateFile,
    step: SequenceStep,
  ): Promise<StateFile> {
    let bindingRef = step.bindingRef;
    if (!bindingRef) {
      const created = createWorkerBinding(this.store, state);
      bindingRef = created.bindingRef;
      state = this.bindHeadStep(state, bindingRef);
    }
    const success = await this.executeJsonTurn(
      state,
      bindingRef,
      "learning-loop-worker",
    );
    state = success.state;
    if (!success.resultRef || state.lifecycle !== "RUNNING") return state;
    if (success.control?.role !== "WORKER") {
      throw new Error("validated Worker result lacks a Worker control projection");
    }

    const binding = success.binding;
    state = this.saveState(
      {
        ...state,
        sequence: state.sequence.slice(1),
        node: state.sequence[1]?.role ?? null,
        activeTaskBindingRef: null,
        activeTurnRef: null,
        pending: {
          objectKind: binding.objectKind,
          objectId: binding.objectId,
          revision: binding.revision,
          parentAnchorId: binding.parentAnchorId,
          workTaskBindingRef: success.bindingRef,
          workTaskRef: binding.taskRef,
          workTurnRef: success.turnRef,
          workRef: success.resultRef,
          workOutcome: success.control.workOutcome,
          reviewTaskBindingRef: null,
          reviewTurnRef: null,
          reviewRef: null,
          reviewVerdict: null,
        },
        preReview: null,
      },
      "WORK_RESULT_READY",
    );
    return state;
  }

  private async executeReviewer(
    state: StateFile,
    step: SequenceStep,
  ): Promise<StateFile> {
    let bindingRef = step.bindingRef;
    if (!bindingRef) {
      const created = createReviewerBinding(
        this.store,
        state,
        step.mode === "PRE_REVIEW" ? "PRE_REVIEW" : "PAIR_REVIEW",
      );
      bindingRef = created.bindingRef;
      state = this.bindHeadStep(state, bindingRef);
    }
    const success = await this.executeJsonTurn(
      state,
      bindingRef,
      "learning-loop-reviewer",
    );
    state = success.state;
    if (!success.resultRef || state.lifecycle !== "RUNNING") return state;
    if (success.control?.role !== "REVIEWER") {
      throw new Error(
        "validated Reviewer result lacks a Reviewer control projection",
      );
    }

    const task = this.store.readJson<{
      inputs: { reviewTarget?: string };
    }>(success.binding.taskRef);
    if (step.mode === "PRE_REVIEW") {
      const workRef = task.inputs.reviewTarget;
      if (!workRef) throw new Error("PRE_REVIEW task lacks reviewTarget");
      commitPreReview(
        this.store,
        success.binding,
        workRef,
        success.resultRef,
        success.control.reviewVerdict,
      );
      rebuildResearchMemory(this.store, state);
      this.setTurnState(success.turnRef, "COMMITTED");
      state = this.saveState(
        {
          ...state,
          sequence: state.sequence.slice(1),
          node: state.sequence[1]?.role ?? null,
          activeTaskBindingRef: null,
          activeTurnRef: null,
          preReview: {
            objectKind: success.binding.objectKind,
            objectId: success.binding.objectId,
            revision: success.binding.revision,
            parentAnchorId: success.binding.parentAnchorId,
            workRef,
            workOutcome: this.store.readJson<WorkResult>(workRef).workOutcome,
            reviewTaskBindingRef: success.bindingRef,
            reviewTurnRef: success.turnRef,
            reviewRef: success.resultRef,
            reviewVerdict: success.control.reviewVerdict,
          },
        },
        "PRE_REVIEW_READY",
      );
      return state;
    }

    if (!state.pending) {
      throw new Error("PAIR_REVIEW completed without pending Worker result");
    }
    state = this.saveState(
      {
        ...state,
        sequence: state.sequence.slice(1),
        node: state.sequence[1]?.role ?? null,
        activeTaskBindingRef: null,
        activeTurnRef: null,
        pending: {
          ...state.pending,
          reviewTaskBindingRef: success.bindingRef,
          reviewTurnRef: success.turnRef,
          reviewRef: success.resultRef,
          reviewVerdict: success.control.reviewVerdict,
        },
      },
      "REVIEW_RESULT_READY",
    );
    return state;
  }

  private async executeDecision(state: StateFile): Promise<StateFile> {
    const allowed = allowedDecisions(this.store, state);
    const correction =
      state.correction?.role === "DECISION" ? state.correction : null;
    let contextRef: string;
    if (state.activeTurnRef) {
      const active = this.store.readTurn(state.activeTurnRef);
      if (!active.decisionContextRef) {
        throw new Error("active Decision Turn lacks decisionContextRef");
      }
      contextRef = active.decisionContextRef;
    } else if (correction) {
      const prior = this.store.readTurn(correction.retryOfTurnRef);
      if (!prior.decisionContextRef) {
        throw new Error("Decision correction lacks frozen decision context");
      }
      contextRef = prior.decisionContextRef;
    } else {
      const failed = this.latestDecisionRuntimeFailure(state.round);
      if (failed?.turn.decisionContextRef) {
        contextRef = failed.turn.decisionContextRef;
      } else {
        const contextId = this.store.newId("decision-context");
        contextRef = `contexts/${contextId}/decision_context.json`;
        const observationRef =
          `contexts/${contextId}/decision_observation.json`;
        writeDecisionObservation(this.store, state, allowed, observationRef);
        this.store.writeImmutableJson(
          contextRef,
          createDecisionContext(this.store, state, observationRef),
        );
        this.store.appendEvent("DECISION_CONTEXT_PROJECTED", [
          contextRef,
          observationRef,
        ]);
      }
    }

    const execution = await this.executeRawTurn({
      state,
      role: "DECISION",
      skill: "learning-loop-decision",
      taskBindingRef: null,
      decisionContextRef: contextRef,
      prompt: buildDecisionPrompt({
        contextPath: this.store.absolute(contextRef),
        allowed,
        correction: this.promptCorrection(correction),
        runtimeRetry: this.promptRuntimeRetry(
          "DECISION",
          null,
          contextRef,
        ),
      }),
      outputSchema: null,
    });
    state = execution.state;
    if (execution.raw.status !== "completed") return state;

    const parsed = parseDecisionProtocol(execution.raw.text, allowed);
    const errors = [...parsed.errors];
    if (!parsed.result || errors.length > 0) {
      return this.rejectAgentOutput(
        state,
        execution.turnRef,
        execution.turn,
        errors,
        "decision-line-protocol-v1",
      );
    }

    this.writeValidationAudit(execution.turn.validationAuditRef, [
      {
        check: "DECISION_PROTOCOL",
        path: "/",
        passed: true,
        message: "one allowed decision literal extracted",
      },
    ]);
    const controlRef = `turns/${execution.turn.turnId}/control.json`;
    this.store.writeJson(controlRef, {
      role: "DECISION",
      decision: parsed.result.decision,
      guidance: parsed.result.guidance,
    } satisfies CoreControlProjection);
    this.store.mutateTurn(execution.turnRef, (turn) => ({
      ...turn,
      turnState: "PENDING_DECISION",
      completedAt: new Date().toISOString(),
      resultRef: turn.rawOutputRef,
      controlRef,
    }));
    state = this.saveState(
      {
        ...state,
        activeTaskBindingRef: null,
        correction: null,
      },
      "DECISION_OUTPUT_VALID",
    );
    return this.applyDecision(
      state,
      execution.turnRef,
      parsed.result,
    );
  }

  private async executeJsonTurn(
    state: StateFile,
    bindingRef: string,
    skill:
      | "learning-loop-worker"
      | "learning-loop-reviewer",
  ): Promise<JsonTurnExecution> {
    const binding = this.store.readJson<TaskBinding>(bindingRef);
    const correction =
      state.correction?.role === binding.role ? state.correction : null;
    const execution = await this.executeRawTurn({
      state,
      role: binding.role,
      skill,
      taskBindingRef: bindingRef,
      decisionContextRef: null,
      prompt: buildContentPrompt({
        skillName: skill,
        taskPath: this.store.absolute(binding.taskRef),
        guidance: state.decisionGuidance,
        correction: this.promptCorrection(correction),
        runtimeRetry: this.promptRuntimeRetry(
          binding.role,
          bindingRef,
          null,
        ),
      }),
      outputSchema: null,
    });
    state = execution.state;
    if (execution.raw.status !== "completed") {
      return {
        state,
        turnRef: execution.turnRef,
        bindingRef,
        binding,
        resultRef: null,
        result: null,
        control: null,
      };
    }

    const parsed = parseStrictJsonObject(execution.raw.text);
    const core = parsed.parsed
      ? validateCoreOutputForAction(binding.action, parsed.parsed)
      : { control: null, errors: [] };
    const errors = [...parsed.errors, ...core.errors];
    if (!parsed.parsed || !core.control || errors.length > 0) {
      state = this.rejectAgentOutput(
        state,
        execution.turnRef,
        execution.turn,
        errors,
        binding.resultRefName,
      );
      return {
        state,
        turnRef: execution.turnRef,
        bindingRef,
        binding,
        resultRef: null,
        result: null,
        control: null,
      };
    }

    const checks: ValidationAudit["checks"] = [
      {
        check: "JSON_PARSE",
        path: "/",
        passed: true,
        message: "one bare JSON object parsed",
      },
      {
        check: "CORE_CONTROL",
        path: "/",
        passed: true,
        message: `${binding.role} core control literal validated`,
      },
    ];
    const advisories = validateReferenceTemplateForAction(
      binding.action,
      parsed.parsed,
    ).map((advisory) => ({
      check: "REFERENCE_TEMPLATE" as const,
      ...advisory,
    }));
    this.writeValidationAudit(
      execution.turn.validationAuditRef,
      checks,
      advisories,
    );
    const resultRef = `results/${execution.turn.turnId}.json`;
    const controlRef = `turns/${execution.turn.turnId}/control.json`;
    this.store.writeJson(resultRef, parsed.parsed);
    this.store.writeJson(controlRef, core.control);
    this.store.mutateTurn(execution.turnRef, (turn) => ({
      ...turn,
      turnState: "PENDING_DECISION",
      completedAt: new Date().toISOString(),
      resultRef,
      controlRef,
    }));
    state = this.saveState(
      {
        ...state,
        correction: null,
      },
      "AGENT_OUTPUT_VALID",
    );
    return {
      state,
      turnRef: execution.turnRef,
      bindingRef,
      binding,
      resultRef,
      result: parsed.parsed as unknown as WorkResult | ReviewResult,
      control: core.control,
    };
  }

  private async executeRawTurn(input: {
    state: StateFile;
    role: LoopRole;
    skill: string;
    taskBindingRef: string | null;
    decisionContextRef: string | null;
    prompt: string;
    outputSchema: Record<string, unknown> | null;
  }): Promise<RawExecution> {
    let state = input.state;
    if (state.activeTurnRef) {
      const turn = this.store.readTurn(state.activeTurnRef);
      if (turn.role !== input.role) {
        throw new Error(
          `active Turn role ${turn.role} does not match ${input.role}`,
        );
      }
      if (
        turn.outputCapture !== "COMPLETE" ||
        !turn.rawOutputRef ||
        !this.store.exists(turn.rawOutputRef)
      ) {
        throw new Error("active Turn has no captured output after reconciliation");
      }
      const summary = this.latestProviderSummary(turn.runtimeRef);
      const raw: RawTurnResult = {
        status: "completed",
        text: this.store.readText(turn.rawOutputRef),
        providerThreadId: turn.providerThreadId,
        providerTurnId: turn.providerTurnId,
        usage: summary?.usage ?? zeroUsage(),
        toolEvents: summary?.toolEvents ?? [],
        rawEvents: [],
        compacted: summary?.compacted ?? false,
        outputCapture: "COMPLETE",
        partialText: turn.partialOutputRef && this.store.exists(turn.partialOutputRef)
          ? this.store.readText(turn.partialOutputRef)
          : "",
        failureKind: turn.runtimeFailureKind,
        interruptError: summary?.interruptError ?? null,
        lastActivityAt: summary?.lastActivityAt ?? turn.completedAt ?? turn.startedAt,
        incrementalEventsPersisted: true,
        error: null,
        elapsedMs: summary?.elapsedMs ?? 0,
      };
      this.store.appendEvent("CAPTURED_TURN_REPLAYED", [
        state.activeTurnRef,
        turn.rawOutputRef,
      ]);
      return {
        state,
        turnRef: state.activeTurnRef,
        turn,
        raw,
      };
    }

    const turnId = this.store.newId("turn");
    const turnRef = `turns/${turnId}/turn.json`;
    const promptRef = `turns/${turnId}/prompt.txt`;
    const validationAuditRef = `turns/${turnId}/validation_audit.json`;
    const run = this.store.readRun();
    const recovery = state.runtimeRecovery;
    if (
      recovery &&
      (
        recovery.role !== input.role ||
        recovery.taskBindingRef !== input.taskBindingRef ||
        recovery.decisionContextRef !== input.decisionContextRef
      )
    ) {
      throw new Error("runtime recovery authorization does not match sequence head");
    }
    const timeoutProfile = this.effectiveTimeoutProfile(
      run.budgets.timeoutProfiles[input.role],
      recovery?.timeoutOverride ?? {},
    );
    this.store.writeText(promptRef, input.prompt);
    this.writeValidationAudit(validationAuditRef, [], []);
    const retryOf = state.correction?.role === input.role
      ? state.correction.retryOfTurnRef
      : this.latestRuntimeFailure(
        input.role,
        input.taskBindingRef,
        input.decisionContextRef,
      );
    const turn: TurnFile = {
      turnId,
      role: input.role,
      round: state.round,
      attempt: this.nextAttempt(
        input.role,
        input.taskBindingRef,
        input.decisionContextRef,
      ),
      taskBindingRef: input.taskBindingRef,
      decisionContextRef: input.decisionContextRef,
      retryOf,
      skill: input.skill,
      turnState: "RUNNING",
      startedAt: new Date().toISOString(),
      completedAt: null,
      promptRef,
      outputCapture: "NONE",
      partialOutputRef: null,
      rawOutputRef: null,
      resultRef: null,
      controlRef: null,
      validationAuditRef,
      errorReportRef: null,
      runtimeErrorRef: null,
      runtimeRef: `turns/${turnId}/runtime.jsonl`,
      providerThreadId: null,
      providerTurnId: null,
      providerStatus: null,
      runtimeFailureKind: null,
      timeoutProfile,
      recoveryRef: recovery?.recoveryRef ?? null,
    };
    this.store.writeTurn(turn);
    this.store.runtimeEvent(turnId, {
      type: "controller_turn_started",
      at: new Date().toISOString(),
      timeoutProfile,
      recoveryRef: turn.recoveryRef,
    });
    this.store.appendTurnToRound(state.round, turnRef);
    state = this.saveState(
      {
        ...state,
        activeTurnRef: turnRef,
        activeTaskBindingRef: input.taskBindingRef,
        runtimeRecovery: null,
      },
      "TURN_STARTED",
    );

    let raw: RawTurnResult;
    try {
      raw = await this.runtime.run({
        turnId,
        role: input.role,
        prompt: input.prompt,
        outputSchema: input.outputSchema,
        cwd: run.projectRoot,
        model: run.model,
        effort: input.role === "DECISION" ? "max" : "high",
        timeoutProfile,
        maxInputTokens: run.budgets.maxInputTokens,
        maxOutputTokens: run.budgets.maxOutputTokens,
        onRuntimeEvent: (event) =>
          this.persistRuntimeEvent(turnRef, turnId, event),
      });
    } catch (error) {
      raw = failedRuntimeResult(error);
    }
    this.persistRuntime(turnId, raw);
    const partialOutputRef = `turns/${turnId}/partial_output.txt`;
    if (
      raw.outputCapture !== "COMPLETE" &&
      raw.partialText &&
      !this.store.exists(partialOutputRef)
    ) {
      this.store.writeText(partialOutputRef, raw.partialText);
    }
    const rawOutputRef = `turns/${turnId}/output.txt`;
    const hasCompleteOutput =
      raw.outputCapture === "COMPLETE" && Boolean(raw.text.trim());
    if (hasCompleteOutput) this.store.writeText(rawOutputRef, raw.text);
    const runtimeErrorRef = `turns/${turnId}/runtime_error.json`;
    if (raw.status !== "completed" || raw.failureKind) {
      this.store.writeJson(runtimeErrorRef, {
        failureKind: raw.failureKind ?? "PROVIDER_ERROR",
        error: raw.error,
        interruptError: raw.interruptError,
        timeoutProfile,
        lastActivityAt: raw.lastActivityAt,
        outputCapture: raw.outputCapture,
        partialOutputRef:
          this.store.exists(partialOutputRef) ? partialOutputRef : null,
        rawOutputRef: hasCompleteOutput ? rawOutputRef : null,
      });
    }
    const updatedTurn = this.store.mutateTurn(turnRef, (value) => ({
      ...value,
      outputCapture: raw.outputCapture,
      partialOutputRef:
        this.store.exists(partialOutputRef) ? partialOutputRef : null,
      rawOutputRef: hasCompleteOutput ? rawOutputRef : null,
      runtimeErrorRef:
        this.store.exists(runtimeErrorRef) ? runtimeErrorRef : null,
      providerThreadId: raw.providerThreadId,
      providerTurnId: raw.providerTurnId,
      providerStatus: raw.status,
      runtimeFailureKind: raw.failureKind,
    }));

    if (raw.status !== "completed" && hasCompleteOutput) {
      this.store.appendEvent("CAPTURED_COMPLETE_MESSAGE_REPLAYED", [
        turnRef,
        rawOutputRef,
      ]);
      raw = { ...raw, status: "completed" };
    }

    if (raw.status !== "completed" || !hasCompleteOutput) {
      this.store.mutateTurn(turnRef, (value) => ({
        ...value,
        turnState: "RUNTIME_FAILED",
        completedAt: new Date().toISOString(),
      }));
      state = this.saveState(
        {
          ...state,
          activeTurnRef: null,
          correction: null,
        },
        "TURN_RUNTIME_FAILED",
      );
      const nonRetriableReason = nonRetriableRuntimeReason(raw.error);
      if (nonRetriableReason) {
        state = this.failState(
          state,
          `${input.role} non-retriable runtime failure: ${nonRetriableReason}`,
          "NON_RECOVERABLE",
        );
      } else if (
        this.countTurnState(
          input.role,
          input.taskBindingRef,
          input.decisionContextRef,
          "RUNTIME_FAILED",
        ) > this.store.readRun().budgets.maxRuntimeRetries
      ) {
        appendTerminalRuntimeTrajectory(this.store, state, turnRef);
        state = this.failState(
          state,
          `${input.role} exceeded runtime retry budget`,
          "RUNTIME_RETRY_EXHAUSTED",
        );
      }
    }
    return { state, turnRef, turn: updatedTurn, raw };
  }

  private rejectAgentOutput(
    state: StateFile,
    turnRef: string,
    turn: TurnFile,
    errors: OutputError[],
    correctRefName: string,
  ): StateFile {
    const checks: ValidationAudit["checks"] = errors.map((error) => ({
      ...error,
      passed: false,
    }));
    this.writeValidationAudit(turn.validationAuditRef, checks, []);
    const errorReport: OutputErrorReport = { errors };
    const errorReportRef =
      `turns/${turn.turnId}/output_error_report.json`;
    this.store.writeJson(errorReportRef, errorReport);
    const updated = this.store.mutateTurn(turnRef, (value) => ({
      ...value,
      turnState: "INVALID_OUTPUT",
      completedAt: new Date().toISOString(),
      errorReportRef,
    }));
    state = this.saveState(
      {
        ...state,
        activeTurnRef: null,
        correction: {
          role: turn.role,
          retryOfTurnRef: turnRef,
          previousOutputRef:
            updated.rawOutputRef ?? `turns/${turn.turnId}/output.txt`,
          errorReportRef,
          correctRefName,
        },
      },
      "OUTPUT_REJECTED",
    );
    if (
      this.countTurnState(
        turn.role,
        turn.taskBindingRef,
        turn.decisionContextRef,
        "INVALID_OUTPUT",
      ) > this.store.readRun().budgets.maxOutputRetries
    ) {
      return this.failState(
        state,
        `${turn.role} exceeded output-correction retry budget`,
      );
    }
    return state;
  }

  private applyDecision(
    state: StateFile,
    decisionTurnRef: string,
    result: DecisionProtocolResult,
  ): StateFile {
    const cycleState = state;
    const decision = result.decision;
    this.setTurnState(decisionTurnRef, "COMMITTED");
    if (decision === "RETRY_WORKER") {
      if (!state.pending) throw new Error("RETRY_WORKER lacks pending pair");
      if (
        state.semanticRetries.worker >=
          this.store.readRun().budgets.maxSemanticRetries
      ) {
        const failed = this.failState(
          {
            ...state,
            activeTurnRef: null,
            latestDecisionTurnRef: decisionTurnRef,
            decisionGuidance: result.guidance,
          },
          "WORKER semantic retry budget exhausted; pending results were not committed",
        );
        return this.finishDecisionCycle(
          cycleState,
          failed,
          decisionTurnRef,
          result,
        );
      }
      this.setTurnState(
        state.pending.workTurnRef,
        "SUPERSEDED_BY_RETRY",
      );
      if (state.pending.reviewTurnRef) {
        this.setTurnState(
          state.pending.reviewTurnRef,
          "SUPERSEDED_BY_RETRY",
        );
      }
      const next = this.saveState(
        {
          ...state,
          transitions: state.transitions + 1,
          sequence: sequenceAfterDecision(
            decision,
            state.pending.workTaskBindingRef,
          ),
          node: "WORKER",
          activeTaskBindingRef: null,
          activeTurnRef: null,
          pending: null,
          preReview: null,
          decisionGuidance: result.guidance,
          latestDecisionTurnRef: decisionTurnRef,
          semanticRetries: {
            ...state.semanticRetries,
            worker: state.semanticRetries.worker + 1,
          },
        },
        "SEMANTIC_RETRY_WORKER_SCHEDULED",
      );
      return this.finishDecisionCycle(
        cycleState,
        next,
        decisionTurnRef,
        result,
      );
    }
    if (decision === "RETRY_REVIEWER") {
      if (!state.pending?.reviewTaskBindingRef || !state.pending.reviewTurnRef) {
        throw new Error("RETRY_REVIEWER lacks pending Reviewer binding");
      }
      if (
        state.semanticRetries.reviewer >=
          this.store.readRun().budgets.maxSemanticRetries
      ) {
        const failed = this.failState(
          {
            ...state,
            activeTurnRef: null,
            latestDecisionTurnRef: decisionTurnRef,
            decisionGuidance: result.guidance,
          },
          "REVIEWER semantic retry budget exhausted; pending results were not committed",
        );
        return this.finishDecisionCycle(
          cycleState,
          failed,
          decisionTurnRef,
          result,
        );
      }
      this.setTurnState(
        state.pending.reviewTurnRef,
        "SUPERSEDED_BY_RETRY",
      );
      const next = this.saveState(
        {
          ...state,
          transitions: state.transitions + 1,
          sequence: sequenceAfterDecision(
            decision,
            state.pending.reviewTaskBindingRef,
          ),
          node: "REVIEWER",
          activeTaskBindingRef: null,
          activeTurnRef: null,
          pending: {
            ...state.pending,
            reviewTaskBindingRef: null,
            reviewTurnRef: null,
            reviewRef: null,
            reviewVerdict: null,
          },
          decisionGuidance: result.guidance,
          latestDecisionTurnRef: decisionTurnRef,
          semanticRetries: {
            ...state.semanticRetries,
            reviewer: state.semanticRetries.reviewer + 1,
          },
        },
        "SEMANTIC_RETRY_REVIEWER_SCHEDULED",
      );
      return this.finishDecisionCycle(
        cycleState,
        next,
        decisionTurnRef,
        result,
      );
    }

    if (!state.pending?.reviewTurnRef || !state.pending.reviewRef) {
      throw new Error(`${decision} requires a complete pending pair`);
    }
    commitPending(this.store, state, decisionTurnRef);
    this.setTurnState(state.pending.workTurnRef, "COMMITTED");
    this.setTurnState(state.pending.reviewTurnRef, "COMMITTED");

    if (decision === "FINISH_WORKFLOW") {
      if (computeRemainingRequirements(this.store, {
        ...state,
        pending: null,
      }, false).length > 0) {
        throw new Error(
          "FINISH_WORKFLOW cannot close non-empty mechanical requirements",
        );
      }
      renderFinalReport(this.store);
      const finished = this.saveState(
        {
          ...state,
          lifecycle: "FINISHED",
          reason: null,
          pauseKind: null,
          failureKind: null,
          node: null,
          transitions: state.transitions + 1,
          sequence: [],
          activeTaskBindingRef: null,
          activeTurnRef: null,
          pending: null,
          preReview: null,
          decisionGuidance: result.guidance,
          latestDecisionTurnRef: decisionTurnRef,
          runtimeRecovery: null,
          semanticRetries: { worker: 0, reviewer: 0 },
        },
        "WORKFLOW_FINISHED",
      );
      return this.finishDecisionCycle(
        cycleState,
        finished,
        decisionTurnRef,
        result,
      );
    }

    const nextRound = state.round + 1;
    const nextSequence = sequenceAfterDecision(decision);
    this.store.writeRound({
      round: nextRound,
      branch: decision,
      turnRefs: [],
      committedAt: null,
    });
    const nextState: StateFile = {
      ...state,
      round: nextRound,
      transitions: state.transitions + 1,
      sequence: nextSequence,
      node: nextSequence[0]?.role ?? null,
      activeTaskBindingRef: null,
      activeTurnRef: null,
      pending: null,
      preReview: null,
      decisionGuidance: result.guidance,
      latestDecisionTurnRef: decisionTurnRef,
      semanticRetries: { worker: 0, reviewer: 0 },
    };
    const authorizedThroughRound = this.authorizedThroughRound(state);
    if (state.round >= authorizedThroughRound) {
      const paused = this.pauseState(
        nextState,
        `authorized round budget through round ${authorizedThroughRound} exhausted; round ${nextRound} is prepared and requires explicit resume`,
        "ROUND_BUDGET_EXHAUSTED",
      );
      return this.finishDecisionCycle(
        cycleState,
        paused,
        decisionTurnRef,
        result,
      );
    }
    const next = this.saveState(
      nextState,
      decision === "RUN_REVIEWER"
        ? "REVIEWER_BRANCH_SCHEDULED"
        : "WORKER_BRANCH_SCHEDULED",
    );
    return this.finishDecisionCycle(
      cycleState,
      next,
      decisionTurnRef,
      result,
    );
  }

  private reconcileInterruptedTurn(state: StateFile): StateFile {
    if (!state.activeTurnRef) return state;
    let turn = this.store.readTurn(state.activeTurnRef);
    if (turn.turnState === "RUNNING") {
      turn = this.rebuildTurnCaptureFromRuntime(state.activeTurnRef, turn);
    }
    if (
      turn.turnState === "RUNNING" &&
      turn.outputCapture === "COMPLETE" &&
      !turn.rawOutputRef
    ) {
      const recovered = this.recoverCompletedMessageFromRuntime(turn);
      if (recovered) {
        const capturedRef = `turns/${turn.turnId}/output.txt`;
        this.store.writeText(capturedRef, recovered);
        turn = this.store.mutateTurn(state.activeTurnRef, (value) => ({
          ...value,
          rawOutputRef: capturedRef,
        }));
        this.store.appendEvent("CAPTURED_MESSAGE_RECOVERED_FROM_RUNTIME", [
          state.activeTurnRef,
          turn.runtimeRef,
          capturedRef,
        ]);
      }
    }
    if (
      ["RUNNING", "PENDING_DECISION", "COMMITTED"].includes(turn.turnState) &&
      turn.outputCapture === "COMPLETE" &&
      Boolean(turn.rawOutputRef && this.store.exists(turn.rawOutputRef))
    ) {
      return state;
    }
    if (turn.turnState !== "RUNNING") {
      return this.saveState(
        { ...state, activeTurnRef: null },
        "STALE_ACTIVE_TURN_CLEARED",
      );
    }
    const runtimeErrorRef = `turns/${turn.turnId}/runtime_error.json`;
    this.store.writeJson(runtimeErrorRef, {
      failureKind: turn.runtimeFailureKind ?? "PROVIDER_ERROR",
      error: "Controller restarted before Provider Turn reached a usable terminal message",
      interruptError: null,
      timeoutProfile: turn.timeoutProfile,
      lastActivityAt: turn.completedAt ?? turn.startedAt,
      outputCapture: turn.outputCapture,
      partialOutputRef: turn.partialOutputRef,
      rawOutputRef: turn.rawOutputRef,
    });
    this.store.mutateTurn(state.activeTurnRef, (value) => ({
      ...value,
      turnState: "RUNTIME_FAILED",
      completedAt: new Date().toISOString(),
      providerStatus: value.providerStatus ?? "interrupted",
      runtimeFailureKind: value.runtimeFailureKind ?? "PROVIDER_ERROR",
      runtimeErrorRef,
    }));
    const next = this.saveState(
      {
        ...state,
        activeTurnRef: null,
        correction: null,
      },
      "TURN_INTERRUPTED_WITHOUT_OUTPUT",
    );
    if (
      this.countTurnState(
        turn.role,
        turn.taskBindingRef,
        turn.decisionContextRef,
        "RUNTIME_FAILED",
      ) > this.store.readRun().budgets.maxRuntimeRetries
    ) {
      appendTerminalRuntimeTrajectory(this.store, next, state.activeTurnRef);
      return this.failState(
        next,
        `${turn.role} exceeded runtime retry budget`,
        "RUNTIME_RETRY_EXHAUSTED",
      );
    }
    return next;
  }

  private bindHeadStep(state: StateFile, bindingRef: string): StateFile {
    const [head, ...tail] = state.sequence;
    if (!head) throw new Error("cannot bind an empty sequence");
    return this.saveState(
      {
        ...state,
        sequence: [{ ...head, bindingRef }, ...tail],
        activeTaskBindingRef: bindingRef,
      },
      "SEQUENCE_STEP_BOUND",
    );
  }

  private promptCorrection(
    correction: StateFile["correction"],
  ): OutputCorrectionPrompt | null {
    if (!correction) return null;
    return {
      previousOutputPath: this.store.absolute(correction.previousOutputRef),
      errorReportPath: this.store.absolute(correction.errorReportRef),
      correctRefName: correction.correctRefName,
    };
  }

  private promptRuntimeRetry(
    role: LoopRole,
    bindingRef: string | null,
    contextRef: string | null,
  ): RuntimeRetryPrompt | null {
    const latest = this.matchingTurns(role, bindingRef, contextRef).at(-1);
    if (!latest || latest.turn.turnState !== "RUNTIME_FAILED") return null;
    return {
      previousTurnPath: this.store.absolute(latest.ref),
      failure:
        latest.turn.runtimeFailureKind ??
        this.runtimeErrorMessage(latest.turn) ??
        "PROVIDER_ERROR",
      partialOutputPath:
        latest.turn.partialOutputRef &&
          this.store.exists(latest.turn.partialOutputRef)
          ? this.store.absolute(latest.turn.partialOutputRef)
          : null,
    };
  }

  private persistRuntimeEvent(
    turnRef: string,
    turnId: string,
    event: RuntimePersistenceEvent,
  ): void {
    this.store.runtimeEvent(turnId, event);
    if (event.type === "provider_started") {
      this.store.mutateTurn(turnRef, (turn) => ({
        ...turn,
        providerThreadId: event.threadId,
        providerTurnId: event.providerTurnId,
      }));
      return;
    }
    if (event.type === "output_delta") {
      const partialOutputRef = `turns/${turnId}/partial_output.txt`;
      this.store.appendText(partialOutputRef, event.delta);
      this.store.mutateTurn(turnRef, (turn) => ({
        ...turn,
        outputCapture:
          turn.outputCapture === "NONE" ? "PARTIAL" : turn.outputCapture,
        partialOutputRef,
      }));
      return;
    }
    if (
      event.type === "message_completed" &&
      event.phase !== "commentary"
    ) {
      this.store.mutateTurn(turnRef, (turn) => ({
        ...turn,
        outputCapture: "COMPLETE",
      }));
      return;
    }
    if (event.type === "timeout") {
      this.store.mutateTurn(turnRef, (turn) => ({
        ...turn,
        outputCapture: event.capture,
        runtimeFailureKind: event.kind,
      }));
    }
  }

  private effectiveTimeoutProfile(
    base: TurnTimeoutProfile,
    override: Partial<TurnTimeoutProfile>,
  ): TurnTimeoutProfile {
    const profile = { ...base, ...override };
    for (const [key, value] of Object.entries(profile)) {
      if (!Number.isInteger(value) || value < 1) {
        throw new Error(`${key} must be a positive integer`);
      }
    }
    if (profile.hardTimeoutMs < profile.idleTimeoutMs) {
      throw new Error("hardTimeoutMs must be >= idleTimeoutMs");
    }
    return profile;
  }

  private recoverCompletedMessageFromRuntime(turn: TurnFile): string | null {
    if (!this.store.exists(turn.runtimeRef)) return null;
    const messages = this.store
      .readJsonLines<RuntimePersistenceEvent>(turn.runtimeRef)
      .filter((event) => event.type === "message_completed")
      .map((event) => ({ text: event.text, phase: event.phase }));
    const selected = selectProtocolAgentMessage(null, messages);
    return selected.error === null && selected.text ? selected.text : null;
  }

  private rebuildTurnCaptureFromRuntime(
    turnRef: string,
    turn: TurnFile,
  ): TurnFile {
    if (!this.store.exists(turn.runtimeRef)) return turn;
    const events = this.store.readJsonLines<RuntimePersistenceEvent>(
      turn.runtimeRef,
    );
    const deltas = events
      .filter((event) => event.type === "output_delta")
      .map((event) => event.delta);
    let partialOutputRef = turn.partialOutputRef;
    if (deltas.length > 0 && (!partialOutputRef || !this.store.exists(partialOutputRef))) {
      partialOutputRef = `turns/${turn.turnId}/partial_output.txt`;
      this.store.writeText(partialOutputRef, deltas.join(""));
    }
    const complete = events.some(
      (event) =>
        event.type === "message_completed" && event.phase !== "commentary",
    );
    const timeout = events
      .filter((event) => event.type === "timeout")
      .at(-1);
    const started = events
      .filter((event) => event.type === "provider_started")
      .at(-1);
    const capture = complete
      ? "COMPLETE"
      : deltas.length > 0
      ? "PARTIAL"
      : turn.outputCapture ?? "NONE";
    if (
      capture === turn.outputCapture &&
      partialOutputRef === turn.partialOutputRef &&
      !timeout &&
      !started
    ) return turn;
    return this.store.mutateTurn(turnRef, (value) => ({
      ...value,
      outputCapture: capture,
      partialOutputRef,
      providerThreadId:
        started?.type === "provider_started"
          ? started.threadId
          : value.providerThreadId,
      providerTurnId:
        started?.type === "provider_started"
          ? started.providerTurnId
          : value.providerTurnId,
      runtimeFailureKind:
        timeout?.type === "timeout" ? timeout.kind : value.runtimeFailureKind,
    }));
  }

  private runtimeErrorMessage(turn: TurnFile): string | null {
    if (!turn.runtimeErrorRef || !this.store.exists(turn.runtimeErrorRef)) {
      return null;
    }
    const value = this.store.readJson<{ error?: unknown }>(turn.runtimeErrorRef);
    return typeof value.error === "string" ? value.error : null;
  }

  private latestProviderSummary(turnRuntimeRef: string): {
    usage: TokenUsage;
    toolEvents: RuntimeToolEvent[];
    compacted: boolean;
    interruptError: string | null;
    lastActivityAt: string;
    elapsedMs: number;
  } | null {
    if (!this.store.exists(turnRuntimeRef)) return null;
    const summaries = this.store
      .readJsonLines<Record<string, unknown>>(turnRuntimeRef)
      .filter((event) => event.type === "provider_summary");
    const value = summaries.at(-1);
    if (!value) return null;
    return {
      usage: isTokenUsage(value.usage) ? value.usage : zeroUsage(),
      toolEvents: Array.isArray(value.toolEvents)
        ? value.toolEvents as RuntimeToolEvent[]
        : [],
      compacted: value.compacted === true,
      interruptError:
        typeof value.interruptError === "string" ? value.interruptError : null,
      lastActivityAt:
        typeof value.lastActivityAt === "string"
          ? value.lastActivityAt
          : new Date(0).toISOString(),
      elapsedMs: Number(value.elapsedMs ?? 0),
    };
  }

  private finishDecisionCycle(
    cycleState: StateFile,
    resultingState: StateFile,
    decisionTurnRef: string,
    result: DecisionProtocolResult,
  ): StateFile {
    appendDecisionTrajectory(
      this.store,
      cycleState,
      resultingState,
      decisionTurnRef,
      result,
    );
    rebuildResearchMemory(this.store, resultingState);
    if (
      resultingState.lifecycle === "PAUSED" ||
      resultingState.lifecycle === "FINISHED"
    ) {
      writeCheckpoint(
        this.store,
        resultingState,
        resultingState.reason ?? "workflow finished",
      );
    }
    return resultingState;
  }

  private setTurnState(
    turnRef: string,
    turnState: TurnFile["turnState"],
  ): void {
    this.store.mutateTurn(turnRef, (turn) => ({
      ...turn,
      turnState,
      completedAt: turn.completedAt ?? new Date().toISOString(),
    }));
  }

  private writeValidationAudit(
    ref: string,
    checks: ValidationAudit["checks"],
    advisories: ValidationAudit["advisories"] = [],
  ): void {
    this.store.writeJson(
      ref,
      { checks, advisories } satisfies ValidationAudit,
    );
  }

  private persistRuntime(turnId: string, raw: RawTurnResult): void {
    this.store.runtimeEvent(turnId, {
      type: "provider_summary",
      status: raw.status,
      providerThreadId: raw.providerThreadId,
      providerTurnId: raw.providerTurnId,
      usage: raw.usage,
      toolEvents: raw.toolEvents,
      compacted: raw.compacted,
      outputCapture: raw.outputCapture,
      failureKind: raw.failureKind,
      interruptError: raw.interruptError,
      lastActivityAt: raw.lastActivityAt,
      error: raw.error,
      elapsedMs: raw.elapsedMs,
    });
    if (!raw.incrementalEventsPersisted) {
      for (const event of raw.rawEvents) this.store.runtimeEvent(turnId, event);
      for (const event of raw.toolEvents) {
        this.store.runtimeEvent(turnId, { type: "tool", event });
      }
    }
  }

  private nextAttempt(
    role: LoopRole,
    bindingRef: string | null,
    contextRef: string | null,
  ): number {
    return this.matchingTurns(role, bindingRef, contextRef).length + 1;
  }

  private latestRuntimeFailure(
    role: LoopRole,
    bindingRef: string | null,
    contextRef: string | null,
  ): string | null {
    return this.matchingTurns(role, bindingRef, contextRef)
      .reverse()
      .find((item) => item.turn.turnState === "RUNTIME_FAILED")?.ref ?? null;
  }

  private latestDecisionRuntimeFailure(
    round: number,
  ): { ref: string; turn: TurnFile } | null {
    const latest = this.store.turnRefs()
      .map((ref) => ({ ref, turn: this.store.readTurn(ref) }))
      .filter(({ turn }) => turn.role === "DECISION" && turn.round === round)
      .sort((left, right) =>
        left.turn.startedAt.localeCompare(right.turn.startedAt)
      )
      .at(-1);
    return latest?.turn.turnState === "RUNTIME_FAILED" ? latest : null;
  }

  private countTurnState(
    role: LoopRole,
    bindingRef: string | null,
    contextRef: string | null,
    turnState: TurnFile["turnState"],
  ): number {
    return this.matchingTurns(role, bindingRef, contextRef)
      .filter((item) => item.turn.turnState === turnState).length;
  }

  private matchingTurns(
    role: LoopRole,
    bindingRef: string | null,
    contextRef: string | null,
  ): Array<{ ref: string; turn: TurnFile }> {
    return this.store.turnRefs()
      .map((ref) => ({ ref, turn: this.store.readTurn(ref) }))
      .filter((item) =>
        item.turn.role === role &&
        item.turn.taskBindingRef === bindingRef &&
        item.turn.decisionContextRef === contextRef
      )
      .sort((left, right) =>
        left.turn.startedAt.localeCompare(right.turn.startedAt)
      );
  }

  private saveState(
    value: Omit<StateFile, "revision"> & { revision: number },
    event: string,
  ): StateFile {
    const current = this.store.readState();
    const next: StateFile = {
      ...value,
      revision: current.revision + 1,
    };
    this.store.writeState(next, event);
    return next;
  }

  private resumeState(
    state: StateFile,
    additionalRounds?: number,
  ): StateFile {
    const shouldGrant =
      state.pauseKind === "ROUND_BUDGET_EXHAUSTED" ||
      additionalRounds !== undefined;
    let roundBudget = state.roundBudget;
    if (!roundBudget) {
      throw new Error("formatVersion 7 state lacks roundBudget");
    }
    if (shouldGrant) {
      const grant = additionalRounds ?? this.store.readRun().budgets.maxRounds;
      if (!Number.isInteger(grant) || grant < 1) {
        throw new Error("additionalRounds must be a positive integer");
      }
      const baseRound = Math.max(
        roundBudget.authorizedThroughRound,
        state.round - 1,
      );
      const authorizationId = this.store.newId("round-authorization");
      const authorizationRef =
        `authorizations/rounds/${authorizationId}.json`;
      const record: RoundAuthorizationRecord = {
        formatVersion: CURRENT_FORMAT_VERSION,
        authorizationId,
        createdAt: new Date().toISOString(),
        sourceStateRevision: state.revision,
        additionalRounds: grant,
        firstAuthorizedRound: baseRound + 1,
        authorizedThroughRound: baseRound + grant,
      };
      this.store.writeImmutableJson(authorizationRef, record);
      this.store.appendEvent("ROUND_AUTHORIZATION_GRANTED", [
        authorizationRef,
      ]);
      roundBudget = {
        authorizedThroughRound: record.authorizedThroughRound,
        lastAuthorizationRef: authorizationRef,
      };
    }
    return this.saveState(
      {
        ...state,
        lifecycle: "RUNNING",
        reason: null,
        pauseKind: null,
        roundBudget,
        failureKind: null,
      },
      "RUN_RESUMED",
    );
  }

  private authorizedThroughRound(state: StateFile): number {
    const value = state.roundBudget?.authorizedThroughRound;
    if (!Number.isInteger(value) || Number(value) < 1) {
      throw new Error("formatVersion 7 state has invalid roundBudget");
    }
    return Number(value);
  }

  private pauseState(
    state: StateFile,
    reason: string,
    pauseKind: NonNullable<StateFile["pauseKind"]>,
  ): StateFile {
    const next = this.saveState(
      {
        ...state,
        lifecycle: "PAUSED",
        reason,
        pauseKind,
        failureKind: null,
        runtimeRecovery: null,
        node: null,
      },
      "WORKFLOW_PAUSED",
    );
    this.store.writeJson("final/outcome.json", {
      workflowOutcome: "PAUSED",
      reportRef: null,
      reason,
    } satisfies RunOutcome);
    return next;
  }

  private failState(
    state: StateFile,
    reason: string,
    failureKind: WorkflowFailureKind = "NON_RECOVERABLE",
  ): StateFile {
    const next = this.saveState(
      {
        ...state,
        lifecycle: "FAILED",
        reason,
        pauseKind: null,
        failureKind,
        runtimeRecovery: null,
        node: null,
      },
      "WORKFLOW_FAILED",
    );
    this.store.writeJson("final/outcome.json", {
      workflowOutcome: "FAILED",
      reportRef: null,
      reason,
    } satisfies RunOutcome);
    return next;
  }

  private outcomeFromState(state: StateFile): RunOutcome {
    if (state.lifecycle === "FINISHED") {
      return {
        workflowOutcome: "FINISHED",
        reportRef: "final/report.md",
        reason: null,
      };
    }
    if (state.lifecycle === "PAUSED") {
      return {
        workflowOutcome: "PAUSED",
        reportRef: null,
        reason: state.reason,
      };
    }
    return {
      workflowOutcome: "FAILED",
      reportRef: null,
      reason: state.reason ?? "workflow stopped without FINISH_WORKFLOW",
    };
  }
}

function failedRuntimeResult(error: unknown): RawTurnResult {
  const at = new Date().toISOString();
  return {
    status: "failed",
    text: "",
    providerThreadId: null,
    providerTurnId: null,
    usage: zeroUsage(),
    toolEvents: [],
    rawEvents: [],
    compacted: false,
    outputCapture: "NONE",
    partialText: "",
    failureKind: "PROVIDER_ERROR",
    interruptError: null,
    lastActivityAt: at,
    incrementalEventsPersisted: false,
    error: error instanceof Error ? error.message : String(error),
    elapsedMs: 0,
  };
}

function nonRetriableRuntimeReason(error: string | null): string | null {
  if (!error) return null;
  const decoded = decodeProviderError(error);
  const code = typeof decoded.code === "string" ? decoded.code : null;
  const type = typeof decoded.type === "string" ? decoded.type : null;
  if (
    code !== "invalid_json_schema" &&
    type !== "invalid_request_error" &&
    !error.includes("invalid_json_schema") &&
    !error.includes("invalid_request_error")
  ) {
    return null;
  }
  const message =
    typeof decoded.message === "string" && decoded.message.trim()
      ? decoded.message.trim()
      : error.trim();
  const label = code ?? type ?? "invalid_request_error";
  return `${label}: ${message}`;
}

function decodeProviderError(
  source: string,
): Record<string, unknown> {
  try {
    const parsed = JSON.parse(source);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const root = parsed as Record<string, unknown>;
    const nested = root.error;
    return nested && typeof nested === "object" && !Array.isArray(nested)
      ? nested as Record<string, unknown>
      : root;
  } catch {
    return {};
  }
}

function isTokenUsage(value: unknown): value is TokenUsage {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return [
    "inputTokens",
    "cachedInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
  ].every((key) => typeof record[key] === "number");
}

function zeroUsage(): TokenUsage {
  return {
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningOutputTokens: 0,
    totalTokens: 0,
  };
}
