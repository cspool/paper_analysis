import type {
  AnyTurnResult,
  AnyTurnTask,
  ClosureReviewTaskEnvelope,
  DirectionReviewTaskEnvelope,
  EvidenceReaderTaskEnvelope,
  RegisteredRole,
  StageContract,
  ValidationReport,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  ROLE_REASONING_EFFORT,
} from "../contracts/index.ts";
import { normalizeAgentOutput } from "../failure_handling/output_normalizer.ts";
import {
  validateEvidenceRuntimeTrace,
  validateRuntimeToolEvents,
} from "../security/no_experiment_guard.ts";
import {
  validateClosureReview,
  validateEvidencePacket,
  validateReviewDelta,
  validateTaskForDispatch,
  validateWorkflowDecisionProposal,
  mergeReports,
  type TaskValidationContext,
} from "../validators/index.ts";
import {
  buildClosureReviewerPrompt,
  buildDirectionReviewerPrompt,
  buildEvidenceReaderPrompt,
  buildWorkflowTurnPrompt,
} from "./prompt_builder.ts";
import type {
  FreshTurnRuntime,
  FrozenTurnDispatch,
  RawTurnResult,
} from "./runtime.ts";
import { loadSkillPackage } from "./skill_package.ts";

export interface DispatchAttemptInput {
  role: RegisteredRole;
  task: AnyTurnTask;
  stageContract: StageContract;
  taskValidationContext: TaskValidationContext;
  skillRoot: string;
  expectedSchema: unknown;
  runtime: FreshTurnRuntime;
  model: string;
  providerWireEffort: string;
  cwd: string;
}

export interface DispatchAttemptResult {
  dispatched: boolean;
  preDispatchReport: ValidationReport;
  rawTurn: RawTurnResult | null;
  normalized: ReturnType<typeof normalizeAgentOutput> | null;
  result: AnyTurnResult | null;
  resultValidationReport: ValidationReport | null;
  securityReport: ValidationReport | null;
  skillPackageHash: string;
  promptText: string | null;
  promptSha256: string | null;
}

export interface CapturedTurnValidation {
  normalized: ReturnType<typeof normalizeAgentOutput> | null;
  result: AnyTurnResult | null;
  resultValidationReport: ValidationReport | null;
  securityReport: ValidationReport;
}

export async function dispatchFreshTurnAttempt(
  input: DispatchAttemptInput,
): Promise<DispatchAttemptResult> {
  const skill = loadSkillPackage(input.skillRoot);
  if (skill.sha256 !== input.taskValidationContext.skillSha256) {
    throw new Error("loaded Skill package hash differs from frozen task binding");
  }
  const preDispatchReport = validateTaskForDispatch(
    input.task,
    input.taskValidationContext,
  );
  if (!preDispatchReport.valid) {
    return {
      dispatched: false,
      preDispatchReport,
      rawTurn: null,
      normalized: null,
      result: null,
      resultValidationReport: null,
      securityReport: null,
      skillPackageHash: skill.sha256,
      promptText: null,
      promptSha256: null,
    };
  }

  const prompt = buildPrompt(input, skill.skillMarkdown);
  const dispatch: FrozenTurnDispatch = {
    attemptId: getAttemptId(prompt.task),
    taskId: prompt.task.taskId,
    role: input.role,
    model: input.model,
    logicalEffort: ROLE_REASONING_EFFORT[input.role],
    providerWireEffort: input.providerWireEffort,
    prompt: prompt.prompt,
    outputSchema: input.expectedSchema,
    cwd: input.cwd,
    timeoutMs: input.stageContract.budget.timeoutMs,
  };
  let rawTurn: RawTurnResult;
  try {
    rawTurn = await input.runtime.run(dispatch);
  } catch (error) {
    rawTurn = failedRuntimeResult(
      dispatch,
      error instanceof Error ? error.message : String(error),
    );
  }
  const captured = validateCapturedTurnOutput(
    input.role,
    prompt.task,
    rawTurn,
  );
  if (
    !captured.securityReport.valid ||
    rawTurn.status !== "completed"
  ) {
    return {
      dispatched: true,
      preDispatchReport,
      rawTurn,
      normalized: null,
      result: null,
      resultValidationReport: null,
      securityReport: captured.securityReport,
      skillPackageHash: skill.sha256,
      promptText: prompt.prompt,
      promptSha256: prompt.promptSha256,
    };
  }
  return {
    dispatched: true,
    preDispatchReport,
    rawTurn,
    normalized: captured.normalized,
    result: captured.result,
    resultValidationReport: captured.resultValidationReport,
    securityReport: captured.securityReport,
    skillPackageHash: skill.sha256,
    promptText: prompt.prompt,
    promptSha256: prompt.promptSha256,
  };
}

/**
 * Re-run the pure post-provider validation pipeline for a durable captured raw
 * Turn. This function never dispatches, resumes, or mutates provider state.
 */
export function validateCapturedTurnOutput(
  role: RegisteredRole,
  task: AnyTurnTask,
  rawTurn: RawTurnResult,
): CapturedTurnValidation {
  const securityReport = validateRuntimeToolEvents(
    role,
    rawTurn.toolEvents,
    role === "evidence_reader"
      ? (task as EvidenceReaderTaskEnvelope)
      : undefined,
  );
  if (rawTurn.compacted) {
    securityReport.valid = false;
    securityReport.errors.push({
      code: "security.context_compaction",
      jsonPointer: "/rawTurn/compacted",
      message:
        "fresh single-Turn execution compacted its authoritative input",
    });
  }
  if (!securityReport.valid || rawTurn.status !== "completed") {
    return {
      normalized: null,
      result: null,
      resultValidationReport: null,
      securityReport,
    };
  }
  const normalized = normalizeAgentOutput(rawTurn.text);
  if (normalized.parsed === null) {
    return {
      normalized,
      result: null,
      resultValidationReport: normalizationReport(normalized),
      securityReport,
    };
  }
  const result = normalized.parsed as AnyTurnResult;
  let resultValidationReport = validateRoleResult(role, result, task);
  if (role === "evidence_reader" && resultValidationReport.valid) {
    resultValidationReport = mergeReports(
      resultValidationReport,
      validateEvidenceRuntimeTrace(
        task as EvidenceReaderTaskEnvelope,
        result as never,
        rawTurn.toolEvents,
      ),
    );
  }
  return {
    normalized,
    result,
    resultValidationReport,
    securityReport,
  };
}

function buildPrompt(
  input: DispatchAttemptInput,
  skillMarkdown: string,
): {
  task: AnyTurnTask;
  prompt: string;
  promptSha256: string;
  inputHash: string;
} {
  const common = {
    skillMarkdown,
    expectedSchema: input.expectedSchema,
  };
  switch (input.role) {
    case "workflow_decision":
      return buildWorkflowTurnPrompt({
        task: input.task as WorkflowTurnTask,
        ...common,
      });
    case "evidence_reader":
      return buildEvidenceReaderPrompt({
        task: input.task as EvidenceReaderTaskEnvelope,
        ...common,
      });
    case "direction_reviewer":
      return buildDirectionReviewerPrompt({
        task: input.task as DirectionReviewTaskEnvelope,
        ...common,
      });
    case "closure_reviewer":
      return buildClosureReviewerPrompt({
        task: input.task as ClosureReviewTaskEnvelope,
        ...common,
      });
  }
}

function validateRoleResult(
  role: RegisteredRole,
  result: AnyTurnResult,
  task: AnyTurnTask,
): ValidationReport {
  switch (role) {
    case "workflow_decision":
      return validateWorkflowDecisionProposal(
        result as never,
        task as WorkflowTurnTask,
      );
    case "evidence_reader":
      return validateEvidencePacket(
        result as never,
        task as EvidenceReaderTaskEnvelope,
      );
    case "direction_reviewer":
      return validateReviewDelta(
        result as never,
        task as DirectionReviewTaskEnvelope,
      );
    case "closure_reviewer":
      return validateClosureReview(
        result as never,
        task as ClosureReviewTaskEnvelope,
      );
  }
}

function normalizationReport(
  normalized: ReturnType<typeof normalizeAgentOutput>,
): ValidationReport {
  return {
    validatorVersion: "simple-semantic-loop-validator/1",
    valid: false,
    errors: [
      {
        code: `normalization.${normalized.errorCode ?? "invalid"}`,
        jsonPointer: null,
        message: normalized.errorMessage ?? "invalid output",
      },
    ],
    checkedArtifactHashes: [],
    checkedObjectRefs: [],
  };
}

function getAttemptId(task: AnyTurnTask): string {
  return task.attemptId;
}

function failedRuntimeResult(
  dispatch: FrozenTurnDispatch,
  message: string,
): RawTurnResult {
  return {
    attemptId: dispatch.attemptId,
    providerThreadId: `unavailable-thread-${dispatch.attemptId}`,
    providerTurnId: `unavailable-turn-${dispatch.attemptId}`,
    status: "failed",
    text: "",
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
    error: message,
    elapsedMs: 0,
  };
}
