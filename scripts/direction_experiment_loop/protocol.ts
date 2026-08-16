import {
  EVIDENCE_SCOPES,
  EXPERIMENT_DECISIONS,
  JUDGE_ASSESSMENTS,
  type ExperimentContractProposal,
  type ExperimentDecisionResult,
  type JudgeResult,
} from "./types.ts";

export interface ParseResult<T> {
  value: T | null;
  errors: string[];
}

export interface DecisionProtocolLimits {
  maxEstimatedMinutes?: number;
}

export function parseDecisionResult(
  text: string,
  limits: DecisionProtocolLimits = {},
): ParseResult<ExperimentDecisionResult> {
  const parsed = parseJsonObject(text);
  if (!parsed.value) return { value: null, errors: [parsed.error!] };
  const value = parsed.value;
  const errors: string[] = [];
  const decision = stringField(value.decision);
  const evidenceScope = stringField(value.evidenceScope);
  const reason = stringField(value.reason).trim();
  const contract = parseContract(value.experimentContract, errors, limits);
  const reviewFocus = nullableString(value.reviewFocus);

  if (!EXPERIMENT_DECISIONS.includes(decision as ExperimentDecisionResult["decision"])) {
    errors.push(`decision must be one of: ${EXPERIMENT_DECISIONS.join(" | ")}`);
  }
  if (!EVIDENCE_SCOPES.includes(evidenceScope as ExperimentDecisionResult["evidenceScope"])) {
    errors.push(`evidenceScope must be one of: ${EVIDENCE_SCOPES.join(" | ")}`);
  }
  if (!reason) errors.push("reason must be a non-empty string");
  if (decision === "RUN_LAB" && !contract) {
    errors.push("RUN_LAB requires one complete experimentContract object");
  }
  if (decision !== "RUN_LAB" && value.experimentContract != null) {
    errors.push("experimentContract must be null unless decision is RUN_LAB");
  }
  if (decision === "RUN_JUDGE" && !reviewFocus) {
    errors.push("RUN_JUDGE requires a non-empty reviewFocus");
  }
  if (decision !== "RUN_JUDGE" && value.reviewFocus != null) {
    errors.push("reviewFocus must be null unless decision is RUN_JUDGE");
  }
  if (errors.length > 0) return { value: null, errors };
  return {
    value: {
      decision: decision as ExperimentDecisionResult["decision"],
      evidenceScope: evidenceScope as ExperimentDecisionResult["evidenceScope"],
      reason,
      experimentContract: decision === "RUN_LAB" ? contract : null,
      reviewFocus: decision === "RUN_JUDGE" ? reviewFocus : null,
    },
    errors: [],
  };
}

export function parseJudgeResult(text: string): ParseResult<JudgeResult> {
  const parsed = parseJsonObject(text);
  if (!parsed.value) return { value: null, errors: [parsed.error!] };
  const value = parsed.value;
  const errors: string[] = [];
  const assessment = stringField(value.assessment);
  const evidenceScope = stringField(value.evidenceScope);
  const reason = stringField(value.reason).trim();
  const remainingUncertainty = stringField(value.remainingUncertainty).trim();
  if (!JUDGE_ASSESSMENTS.includes(assessment as JudgeResult["assessment"])) {
    errors.push(`assessment must be one of: ${JUDGE_ASSESSMENTS.join(" | ")}`);
  }
  if (!EVIDENCE_SCOPES.includes(evidenceScope as JudgeResult["evidenceScope"])) {
    errors.push(`evidenceScope must be one of: ${EVIDENCE_SCOPES.join(" | ")}`);
  }
  if (!reason) errors.push("reason must be a non-empty string");
  if (!remainingUncertainty) {
    errors.push("remainingUncertainty must be a non-empty string; use NONE when none remains");
  }
  if (errors.length > 0) return { value: null, errors };
  return {
    value: {
      assessment: assessment as JudgeResult["assessment"],
      evidenceScope: evidenceScope as JudgeResult["evidenceScope"],
      reason,
      remainingUncertainty,
    },
    errors: [],
  };
}

export function decisionCorrectionSuffix(errors: string[]): string {
  return [
    "",
    "上一次 final_answer 未通过状态机核心协议检查。保持同一语义判断，只修正 JSON。",
    `错误：${errors.join("；")}`,
    "期望格式：",
    '{"decision":"RUN_LAB | RUN_JUDGE | COMPLETE_SUPPORT | COMPLETE_REJECT | RETURN_TO_LEARNING | BLOCKED","evidenceScope":"DESIGN_AUDIT_ONLY | WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY","reason":"非空简短理由","experimentContract":{"objective":"一个决策性不确定性","comparison":"...","conditions":"...","stopConditions":["按优先级排列的终止条件"],"estimatedMinutes":120,"allowedWeakening":["..."],"forbiddenWeakening":["..."],"completionEvidence":"按退出路径描述最小证据"},"reviewFocus":null}',
    "RUN_LAB 才填写 experimentContract；RUN_JUDGE 才填写 reviewFocus；其余对应字段为 null。只输出一个 JSON 对象。",
  ].join("\n");
}

export function judgeCorrectionSuffix(errors: string[]): string {
  return [
    "",
    "上一次 final_answer 未通过评判协议检查。保持同一证据判断，只修正 JSON。",
    `错误：${errors.join("；")}`,
    "期望格式：",
    '{"assessment":"VALID_POSITIVE | VALID_NEGATIVE | INCONCLUSIVE | INVALID","evidenceScope":"DESIGN_AUDIT_ONLY | WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY","reason":"非空简短判断","remainingUncertainty":"主要未决项；没有则写 NONE"}',
    "只输出一个 JSON 对象。",
  ].join("\n");
}

function parseContract(
  value: unknown,
  errors: string[],
  limits: DecisionProtocolLimits,
): ExperimentContractProposal | null {
  if (value == null) return null;
  const object = asObject(value);
  if (!object) {
    errors.push("experimentContract must be an object or null");
    return null;
  }
  const objective = stringField(object.objective).trim();
  const comparison = stringField(object.comparison).trim();
  const conditions = stringField(object.conditions).trim();
  const completionEvidence = stringField(object.completionEvidence).trim();
  const stopConditions = nonEmptyStringArray(object.stopConditions, true);
  const allowedWeakening = nonEmptyStringArray(object.allowedWeakening, false);
  const forbiddenWeakening = nonEmptyStringArray(object.forbiddenWeakening, false);
  const estimatedMinutes = numberField(object.estimatedMinutes);

  for (const [name, field] of [
    ["objective", objective],
    ["comparison", comparison],
    ["conditions", conditions],
    ["completionEvidence", completionEvidence],
  ] as const) {
    if (!field) errors.push(`experimentContract.${name} must be a non-empty string`);
  }
  if (!stopConditions) {
    errors.push("experimentContract.stopConditions must be a non-empty array of non-empty strings");
  }
  if (!allowedWeakening) {
    errors.push("experimentContract.allowedWeakening must be an array of non-empty strings");
  }
  if (!forbiddenWeakening) {
    errors.push("experimentContract.forbiddenWeakening must be an array of non-empty strings");
  }
  if (estimatedMinutes === null || estimatedMinutes <= 0) {
    errors.push("experimentContract.estimatedMinutes must be a positive finite number");
  } else if (
    limits.maxEstimatedMinutes !== undefined &&
    estimatedMinutes > limits.maxEstimatedMinutes
  ) {
    errors.push(
      `experimentContract.estimatedMinutes must not exceed current Lab envelope ${limits.maxEstimatedMinutes}`,
    );
  }
  if (!objective || !comparison || !conditions || !completionEvidence ||
    !stopConditions || !allowedWeakening || !forbiddenWeakening ||
    estimatedMinutes === null || estimatedMinutes <= 0) return null;
  return {
    objective,
    comparison,
    conditions,
    stopConditions,
    estimatedMinutes,
    allowedWeakening,
    forbiddenWeakening,
    completionEvidence,
  };
}

function parseJsonObject(text: string): {
  value: Record<string, unknown> | null;
  error: string | null;
} {
  const candidates = [text.trim()];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  if (fenced) candidates.push(fenced);
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first >= 0 && last > first) candidates.push(text.slice(first, last + 1));
  for (const candidate of [...new Set(candidates)].filter(Boolean)) {
    try {
      const value = JSON.parse(candidate) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return { value: value as Record<string, unknown>, error: null };
      }
    } catch {
      // Try the next tolerant extraction candidate.
    }
  }
  return {
    value: null,
    error: "final answer does not contain one parseable JSON object",
  };
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringField(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function nonEmptyStringArray(value: unknown, requireItem: boolean): string[] | null {
  if (!Array.isArray(value) || (requireItem && value.length === 0)) return null;
  const result = value.map((item) => typeof item === "string" ? item.trim() : "");
  return result.every(Boolean) ? result : null;
}

function numberField(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
