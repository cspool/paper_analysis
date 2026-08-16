import type { LoopDecision } from "./types.ts";

export interface OutputCorrectionPrompt {
  previousOutputPath: string;
  errorReportPath: string;
  correctRefName: string;
}

export interface RuntimeRetryPrompt {
  previousTurnPath: string;
  failure: string;
  partialOutputPath: string | null;
}

export function buildContentPrompt(input: {
  skillName: "learning-loop-worker" | "learning-loop-reviewer";
  taskPath: string;
  guidance: string | null;
  correction?: OutputCorrectionPrompt | null;
  runtimeRetry?: RuntimeRetryPrompt | null;
}): string {
  const lines = [
    `使用 $${input.skillName}`,
    "",
    `本次任务：${input.taskPath}`,
    `Decision guidance：${input.guidance ?? "无"}`,
    "",
    "按照 Skill 指定的 Result Ref 输出一个 JSON 对象。",
  ];
  appendCorrection(lines, input.correction ?? null);
  appendRuntimeRetry(lines, input.runtimeRetry ?? null);
  return `${lines.join("\n")}\n`;
}

export function buildDecisionPrompt(input: {
  contextPath: string;
  allowed: LoopDecision[];
  correction?: OutputCorrectionPrompt | null;
  runtimeRetry?: RuntimeRetryPrompt | null;
}): string {
  const lines = [
    "使用 $learning-loop-decision",
    "",
    `本次决策上下文：${input.contextPath}`,
    "先按 Decision Skill 读取其中的 observationRef，再按需回读原始结论 Ref。",
    "",
    "[ALLOWED_DECISIONS]",
    ...input.allowed.map((decision) => `- ${decision}`),
    "",
    "[OUTPUT_PROTOCOL]",
    "decision = <一个允许的字面量>",
    "guidance = <可选的不透明自然语言；Script 只保存并转发，不解释>",
    ...(input.allowed.includes("RUN_EXP_GOAL")
      ? [
        "若 decision = RUN_EXP_GOAL，guidance 必须是一行非空、有界的 experiment objective；Script 原样注入 EXP Goal。",
      ]
      : []),
  ];
  appendCorrection(lines, input.correction ?? null);
  appendRuntimeRetry(lines, input.runtimeRetry ?? null);
  return `${lines.join("\n")}\n`;
}

export function buildExperimentGoalPrompt(input: {
  taskPath: string;
}): string {
  return [
    "使用 $learning-exp-goal",
    "",
    `本次 EXP Goal 任务：${input.taskPath}`,
    "读取任务中冻结的 current Anchor、optional Direction 和 experiment objective。",
    "在同一个 Goal 生命周期内迭代环境、代码、测量与诊断；将全部产物保存在任务指定 workspace。",
    "完成时给出简洁、证据化的自然语言结论并正确更新 Goal 状态。",
    "",
  ].join("\n");
}

function appendRuntimeRetry(
  lines: string[],
  retry: RuntimeRetryPrompt | null,
): void {
  if (!retry) return;
  lines.push(
    "",
    "[RUNTIME_RETRY]",
    `上次 Turn：${retry.previousTurnPath}`,
    `运行失败：${retry.failure}`,
  );
  if (retry.partialOutputPath) {
    lines.push(`未完成草稿：${retry.partialOutputPath}`);
  }
  lines.push(
    "",
    "重新执行同一冻结任务，返回一个完整、精简的协议结果。",
    "未完成草稿不是已校验结论；不得从断点续写，也不得让它改变任务绑定。",
  );
}

function appendCorrection(
  lines: string[],
  correction: OutputCorrectionPrompt | null,
): void {
  if (!correction) return;
  lines.push(
    "",
    "[OUTPUT_CORRECTION]",
    `上次输出：${correction.previousOutputPath}`,
    `错误报告：${correction.errorReportPath}`,
    `正确 Ref：${correction.correctRefName}`,
    "",
    "重做同一任务并返回完整结果。",
  );
}
