import { createWriteStream, existsSync, mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { spawn } from "node:child_process";
import type {
  BatchResult,
  BatchTask,
  RelevanceLayer,
} from "./types.ts";

export interface CodexAttemptOptions {
  codexBin: string;
  projectRoot: string;
  taskPath: string;
  schemaPath: string;
  attemptDir: string;
  model: string | null;
  useWebSearch: boolean;
  timeoutMs: number;
  correctionPath?: string | null;
}

export interface CodexAttemptOutcome {
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  timedOut: boolean;
  outputPath: string;
  providerRawPath: string;
  stderrPath: string;
  promptPath: string;
  error: string | null;
}

export function buildBatchPrompt(
  taskPath: string,
  correctionPath: string | null = null,
): string {
  return [
    "你是 Paper Catch 批处理 Loop 中的一个 fresh Codex 筛选会话。",
    "Script 已完成 Git 增量统计、标题提取、去重和 batch 绑定；你不控制 Loop，也不引入 batch 外论文。",
    "",
    `本批冻结任务：${taskPath}`,
    "先读取 task 中的兴趣主题、条目模板和 candidates。必要时可用网页搜索核对候选论文语义、开源库或发表信息。",
    "",
    "筛选规则：",
    "1. 只选择性能加速工作，允许算法层、框架层、系统层、硬件层或其组合。",
    "2. 排除以训练方法、训练扩展或训练效率为主要贡献且不面向推理/执行性能的工作。",
    "3. 优先更新更近、明确有开源实现、性能问题和 baseline change 清楚的工作。",
    "4. 不因标题出现 efficient/fast/optimization 就自动入选；依据候选上下文或核实证据判断。",
    "5. 不得猜测代码链接。无法确认时 openSource=UNKNOWN、codeUrl=null。",
    "6. 每个 candidateId 必须恰好出现在 selected 或 rejected 一次。",
    "7. selected 的语义字段必须足以让 Script 按 entry template 渲染一个独立条目。",
    "",
    "只输出符合 output schema 的一个 JSON 对象，不修改文件，不输出 Markdown 报告。",
    ...(correctionPath
      ? [
        "",
        "[OUTPUT_CORRECTION]",
        `上次输出校验错误：${correctionPath}`,
        "重新处理同一冻结 batch，返回完整 JSON；不得只补写缺失候选。",
      ]
      : []),
  ].join("\n");
}

export async function invokeCodexBatch(
  options: CodexAttemptOptions,
): Promise<CodexAttemptOutcome> {
  mkdirSync(options.attemptDir, { recursive: true });
  const outputPath = resolve(options.attemptDir, "output.json");
  const providerRawPath = resolve(options.attemptDir, "provider_raw.jsonl");
  const stderrPath = resolve(options.attemptDir, "stderr.log");
  const promptPath = resolve(options.attemptDir, "prompt.txt");
  const prompt = buildBatchPrompt(options.taskPath, options.correctionPath ?? null);
  await Bunless.writeText(promptPath, `${prompt}\n`);

  const args = [
    "-a",
    "never",
    ...(options.useWebSearch ? ["--search"] : ["-c", 'web_search="disabled"']),
    ...(options.model ? ["-m", options.model] : []),
    "-c",
    'model_reasoning_effort="high"',
    "exec",
    "--ephemeral",
    "--json",
    "--sandbox",
    "read-only",
    "--cd",
    options.projectRoot,
    "--output-schema",
    options.schemaPath,
    "--output-last-message",
    outputPath,
    "--color",
    "never",
    prompt,
  ];

  const raw = createWriteStream(providerRawPath, { flags: "w" });
  const stderr = createWriteStream(stderrPath, { flags: "w" });
  let timedOut = false;
  let spawnError: string | null = null;
  const child = spawn(options.codexBin, args, {
    cwd: options.projectRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.pipe(raw);
  child.stderr.on("data", (chunk) => {
    stderr.write(chunk);
    process.stderr.write(chunk);
  });
  child.on("error", (error) => {
    spawnError = `${error.name}: ${error.message}`;
  });

  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGTERM");
    setTimeout(() => {
      if (child.exitCode === null) child.kill("SIGKILL");
    }, 10_000).unref();
  }, options.timeoutMs);
  timer.unref();

  const completion = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>(
    (resolveCompletion) => {
      child.on("close", (code, signal) => resolveCompletion({ code, signal }));
    },
  );
  clearTimeout(timer);
  await Promise.all([closeStream(raw), closeStream(stderr)]);
  return {
    exitCode: completion.code,
    signal: completion.signal,
    timedOut,
    outputPath,
    providerRawPath,
    stderrPath,
    promptPath,
    error: spawnError,
  };
}

export function readAndValidateBatchResult(
  outputPath: string,
  task: BatchTask,
): { result: BatchResult | null; errors: string[] } {
  if (!existsSync(outputPath)) {
    return { result: null, errors: [`missing output file: ${outputPath}`] };
  }
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(outputPath, "utf8"));
  } catch (error) {
    return {
      result: null,
      errors: [`output is not valid JSON: ${error instanceof Error ? error.message : String(error)}`],
    };
  }
  const errors = validateBatchResult(value, task);
  return {
    result: errors.length === 0 ? value as BatchResult : null,
    errors,
  };
}

export function validateBatchResult(value: unknown, task: BatchTask): string[] {
  const errors: string[] = [];
  if (!isObject(value)) return ["result must be a JSON object"];
  if (value.schemaVersion !== "paper-catch-batch-result-v1") {
    errors.push("schemaVersion must be paper-catch-batch-result-v1");
  }
  if (value.runId !== task.runId) errors.push(`runId must equal ${task.runId}`);
  if (value.batchId !== task.batchId) errors.push(`batchId must equal ${task.batchId}`);
  if (!nonEmpty(value.batchSummary)) errors.push("batchSummary must be non-empty");
  if (!Array.isArray(value.selected)) errors.push("selected must be an array");
  if (!Array.isArray(value.rejected)) errors.push("rejected must be an array");
  if (errors.length > 0) return errors;

  const allowed = new Set(task.candidates.map((candidate) => candidate.candidateId));
  const observed = new Map<string, string>();
  for (const [index, selected] of (value.selected as unknown[]).entries()) {
    if (!isObject(selected)) {
      errors.push(`selected[${index}] must be an object`);
      continue;
    }
    validateCandidateId(selected.candidateId, `selected[${index}]`, allowed, observed, errors);
    if (!Array.isArray(selected.layers) || selected.layers.length === 0 ||
      selected.layers.some((layer) => !LAYERS.has(String(layer))) ||
      new Set(selected.layers.map(String)).size !== selected.layers.length) {
      errors.push(`selected[${index}].layers must contain allowed layer literals`);
    }
    if (!PRIORITIES.has(String(selected.priority))) {
      errors.push(`selected[${index}].priority is invalid`);
    }
    for (const field of [
      "relevanceReason",
      "performanceProblem",
      "conciseContribution",
      "newnessReason",
    ] as const) {
      if (!nonEmpty(selected[field])) errors.push(`selected[${index}].${field} must be non-empty`);
    }
    if (!OPEN_SOURCE.has(String(selected.openSource))) {
      errors.push(`selected[${index}].openSource is invalid`);
    }
    if (selected.codeUrl !== null &&
      (typeof selected.codeUrl !== "string" || !/^https?:\/\//.test(selected.codeUrl))) {
      errors.push(`selected[${index}].codeUrl must be null or an HTTP(S) URL`);
    }
  }
  for (const [index, rejected] of (value.rejected as unknown[]).entries()) {
    if (!isObject(rejected)) {
      errors.push(`rejected[${index}] must be an object`);
      continue;
    }
    validateCandidateId(rejected.candidateId, `rejected[${index}]`, allowed, observed, errors);
    if (!nonEmpty(rejected.reason)) errors.push(`rejected[${index}].reason must be non-empty`);
  }
  for (const candidateId of allowed) {
    if (!observed.has(candidateId)) errors.push(`candidate is not classified: ${candidateId}`);
  }
  return errors;
}

function validateCandidateId(
  value: unknown,
  path: string,
  allowed: Set<string>,
  observed: Map<string, string>,
  errors: string[],
): void {
  if (typeof value !== "string" || !allowed.has(value)) {
    errors.push(`${path}.candidateId is not bound to this batch`);
    return;
  }
  const previous = observed.get(value);
  if (previous) errors.push(`candidate ${value} appears in both/duplicate classifications: ${previous}, ${path}`);
  else observed.set(value, path);
}

function closeStream(stream: ReturnType<typeof createWriteStream>): Promise<void> {
  return new Promise((resolveClose) => {
    stream.end(resolveClose);
  });
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmpty(value: unknown): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

const LAYERS = new Set<RelevanceLayer>(["ALGORITHM", "FRAMEWORK", "SYSTEM", "HARDWARE"]);
const PRIORITIES = new Set(["HIGH", "MEDIUM", "LOW"]);
const OPEN_SOURCE = new Set(["YES", "NO", "UNKNOWN"]);

const Bunless = {
  async writeText(path: string, value: string): Promise<void> {
    const { writeFile } = await import("node:fs/promises");
    await writeFile(path, value, "utf8");
  },
};
