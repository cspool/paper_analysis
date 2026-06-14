#!/usr/bin/env -S npx tsx
/**
 * Idea Review Orchestrator — Pure Dual-Session Message Broker
 *
 * Spawns TWO persistent Claude sessions (Question Agent + Answer Agent),
 * routes normalized protocol payloads between them via stdin/stdout pipes.
 * All behavior is defined in the skill files — this script only:
 *   1. Starts sessions with skill content
 *   2. Waits for ready signals
 *   3. Parses and forwards only protocol payloads between sessions
 *   4. Injects allowlisted QA expert references on demand
 *   5. Detects completion marker
 *   6. Logs everything with timestamps
 *
 * Message markers (defined in skills):
 *   QA → ___QA_REFERENCE_REQUEST___, ___QA_QUESTION___, or ___JUDGMENT_COMPLETE___
 *   AA init → ___AA_INIT_COMPLETE___ ... ___AA_INIT_COMPLETE_END___
 *   AA answer → ___AA_OUTPUT_START___ ... ___AA_OUTPUT_END___
 *
 * Usage:
 *   npx tsx scripts/idea_review_orchestrator.ts --idea-note <path> [--max-rounds N] [--max-budget-usd USD] [--resume]
 */

import { spawn, ChildProcess } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import { existsSync, createWriteStream, mkdirSync } from "fs";
import * as readline from "readline";
import * as crypto from "crypto";

// ─── Config ─────────────────────────────────────────────────────────────────

const VAULT_ROOT = "/data3/paper_analysis";
const SKILLS_DIR = path.join(VAULT_ROOT, ".claude", "skills");
const REVIEW_NOTES_DIR = path.join(VAULT_ROOT, "review_notes");
const REVIEW_RUNS_DIR = path.join(VAULT_ROOT, ".claude", "idea-review-runs");
const CLAUDE_CMD = "claude";
// AA: block write tools only (AA needs read/search for context gathering)
const AA_DISALLOWED_TOOLS = [
  "Write", "Edit", "NotebookEdit",
  "mcp__obsidian__obsidian_write_note",
  "mcp__obsidian__obsidian_append_to_note",
  "mcp__obsidian__obsidian_patch_note",
  "mcp__obsidian__obsidian_replace_in_note",
].join(",");

// QA: block ALL tools — QA must be blind, no file access whatsoever
const QA_DISALLOWED_TOOLS = [
  // Obsidian MCP (all)
  "mcp__obsidian__obsidian_search_notes",
  "mcp__obsidian__obsidian_get_note",
  "mcp__obsidian__obsidian_list_notes",
  "mcp__obsidian__obsidian_list_tags",
  "mcp__obsidian__obsidian_manage_tags",
  "mcp__obsidian__obsidian_manage_frontmatter",
  "mcp__obsidian__obsidian_open_in_ui",
  "mcp__obsidian__obsidian_write_note",
  "mcp__obsidian__obsidian_append_to_note",
  "mcp__obsidian__obsidian_patch_note",
  "mcp__obsidian__obsidian_replace_in_note",
  "mcp__obsidian__obsidian_delete_note",
  // File system
  "Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob",
  // Web
  "WebSearch", "WebFetch",
].join(",");

const QA_SKILL_PATH = path.join(SKILLS_DIR, "idea_question", "SKILL.md");
const AA_SKILL_PATH = path.join(SKILLS_DIR, "idea_answer", "SKILL.md");
const QA_REFERENCE_DIR = path.join(SKILLS_DIR, "idea_question", "references");
const QA_REFERENCE_PATHS: Record<string, string> = {
  "动态(调度/并发)的背景/需求": path.join(QA_REFERENCE_DIR, "01-background-and-demand.md"),
  "并发方法的应用和实现": path.join(QA_REFERENCE_DIR, "02-concurrency-implementation.md"),
  "提供并发机制的硬件模块/架构": path.join(QA_REFERENCE_DIR, "03-hardware-mechanisms.md"),
  "影响并发的架构/机制": path.join(QA_REFERENCE_DIR, "04-architecture-limits.md"),
  "架构性能和开销的实验工具": path.join(QA_REFERENCE_DIR, "05-experiment-tools.md"),
};

const DEFAULT_MAX_ROUNDS = 8;
const DEFAULT_MAX_BUDGET_USD = 100;
const AA_RECOVERY_MAX_BUDGET_USD = 5;
const AGENT_TIMEOUT_MS = 600_000;
const PROTOCOL_VERSION = 8;

// ─── Types ──────────────────────────────────────────────────────────────────

interface Session {
  proc: ChildProcess;
  stdoutLines: readline.Interface;
  stdin: any;
  sessionId: string;
  label: string;
  agentRole: "qa" | "aa";
  pendingResolve: ((text: string) => void) | null;
  pendingReject: ((err: Error) => void) | null;
  accumulatedText: string[];
  protocolFallbacks: string[];
  lastResultSummary: TurnResultSummary | null;
  lastApiRetrySummary: ApiRetrySummary | null;
  logStream: NodeJS.WritableStream;
}

interface TurnResultSummary {
  subtype: string;
  is_error: boolean;
  stop_reason?: string | null;
  total_cost_usd?: number;
  input_tokens?: number;
  output_tokens?: number;
}

interface ApiRetrySummary {
  attempt: number;
  max_retries: number;
  retry_delay_ms?: number;
  error_status?: string | number | null;
  error?: string;
}

interface ConversationState {
  protocol_version: number;
  idea_note_path: string;
  idea_note_title: string;
  paper_title?: string;
  paper_subdir?: string;
  qa_session_id: string;
  aa_session_id: string;
  aa_no_tools?: boolean;
  round: number;
  qa_history: { round: number; question: string; answer: string }[];
  qa_loaded_references: string[];
  qa_next_entry: string;
  aa_next_entry: string;
  final_judgment: any;
  started_at: number;
  updated_at: number;
}

interface QAQuestionPayload {
  round: number;
  question: string;
  question_level: number;
  question_category: string;
  question_subcategory?: string;
}

interface QAReferenceRequestPayload {
  round: number;
  question_category: string;
}

interface AAOutputPayload {
  round: number;
  answer: string;
  sources: string[];
  information_gaps: string[];
}

interface AAInitPayload {
  paper_title: string;
  paper_subdir: string;
}

class IsolatedAARecoveryRequiredError extends Error {}

// ─── Helpers ────────────────────────────────────────────────────────────────

function now(): number { return Math.floor(Date.now() / 1000); }
function ts(): string { return new Date().toISOString(); }

function claudeResultError(label: string, result: any): Error {
  const subtype = typeof result.subtype === "string" ? result.subtype : "unknown";
  const details = Array.isArray(result.errors) && result.errors.length > 0
    ? result.errors.join("; ")
    : "Claude session returned an error result";
  const cost = typeof result.total_cost_usd === "number"
    ? `; cost=$${result.total_cost_usd.toFixed(4)}`
    : "";
  const hint = subtype === "error_max_budget_usd"
    ? " Set a higher explicit --max-budget-usd value only if that spend is acceptable."
    : "";
  return new Error(`${label} Claude result error (${subtype}): ${details}${cost}.${hint}`);
}

function summarizeApiRetry(summary: ApiRetrySummary | null): string {
  if (!summary) return "no API retry telemetry";
  const details = [`last API retry=${summary.attempt}/${summary.max_retries}`];
  if (typeof summary.retry_delay_ms === "number") {
    details.push(`next delay=${Math.round(summary.retry_delay_ms)}ms`);
  }
  if (summary.error_status !== undefined) details.push(`status=${String(summary.error_status)}`);
  if (summary.error) details.push(`error=${summary.error}`);
  return details.join(", ");
}

async function ensureDir(dir: string): Promise<void> {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function sanitizeDirName(name: string): string {
  return name.replace(/[<>:"/\\|?*]/g, "_").replace(/\s+/g, " ").trim().substring(0, 120);
}

function sanitizeReviewFileName(name: string): string {
  const sanitized = name
    .replace(/[\u0000-\u001f/\\]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .substring(0, 180);
  return sanitized && sanitized !== "." && sanitized !== ".." ? sanitized : "untitled";
}

function reviewPathForTitle(title: string): string {
  return path.join(REVIEW_NOTES_DIR, `${sanitizeReviewFileName(title)}_review.md`);
}

function buildSessionArgs(
  sessionId: string,
  agentRole: "qa" | "aa",
  maxBudgetUsd: number,
  resumeExisting: boolean,
  disallowedToolsOverride?: string,
  toolsOverride?: string,
): string[] {
  const args = [
    "--model", "deepseek-v4-flash[1m]",
    "--print", "--verbose",
    "--input-format", "stream-json",
    "--output-format", "stream-json",
    resumeExisting ? "--resume" : "--session-id", sessionId,
    "--permission-mode", "bypassPermissions",
    "--disallowedTools", disallowedToolsOverride
      ?? (agentRole === "qa" ? QA_DISALLOWED_TOOLS : AA_DISALLOWED_TOOLS),
    "--add-dir", VAULT_ROOT,
    "--max-budget-usd", String(maxBudgetUsd),
  ];
  if (toolsOverride !== undefined) args.push("--tools", toolsOverride);
  return args;
}

function canAutoResumeInterruptedQA(state: ConversationState): boolean {
  return !state.final_judgment
    && state.round > 0
    && state.qa_history.length === state.round
    && state.qa_next_entry.includes("await=AA_OUTPUT");
}

function canAutoResumeInterruptedAA(state: ConversationState): boolean {
  return !state.final_judgment
    && state.round > 0
    && state.qa_history.length === state.round - 1
    && state.qa_next_entry.includes("await=AA_OUTPUT")
    && state.aa_next_entry.includes("await=QA_QUESTION");
}

async function resolveIdeaNote(input: string): Promise<{
  resolvedPath: string;
  displayTitle: string;
  isAutoDiscovered: boolean;
}> {
  // If input is an existing file path, use it directly (legacy behavior)
  if (existsSync(input)) {
    return {
      resolvedPath: path.resolve(input),
      displayTitle: path.basename(input, path.extname(input)),
      isAutoDiscovered: false,
    };
  }

  // Input is a paper title — search idea_notes/ for a matching file
  const ideaNotesDir = path.join(VAULT_ROOT, "idea_notes");
  const keywords = input.toLowerCase().split(/\s+/).filter(w => w.length > 2);

  if (keywords.length > 0 && existsSync(ideaNotesDir)) {
    const files = await fs.readdir(ideaNotesDir);
    const mdFiles = files.filter(f => f.endsWith(".md"));

    // Score each file by how many keywords match (case-insensitive)
    const scored = mdFiles.map(f => {
      const lower = f.toLowerCase();
      const matchCount = keywords.filter(kw => lower.includes(kw)).length;
      return { file: f, score: matchCount };
    });
    scored.sort((a, b) => b.score - a.score);

    // If the best match has at least half the keywords, use it
    const threshold = Math.max(1, Math.ceil(keywords.length / 2));
    if (scored.length > 0 && scored[0].score >= threshold) {
      const matchedPath = path.join(ideaNotesDir, scored[0].file);
      console.log(`  Auto-discovered idea note: ${matchedPath}`);
      console.log(`  Match: ${scored[0].score}/${keywords.length} keywords (threshold: ${threshold})`);
      return {
        resolvedPath: matchedPath,
        displayTitle: input,
        isAutoDiscovered: true,
      };
    }

    // Show near-misses for debugging
    if (scored[0].score > 0) {
      console.log(`  Best idea-note match: "${scored[0].file}" (score=${scored[0].score}/${keywords.length}, threshold=${threshold})`);
    }
  }

  // No matching idea note — create a stub so AA can bootstrap from the paper
  const safeName = sanitizeDirName(input) || "untitled";
  const stubDir = path.join(VAULT_ROOT, ".claude", "idea-review-runs", safeName);
  await ensureDir(stubDir);
  const stubPath = path.join(stubDir, "_auto_stub.md");
  const stubContent = [
    `# ${input}`,
    ``,
    `## ${input}`,
    ``,
    `> 自动生成的 stub idea note（用户直接传入论文标题，无人工撰写的 idea note）。`,
    `> Answer Agent 必须在 paper_secs/ 中搜索确认完整论文标题和子目录位置。`,
  ].join("\n");
  await fs.writeFile(stubPath, stubContent, "utf-8");
  console.log(`  Created stub idea note: ${stubPath}`);
  return {
    resolvedPath: stubPath,
    displayTitle: input,
    isAutoDiscovered: true,
  };
}

// ─── I/O Logging ────────────────────────────────────────────────────────────

function ioLog(logDir: string, label: string, direction: "IN" | "OUT" | "RETRY", content: string): void {
  const entry = `[${ts()}] ${label} ${direction === "IN" ? "◄ IN" : "► OUT"}\n${content}\n---\n`;
  fs.appendFile(path.join(logDir, "io_log.txt"), entry).catch(() => {});
}

// ─── Session Management ─────────────────────────────────────────────────────

function spawnSession(
  sessionId: string,
  workDir: string,
  label: string,
  agentRole: "qa" | "aa",
  maxBudgetUsd: number,
  resumeExisting: boolean,
  disallowedToolsOverride?: string,
  toolsOverride?: string,
): Session {
  const logPath = path.join(workDir, `${label}_raw.jsonl`);
  const logStream = createWriteStream(logPath, { flags: "a" });

  const args = buildSessionArgs(
    sessionId,
    agentRole,
    maxBudgetUsd,
    resumeExisting,
    disallowedToolsOverride,
    toolsOverride,
  );

  const proc = spawn(CLAUDE_CMD, args, { cwd: VAULT_ROOT, stdio: ["pipe", "pipe", "pipe"] });

  const errLog = createWriteStream(path.join(workDir, `${label}_stderr.log`), { flags: "a" });
  proc.stderr?.pipe(errLog);

  const stdoutLines = readline.createInterface({ input: proc.stdout! });

  const session: Session = {
    proc, stdoutLines, stdin: proc.stdin!, sessionId, label, agentRole,
    pendingResolve: null, pendingReject: null, accumulatedText: [], protocolFallbacks: [],
    lastResultSummary: null, lastApiRetrySummary: null, logStream,
  };

  stdoutLines.on("line", (line: string) => {
    logStream.write(line + "\n");
    try {
      const data = JSON.parse(line);
      if (data.type === "assistant") {
        for (const c of data.message?.content || []) {
          if (c.type === "text") session.accumulatedText.push(c.text);
          // Some model adapters occasionally place the intended final protocol
          // response in a thinking block and emit no visible text. Keep only
          // marker-bearing candidates for narrow protocol recovery at result.
          else if (c.type === "thinking" && typeof c.thinking === "string"
              && hasAgentProtocolMarker(c.thinking, session.agentRole)) {
            session.protocolFallbacks.push(c.thinking);
          }
        }
      } else if (data.type === "system" && data.subtype === "api_retry") {
        session.lastApiRetrySummary = {
          attempt: Number.isInteger(data.attempt) ? data.attempt : 0,
          max_retries: Number.isInteger(data.max_retries) ? data.max_retries : 0,
          ...(typeof data.retry_delay_ms === "number" ? { retry_delay_ms: data.retry_delay_ms } : {}),
          ...(typeof data.error_status === "string" || typeof data.error_status === "number"
              || data.error_status === null
            ? { error_status: data.error_status }
            : {}),
          ...(typeof data.error === "string" ? { error: data.error } : {}),
        };
      } else if (data.type === "result") {
        if (session.pendingResolve || session.pendingReject) {
          session.lastResultSummary = {
            subtype: typeof data.subtype === "string" ? data.subtype : "unknown",
            is_error: Boolean(data.is_error),
            ...(typeof data.stop_reason === "string" || data.stop_reason === null
              ? { stop_reason: data.stop_reason }
              : {}),
            ...(typeof data.total_cost_usd === "number" ? { total_cost_usd: data.total_cost_usd } : {}),
            ...(typeof data.usage?.input_tokens === "number" ? { input_tokens: data.usage.input_tokens } : {}),
            ...(typeof data.usage?.output_tokens === "number" ? { output_tokens: data.usage.output_tokens } : {}),
          };
          const visibleText = session.accumulatedText.join("")
            || (typeof data.result === "string" ? data.result : "");
          const text = selectSessionOutput(visibleText, session.protocolFallbacks, session.agentRole);
          session.accumulatedText = [];
          session.protocolFallbacks = [];
          const resolve = session.pendingResolve;
          const reject = session.pendingReject;
          session.pendingResolve = null; session.pendingReject = null;
          if (data.is_error) reject?.(claudeResultError(session.label, data));
          else {
            if (text !== visibleText) {
              console.warn(`  ${session.label} protocol recovered from thinking-only response`);
            }
            resolve?.(text);
          }
        }
      }
    } catch { /* non-JSON line */ }
  });

  proc.on("error", (err) => {
    if (session.pendingReject) {
      const reject = session.pendingReject;
      session.pendingResolve = null; session.pendingReject = null;
      reject(err);
    }
  });
  proc.on("close", (code, signal) => {
    if (session.pendingReject) {
      const reject = session.pendingReject;
      session.pendingResolve = null; session.pendingReject = null;
      reject(new Error(
        `${session.label} Claude process exited before returning a result `
        + `(code=${String(code)}, signal=${String(signal)}; ${summarizeApiRetry(session.lastApiRetrySummary)})`,
      ));
    }
  });

  return session;
}

function sendToSession(session: Session, text: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    session.accumulatedText = [];
    session.protocolFallbacks = [];
    session.lastResultSummary = null;
    session.lastApiRetrySummary = null;
    const timer = setTimeout(() => {
      session.pendingResolve = null; session.pendingReject = null;
      reject(new Error(
        `${session.label} timeout after ${timeoutMs}ms (${summarizeApiRetry(session.lastApiRetrySummary)})`,
      ));
    }, timeoutMs);

    session.pendingResolve = (t: string) => { clearTimeout(timer); resolve(t); };
    session.pendingReject = (e: Error) => { clearTimeout(timer); reject(e); };

    const msg = JSON.stringify({
      type: "user",
      message: { role: "user", content: [{ type: "text", text }] },
    });
    session.stdin.write(msg + "\n");
  });
}

function isCompleteProtocolResponse(text: string, agentRole: "qa" | "aa"): boolean {
  return Boolean(extractCompleteProtocolResponse(text, agentRole));
}

function buildProtocolRepairPrompt(agentRole: "qa" | "aa", originalInput = ""): string {
  const allowed = agentRole === "qa"
    ? "___QA_QUESTION___、___QA_REFERENCE_REQUEST___、___JUDGMENT_COMPLETE___"
    : "___AA_OUTPUT_START___";
  const lines = [
    "[PROTOCOL_REPAIR]",
    "你上一条响应停在内部步骤，未产生完整协议块与 LOOP。",
    "保持当前 session 内部状态；不要重做上一条输入，不要重置队列、review_material、loaded_paths 或轮次。",
    "从刚才尚未完成的线性步骤继续执行，直到完成当前任务块的最后一个控制流步骤。",
    "禁止调用任何工具；使用当前 session 中已经完成的检索、证据和草稿。",
    `只输出恰好一个允许的协议块（${allowed}）及其对应的 [LOOP: ...]；不要输出分析、过程说明或致歉。`,
    "必须实际输出 marker 与 LOOP；禁止只说“协议块已输出”或“已完成”。",
  ];

  if (agentRole === "aa") {
    const marker = parseQAMarker(originalInput);
    const embeddedRound = originalInput.match(/"round"\s*:\s*(\d+)/)?.[1];
    const round = marker?.type === "question" ? marker.q.round : (embeddedRound || "<保持当前 round>");
    lines.push(
      "按以下字面骨架立即输出；填入现有证据即可，信息不足写入 gaps，不得继续检索：",
      "___AA_OUTPUT_START___",
      `{ "round": ${round} }`,
      "___AA_SOURCES_START___",
      "- <已使用的现有路径；没有则留空>",
      "___AA_SOURCES_END___",
      "___AA_GAPS_START___",
      "- <现有证据无法回答的缺口；没有则留空>",
      "___AA_GAPS_END___",
      "___AA_ANSWER_START___",
      "<直接输出当前问题的简洁回答>",
      "___AA_ANSWER_END___",
      "___AA_OUTPUT_END___",
      `[LOOP: §2 | await=QA_QUESTION | completed_round=${round}]`,
    );
  }
  return lines.join("\n");
}

function summarizeTurnResult(summary: TurnResultSummary | null): string {
  if (!summary) return "result telemetry unavailable";
  const details = [`subtype=${summary.subtype}`];
  if (summary.stop_reason !== undefined) details.push(`stop_reason=${String(summary.stop_reason)}`);
  if (typeof summary.total_cost_usd === "number") details.push(`session_cost=$${summary.total_cost_usd.toFixed(4)}`);
  if (typeof summary.input_tokens === "number") details.push(`input_tokens=${summary.input_tokens}`);
  if (typeof summary.output_tokens === "number") details.push(`output_tokens=${summary.output_tokens}`);
  return details.join(", ");
}

function describeProtocolResponseFailure(text: string, agentRole: "qa" | "aa"): string {
  const trimmed = text.trim();
  if (!trimmed) return "empty visible response";
  const hasLoop = Boolean(parseLoopMarker(text));
  const hasMarker = hasAgentProtocolMarker(text, agentRole);
  if (!hasMarker && !hasLoop) return `narrative-only response: ${JSON.stringify(trimmed.slice(0, 200))}`;
  if (!hasMarker) return "LOOP present but allowed protocol marker missing";
  if (!hasLoop) return "allowed protocol marker present but LOOP missing";
  return "protocol marker or payload is incomplete/invalid";
}

function shouldUseIsolatedAARecovery(text: string, summary: TurnResultSummary | null): boolean {
  return !text.trim()
    && (summary?.stop_reason === "tool_use" || summary?.stop_reason === null);
}

async function sendProtocolTurn(
  session: Session,
  text: string,
  workDir: string,
  label: string,
  timeoutMs: number,
): Promise<string> {
  let output = await sendToSession(session, text, timeoutMs);
  if (isCompleteProtocolResponse(output, session.agentRole)) return output;

  const initialFailure = describeProtocolResponseFailure(output, session.agentRole);
  const initialResult = session.lastResultSummary;
  if (session.agentRole === "aa" && shouldUseIsolatedAARecovery(output, initialResult)) {
    throw new IsolatedAARecoveryRequiredError(
      `${label} adapter returned an empty terminal result (${summarizeTurnResult(initialResult)}); `
      + "the current session cannot reliably complete an in-session repair.",
    );
  }
  const repairPrompt = buildProtocolRepairPrompt(session.agentRole, text);
  console.warn(`  ${label} returned no complete protocol; requesting one in-session repair`);
  ioLog(workDir, label, "RETRY", repairPrompt);
  output = await sendToSession(session, repairPrompt, Math.min(timeoutMs, 300_000));
  if (!isCompleteProtocolResponse(output, session.agentRole)) {
    const repairFailure = describeProtocolResponseFailure(output, session.agentRole);
    throw new Error(
      `${label} protocol repair failed: initial=${initialFailure} (${summarizeTurnResult(initialResult)}); `
      + `repair=${repairFailure} (${summarizeTurnResult(session.lastResultSummary)}). `
      + "Checkpoint retained; inspect raw logs and use --resume when the checkpoint is resumable.",
    );
  }
  return output;
}

/** Send init message, retry once if the billing header auto-response consumed it. */
async function sendInitToSession(
  session: Session,
  text: string,
  label: string,
  timeoutMs: number,
): Promise<string> {
  let output = await sendToSession(session, text, timeoutMs);
  // The very first message of a session may be consumed by the CLI's billing-header
  // auto-response (model sees only the header, not our init). Detect this by the
  // absence of any protocol markers and retry once.
  const hasProtocolMarker = parseLoopMarker(output)
    || output.includes("___QA_QUESTION___")
    || output.includes("___AA_INIT_COMPLETE___")
    || output.includes("___AA_OUTPUT_START___");
  if (!hasProtocolMarker) {
    console.log(`  ${label} billing-header auto-response consumed, retrying init...`);
    ioLog(path.dirname(session.logStream.path as string), label, "RETRY", output.substring(0, 200));
    output = await sendToSession(session, text, Math.min(timeoutMs, 300_000));
  }
  return output;
}

function closeSession(s: Session): void {
  try { s.stdin.end(); } catch { /* */ }
  try { s.stdoutLines.close(); } catch { /* */ }
  try { s.logStream.end(); } catch { /* */ }
  setTimeout(() => { try { s.proc.kill("SIGTERM"); } catch { /* */ } }, 3000);
}

// ─── Marker Parsing ─────────────────────────────────────────────────────────

function extractFirstJsonObject(text: string): { json: string; endIndex: number } | null {
  const start = text.indexOf("{");
  if (start < 0) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const char = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === "\"") inString = false;
      continue;
    }

    if (char === "\"") inString = true;
    else if (char === "{") depth++;
    else if (char === "}") {
      depth--;
      if (depth === 0) return { json: text.slice(start, i + 1), endIndex: i };
      if (depth < 0) return null;
    }
  }
  return null;
}

function parseMarkerJson(text: string, startMarker: string, endMarker: string): any | null {
  const block = extractMarkerBlock(text, startMarker, endMarker);
  if (block === null) return null;

  // Parse the first complete JSON object. Marker bodies may contain later
  // Markdown expressions such as batch={1,2,4}, which are not part of JSON.
  const extracted = extractFirstJsonObject(block);
  if (!extracted) return null;

  try { return JSON.parse(extracted.json); }
  catch { return null; }
}

function extractMarkerBlock(text: string, startMarker: string, endMarker: string): string | null {
  const start = text.indexOf(startMarker);
  if (start < 0) return null;

  const end = text.indexOf(endMarker, start + startMarker.length);
  if (end < 0) return null;
  return text.slice(start + startMarker.length, end).trim();
}

function formatQAQuestionPayload(q: QAQuestionPayload): string {
  const json = JSON.stringify({
    round: q.round,
    question_level: q.question_level,
    question_category: q.question_category,
    ...(q.question_subcategory ? { question_subcategory: q.question_subcategory } : {}),
  }, null, 2);
  return [
    "___QA_QUESTION___",
    json,
    "___QA_QUESTION_TEXT___",
    q.question.trim(),
    "___QA_QUESTION_TEXT_END___",
    "___QA_QUESTION_END___",
  ].join("\n");
}

function formatQAReferenceRequestPayload(r: QAReferenceRequestPayload): string {
  return [
    "___QA_REFERENCE_REQUEST___",
    JSON.stringify({
      round: r.round,
      question_category: r.question_category,
    }, null, 2),
    "___QA_REFERENCE_REQUEST_END___",
  ].join("\n");
}

function formatQAJudgmentPayload(j: any): string {
  const { _body, ...metadata } = j;
  return [
    "___JUDGMENT_COMPLETE___",
    JSON.stringify(metadata, null, 2),
    ...(typeof _body === "string" && _body.trim() ? [_body.trim()] : []),
    "___JUDGMENT_COMPLETE_END___",
  ].join("\n");
}

function formatQAReference(questionCategory: string, content: string): string {
  return [
    "___QA_REFERENCE_START___",
    `question_category: ${questionCategory}`,
    "",
    content.trim(),
    "___QA_REFERENCE_END___",
  ].join("\n");
}

function formatAAOutput(payload: AAOutputPayload): string {
  return [
    "___AA_OUTPUT_START___",
    JSON.stringify({
      round: payload.round,
    }, null, 2),
    "___AA_SOURCES_START___",
    ...payload.sources.map(source => `- ${source}`),
    "___AA_SOURCES_END___",
    "___AA_GAPS_START___",
    ...payload.information_gaps.map(gap => `- ${gap}`),
    "___AA_GAPS_END___",
    "___AA_ANSWER_START___",
    payload.answer.trim(),
    "___AA_ANSWER_END___",
    "___AA_OUTPUT_END___",
  ].join("\n");
}

function formatAAInitPayload(payload: AAInitPayload): string {
  return [
    "___AA_INIT_COMPLETE___",
    JSON.stringify(payload, null, 2),
    "___AA_INIT_COMPLETE_END___",
  ].join("\n");
}

function parseAAInitMarker(text: string): AAInitPayload | null {
  const init = parseMarkerJson(text, "___AA_INIT_COMPLETE___", "___AA_INIT_COMPLETE_END___");
  if (!init || typeof init.paper_title !== "string" || typeof init.paper_subdir !== "string") {
    return null;
  }
  const paperTitle = init.paper_title.trim();
  const paperSubdir = init.paper_subdir.trim();
  if (!paperTitle
      || paperTitle.includes("<")
      || paperTitle.includes(">")
      || !paperSubdir.startsWith("paper_secs/")
      || paperSubdir.includes("<")
      || paperSubdir.includes(">")) return null;
  return {
    paper_title: paperTitle,
    paper_subdir: paperSubdir,
  };
}

function parseQAMarker(text: string):
  | { type: "question"; q: QAQuestionPayload }
  | { type: "reference_request"; r: QAReferenceRequestPayload }
  | { type: "judgment"; j: any }
  | null {
  // Check for judgment (higher priority) — JSON ratings + Markdown body text
  const judgmentBlock = extractMarkerBlock(text, "___JUDGMENT_COMPLETE___", "___JUDGMENT_COMPLETE_END___");
  if (judgmentBlock !== null) {
    const judgment = parseMarkerJson(text, "___JUDGMENT_COMPLETE___", "___JUDGMENT_COMPLETE_END___");
    if (judgment) {
      // Body text is everything after the JSON's closing brace
      const extracted = extractFirstJsonObject(judgmentBlock);
      const bodyText = extracted && extracted.endIndex < judgmentBlock.length - 1
        ? judgmentBlock.slice(extracted.endIndex + 1).trim()
        : "";
      return { type: "judgment", j: { ...judgment, _body: bodyText } };
    }
  }

  // Check for a broker-managed expert reference request.
  const referenceRequest = parseMarkerJson(
    text,
    "___QA_REFERENCE_REQUEST___",
    "___QA_REFERENCE_REQUEST_END___",
  );
  if (referenceRequest
      && Number.isInteger(referenceRequest.round)
      && typeof referenceRequest.question_category === "string"
      && Object.hasOwn(QA_REFERENCE_PATHS, referenceRequest.question_category)) {
    return {
      type: "reference_request",
      r: {
        round: referenceRequest.round,
        question_category: referenceRequest.question_category,
      },
    };
  }

  // Check for question
  const questionJson = parseMarkerJson(text, "___QA_QUESTION___", "___QA_QUESTION_TEXT___");
  if (questionJson
      && Number.isInteger(questionJson.round)
      && (questionJson.question_level === 1 || questionJson.question_level === 2)
      && typeof questionJson.question_category === "string"
      && questionJson.question_category.trim()) {
    // Extract question text from marker block (not JSON — avoids escaping issues)
    const questionText = extractMarkerBlock(text, "___QA_QUESTION_TEXT___", "___QA_QUESTION_TEXT_END___");
    if (questionText === null || !questionText.trim()) return null;
    // The inner text-end marker already gives an unambiguous boundary. Accept a
    // missing redundant outer end marker, then normalize it before forwarding.
    return {
      type: "question",
      q: {
        round: questionJson.round,
        question: questionText.trim(),
        question_level: questionJson.question_level,
        question_category: questionJson.question_category,
        ...(typeof questionJson.question_subcategory === "string" && questionJson.question_subcategory.trim()
          ? { question_subcategory: questionJson.question_subcategory }
          : {}),
      },
    };
  }

  return null;
}

async function resolveQAReferenceRequests(
  qaSession: Session,
  workDir: string,
  state: ConversationState,
  expectedRound: number,
  initialOutput: string,
): Promise<string> {
  const marker = parseQAMarker(initialOutput);
  if (marker?.type !== "reference_request") return initialOutput;
  if (marker.r.round !== expectedRound) {
    throw new Error(
      `QA protocol error: expected reference request round ${expectedRound}, got ${marker.r.round}`,
    );
  }
  if (state.qa_loaded_references.includes(marker.r.question_category)) {
    throw new Error(`QA protocol error: duplicate reference request for ${marker.r.question_category}`);
  }

  const referenceContent = await fs.readFile(QA_REFERENCE_PATHS[marker.r.question_category], "utf-8");
  state.qa_loaded_references.push(marker.r.question_category);
  await saveState(workDir, state);

  const referenceInput = formatQAReference(marker.r.question_category, referenceContent);
  const qaAnchor = buildAnchor("Question Agent", state.qa_next_entry, "QA_REFERENCE",
    `已加载 reference: [${state.qa_loaded_references.join(", ")}]`);
  const qaInput = `${qaAnchor}\n${referenceInput}`;
  console.log(`  QA reference: ${marker.r.question_category}`);
  ioLog(workDir, "QA", "IN", qaInput);
  const qaOutput = await sendProtocolTurn(qaSession, qaInput, workDir, "QA", AGENT_TIMEOUT_MS);
  ioLog(workDir, "QA", "OUT", qaOutput);
  if (parseQAMarker(qaOutput)?.type === "reference_request") {
    throw new Error("QA protocol error: must produce a question or judgment after one reference injection");
  }
  return qaOutput;
}

function parseRawLineItems(block: string): string[] {
  return block
    .split(/\r?\n/)
    .map(line => line.trim().replace(/^[-*]\s+/, ""))
    .filter(Boolean);
}

function parseAAMarker(text: string): AAOutputPayload | null {
  const answer = extractMarkerBlock(text, "___AA_ANSWER_START___", "___AA_ANSWER_END___");
  if (answer === null || !answer.trim()) return null;

  // Protocol v7 keeps natural-language sources/gaps out of JSON so quotes,
  // formulas, and Markdown cannot corrupt the metadata header.
  const segmentedMetadata = parseMarkerJson(text, "___AA_OUTPUT_START___", "___AA_SOURCES_START___");
  const sourcesBlock = extractMarkerBlock(text, "___AA_SOURCES_START___", "___AA_SOURCES_END___");
  const gapsBlock = extractMarkerBlock(text, "___AA_GAPS_START___", "___AA_GAPS_END___");
  if (segmentedMetadata && Number.isInteger(segmentedMetadata.round)
      && sourcesBlock !== null && gapsBlock !== null) {
    return {
      round: segmentedMetadata.round,
      answer: answer.trim(),
      sources: parseRawLineItems(sourcesBlock),
      information_gaps: parseRawLineItems(gapsBlock),
    };
  }

  // Accept valid protocol v6 payloads when parsing logs or tests. Invalid v6
  // JSON remains rejected because safely repairing arbitrary prose is ambiguous.
  const legacyMetadata = parseMarkerJson(text, "___AA_OUTPUT_START___", "___AA_ANSWER_START___");
  if (!legacyMetadata
      || !Number.isInteger(legacyMetadata.round)
      || !Array.isArray(legacyMetadata.sources)
      || !Array.isArray(legacyMetadata.information_gaps)) {
    return null;
  }

  return {
    round: legacyMetadata.round,
    answer: answer.trim(),
    sources: legacyMetadata.sources.map((s: any) => String(s).trim()).filter(Boolean),
    information_gaps: legacyMetadata.information_gaps.map((g: any) => String(g).trim()).filter(Boolean),
  };
}

function describeAAProtocolError(text: string): string {
  if (!text.includes("___AA_OUTPUT_START___")) return "missing ___AA_OUTPUT_START___";
  if (!text.includes("___AA_ANSWER_START___") || !text.includes("___AA_ANSWER_END___")) {
    return "missing complete ___AA_ANSWER_START___/___AA_ANSWER_END___ body";
  }
  if (!text.includes("___AA_SOURCES_START___") || !text.includes("___AA_GAPS_START___")) {
    return "invalid legacy JSON metadata before ___AA_ANSWER_START___; sources/gaps may contain unescaped quotes";
  }
  return "invalid segmented metadata or missing sources/gaps end marker";
}

// ─── LOOP Marker ───────────────────────────────────────────────────────────

function hasAgentProtocolMarker(text: string, agentRole: "qa" | "aa"): boolean {
  if (agentRole === "aa") {
    return text.includes("___AA_INIT_COMPLETE___") || text.includes("___AA_OUTPUT_START___");
  }
  return text.includes("___QA_QUESTION___")
    || text.includes("___QA_REFERENCE_REQUEST___")
    || text.includes("___JUDGMENT_COMPLETE___");
}

function parseLoopMarker(text: string): string {
  // Take the LAST loop marker — it represents the next input's execution entry point.
  const matches = text.match(/\[LOOP:\s*([^\]]+)\]/g);
  if (!matches || matches.length === 0) return "";
  const last = matches[matches.length - 1];
  return last;
}

function extractCompleteProtocolResponse(text: string, agentRole: "qa" | "aa"): string {
  const loop = parseLoopMarker(text);
  if (!loop) return "";

  if (agentRole === "qa") {
    const qaMarker = parseQAMarker(text);
    if (qaMarker?.type === "question") {
      return `${formatQAQuestionPayload(qaMarker.q)}\n${loop}`;
    }
    if (qaMarker?.type === "reference_request") {
      return `${formatQAReferenceRequestPayload(qaMarker.r)}\n${loop}`;
    }
    if (qaMarker?.type === "judgment") {
      return `${formatQAJudgmentPayload(qaMarker.j)}\n${loop}`;
    }
    return "";
  }

  const aaMarker = parseAAMarker(text);
  if (aaMarker) return `${formatAAOutput(aaMarker)}\n${loop}`;

  const aaInitMarker = parseAAInitMarker(text);
  if (aaInitMarker) return `${formatAAInitPayload(aaInitMarker)}\n${loop}`;
  return "";
}

function selectSessionOutput(
  visibleText: string,
  protocolFallbacks: string[],
  agentRole: "qa" | "aa",
): string {
  if (extractCompleteProtocolResponse(visibleText, agentRole)) return visibleText;

  for (let i = protocolFallbacks.length - 1; i >= 0; i--) {
    const recovered = extractCompleteProtocolResponse(protocolFallbacks[i], agentRole);
    if (recovered) return recovered;
  }
  return visibleText;
}

function requireNextEntry(text: string, label: string): string {
  const nextEntry = parseLoopMarker(text);
  if (!nextEntry) {
    throw new Error(`${label} protocol error: missing [LOOP: <next execution §> | ...]`);
  }
  return nextEntry;
}

const QA_LOOP_TASKS: Record<string, string> = {
  "§1": "§1 / §INIT — 初始化",
  "§INIT": "§1 / §INIT — 初始化",
  "§2": "§2 / §R1 — 首轮盲问",
  "§R1": "§2 / §R1 — 首轮盲问",
  "§3": "§3 / §SCREEN — 初筛维度",
  "§SCREEN": "§3 / §SCREEN — 初筛维度",
  "§4": "§4 / §DIM_NEXT — 选取下一维度",
  "§DIM_NEXT": "§4 / §DIM_NEXT — 选取下一维度",
  "§5": "§5 / §DIM_REF — 加载专家知识",
  "§DIM_REF": "§5 / §DIM_REF — 加载专家知识",
  "§6": "§6 / §DIM_ASK — 生成追问",
  "§DIM_ASK": "§6 / §DIM_ASK — 生成追问",
  "§7": "§7 / §DIM_EVAL — 评估回答",
  "§DIM_EVAL": "§7 / §DIM_EVAL — 评估回答",
  "§8": "§8 / §JUDGE — 最终评判",
  "§JUDGE": "§8 / §JUDGE — 最终评判",
  "§TERMINATED": "终止状态",
};

const AA_LOOP_TASKS: Record<string, string> = {
  "§1": "§1 / §INIT — 初始化",
  "§INIT": "§1 / §INIT — 初始化",
  "§2": "§2 / §ANSWER — 接收问题并回答",
  "§ANSWER": "§2 / §ANSWER — 接收问题并回答",
  "§TERMINATED": "终止状态",
};

const QA_LOOP_ACTIONS: Record<string, string> = {
  "§1": "完成初始化并输出 ready，然后暂停等待开始盲评信号",
  "§INIT": "完成初始化并输出 ready，然后暂停等待开始盲评信号",
  "§2": "直接生成固定的五大类别总览问题，然后暂停等待 Answer Agent 回答",
  "§R1": "直接生成固定的五大类别总览问题，然后暂停等待 Answer Agent 回答",
  "§3": "读取首轮回答，完成五大类别初筛，并按筛选结果继续选维度或进入最终评判",
  "§SCREEN": "读取首轮回答，完成五大类别初筛，并按筛选结果继续选维度或进入最终评判",
  "§4": "选择下一个 pending 维度；若全部维度完成则进入最终评判",
  "§DIM_NEXT": "选择下一个 pending 维度；若全部维度完成则进入最终评判",
  "§5": "检查当前维度的专家 reference；已加载则继续生成追问，否则请求 reference 后暂停",
  "§DIM_REF": "检查当前维度的专家 reference；已加载则继续生成追问，否则请求 reference 后暂停",
  "§6": "结合当前维度、已有回答和专家 reference 生成具体追问，然后暂停等待 Answer Agent 回答",
  "§DIM_ASK": "结合当前维度、已有回答和专家 reference 生成具体追问，然后暂停等待 Answer Agent 回答",
  "§7": "评估 Answer Agent 回答，更新当前维度状态，并按结果继续追问或处理下一维度",
  "§DIM_EVAL": "评估 Answer Agent 回答，更新当前维度状态，并按结果继续追问或处理下一维度",
  "§8": "汇总 review_material，输出最终评判与终止状态",
  "§JUDGE": "汇总 review_material，输出最终评判与终止状态",
};

const AA_LOOP_ACTIONS: Record<string, string> = {
  "§1": "定位论文、确认 canonical 标题、建立初始化证据上下文，并输出初始化完成协议",
  "§INIT": "定位论文、确认 canonical 标题、建立初始化证据上下文，并输出初始化完成协议",
  "§2": "提取当前问题，使用已有证据并按需补充检索，输出自包含回答后暂停等待下一问题",
  "§ANSWER": "提取当前问题，使用已有证据并按需补充检索，输出自包含回答后暂停等待下一问题",
};

const RECEIVED_SIGNAL_MEANINGS: Record<string, string> = {
  INIT: "初始化输入",
  START_REVIEW: "`START_REVIEW` 开始盲评信号",
  AA_OUTPUT: "Answer Agent 的 `AA_OUTPUT` 回答",
  QA_REFERENCE: "当前维度的 `QA_REFERENCE` 专家 reference",
  QA_QUESTION: "Question Agent 的 `QA_QUESTION` 问题",
};

function describeLoopSemantics(agentLabel: string, nextEntry: string, receivedSignal: string): string {
  const match = nextEntry.match(/^\[LOOP:\s*([^\]]+)\]$/);
  if (!match) {
    return "无法解析上次保存的恢复点；不得猜测入口。";
  }

  const fields = match[1].split("|").map(field => field.trim()).filter(Boolean);
  const entry = fields[0];
  const metadata = fields.slice(1).map(field => {
    const separator = field.indexOf("=");
    return separator >= 0
      ? { key: field.slice(0, separator).trim(), value: field.slice(separator + 1).trim() }
      : { key: field, value: "" };
  });
  const taskMap = agentLabel.includes("Question") || agentLabel === "QA"
    ? QA_LOOP_TASKS
    : AA_LOOP_TASKS;
  const actionMap = agentLabel.includes("Question") || agentLabel === "QA"
    ? QA_LOOP_ACTIONS
    : AA_LOOP_ACTIONS;
  const task = taskMap[entry] || `${entry} 对应的任务块`;
  const action = actionMap[entry] || "按该任务块的线性步骤处理当前输入并完成最后一个控制流步骤";
  const expectedSignal = metadata.find(item => item.key === "await")?.value;
  const state = metadata.filter(item => item.key !== "await");

  const signal = RECEIVED_SIGNAL_MEANINGS[receivedSignal] || `\`${receivedSignal}\` 信号`;
  const stateDescriptions = state.map(item => {
    if (item.key === "dimension") return `当前维度为「${item.value}」`;
    if (item.key === "round") return `当前轮次为 ${item.value}`;
    if (item.key === "completed_round") return `此前已完成 round ${item.value}`;
    if (item.key === "loaded_paths") return `当前已加载 ${item.value} 条证据路径`;
    if (item.key === "paper_title") return `已确认论文标题为「${item.value}」`;
    if (item.key === "paper_subdir") return `论文目录为「${item.value}」`;
    if (item.key === "recovery") return `当前处于 ${item.value} 恢复模式`;
    return item.value ? `${item.key} 为 ${item.value}` : item.key;
  });
  const mismatch = expectedSignal && receivedSignal && expectedSignal !== receivedSignal
    ? `注意：恢复点原本等待 \`${expectedSignal}\`，但本次收到的是 ${signal}；先检查载荷是否匹配。`
    : "";

  return [
    `从 \`${task}\` 开始；已经收到 ${signal}。`,
    stateDescriptions.length > 0 ? `${stateDescriptions.join("，")}。` : "",
    `本轮请${action}。`,
    mismatch,
  ].filter(Boolean).join("\n");
}

function buildAnchor(
  agentLabel: string,
  nextEntry: string,
  receivedSignal: string,
  extraContext: string,
): string {
  const parts = [`[第 N 轮 — ${agentLabel}]`];
  if (nextEntry) {
    parts.push("本次执行语义：");
    parts.push(describeLoopSemantics(agentLabel, nextEntry, receivedSignal));
  }
  if (extraContext) parts.push(extraContext);
  parts.push("── 协议载荷 ──");
  return parts.join("\n");
}

// ─── State Persistence ──────────────────────────────────────────────────────

async function saveState(workDir: string, state: ConversationState): Promise<void> {
  state.updated_at = now();
  await fs.writeFile(path.join(workDir, "conversation_state.json"), JSON.stringify(state, null, 2), "utf-8");
}

async function loadState(workDir: string): Promise<ConversationState | null> {
  const p = path.join(workDir, "conversation_state.json");
  if (!existsSync(p)) return null;
  try { return JSON.parse(await fs.readFile(p, "utf-8")) as ConversationState; }
  catch { return null; }
}

function parseFinalJudgmentFromRawJsonl(rawJsonl: string): any | null {
  const lines = rawJsonl.trim().split(/\r?\n/).reverse();
  for (const line of lines) {
    try {
      const data = JSON.parse(line);
      const candidates: string[] = [];
      if (data.type === "result" && typeof data.result === "string") {
        candidates.push(data.result);
      }
      if (data.type === "assistant") {
        for (const content of data.message?.content || []) {
          if (content.type === "text" && typeof content.text === "string") {
            candidates.push(content.text);
          } else if (content.type === "thinking" && typeof content.thinking === "string") {
            candidates.push(content.thinking);
          }
        }
      }

      for (const candidate of candidates) {
        const marker = parseQAMarker(candidate);
        if (marker?.type === "judgment") return marker.j;
      }
    } catch { /* ignore malformed or non-JSON raw log lines */ }
  }
  return null;
}

function parseLastQAQuestionFromRawJsonl(rawJsonl: string, expectedRound: number): QAQuestionPayload | null {
  const lines = rawJsonl.trim().split(/\r?\n/).reverse();
  for (const line of lines) {
    try {
      const data = JSON.parse(line);
      const candidates: string[] = [];
      if (data.type === "result" && typeof data.result === "string") {
        candidates.push(data.result);
      }
      if (data.type === "assistant") {
        for (const content of data.message?.content || []) {
          if (content.type === "text" && typeof content.text === "string") {
            candidates.push(content.text);
          } else if (content.type === "thinking" && typeof content.thinking === "string") {
            candidates.push(content.thinking);
          }
        }
      }

      for (const candidate of candidates) {
        const marker = parseQAMarker(candidate);
        if (marker?.type === "question" && marker.q.round === expectedRound) return marker.q;
      }
    } catch { /* ignore malformed or non-JSON raw log lines */ }
  }
  return null;
}

async function recoverFinalJudgmentFromLogs(workDir: string): Promise<any | null> {
  const rawPath = path.join(workDir, "QA_raw.jsonl");
  if (!existsSync(rawPath)) return null;
  try {
    return parseFinalJudgmentFromRawJsonl(await fs.readFile(rawPath, "utf-8"));
  } catch {
    return null;
  }
}

async function recoverLastQAQuestionFromLogs(
  workDir: string,
  expectedRound: number,
): Promise<QAQuestionPayload | null> {
  const rawPath = path.join(workDir, "QA_raw.jsonl");
  if (!existsSync(rawPath)) return null;
  try {
    return parseLastQAQuestionFromRawJsonl(await fs.readFile(rawPath, "utf-8"), expectedRound);
  } catch {
    return null;
  }
}

// ─── Output ─────────────────────────────────────────────────────────────────

async function writeReview(state: ConversationState): Promise<string> {
  const title = state.paper_title?.trim() || state.idea_note_title;
  await ensureDir(REVIEW_NOTES_DIR);
  const outputPath = reviewPathForTitle(title);
  let md = `# ${title}\n\n> ${ts()}\n> Rounds: ${state.round}\n> QA: \`${state.qa_session_id}\`  AA: \`${state.aa_session_id}\`\n\n`;

  if (state.final_judgment) {
    const j = state.final_judgment;
    md += `## 评判\n\n| 维度 | 评定 |\n|------|------|\n`;
    md += `| 相关性 | ${j.relevance} |\n`;
    md += `| 参考价值 | ${j.reference_value} |\n`;
    md += `| 深入价值 | ${j.depth_value} |\n`;
    if (j._body) {
      md += `\n${j._body}\n`;
    }
    md += `\n`;
  }

  md += `## 问答记录\n\n`;
  for (const qa of state.qa_history) {
    md += `### Round ${qa.round}\n\n**Q**：${qa.question}\n\n**A**：${qa.answer}\n\n---\n\n`;
  }

  await fs.writeFile(outputPath, md, "utf-8");
  return outputPath;
}

// ─── CLI ────────────────────────────────────────────────────────────────────

function parseArgs() {
  const a = process.argv.slice(2);
  let ideaNote = "", workDir = "", maxRounds = DEFAULT_MAX_ROUNDS;
  let maxBudgetUsd = DEFAULT_MAX_BUDGET_USD, resume = false;

  const readPathArg = (start: number): { value: string; lastIndex: number } => {
    const parts: string[] = [];
    let i = start;
    while (i < a.length && !a[i].startsWith("--")) parts.push(a[i++]);
    return { value: parts.join(" "), lastIndex: i - 1 };
  };

  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--idea-note" && i + 1 < a.length) {
      const parsed = readPathArg(i + 1);
      ideaNote = parsed.value;
      i = parsed.lastIndex;
    } else if (a[i] === "--work-dir" && i + 1 < a.length) {
      const parsed = readPathArg(i + 1);
      workDir = parsed.value;
      i = parsed.lastIndex;
    } else if (a[i] === "--max-rounds" && i + 1 < a.length) maxRounds = parseInt(a[++i], 10);
    else if (a[i] === "--max-budget-usd" && i + 1 < a.length) maxBudgetUsd = Number(a[++i]);
    else if (a[i] === "--resume") resume = true;
  }
  if (!ideaNote) { console.error("ERROR: --idea-note <path|title> required (file path or paper title)"); process.exit(1); }
  if (!Number.isInteger(maxRounds) || maxRounds < 1) {
    console.error("ERROR: --max-rounds must be a positive integer");
    process.exit(1);
  }
  if (!Number.isFinite(maxBudgetUsd) || maxBudgetUsd <= 0) {
    console.error("ERROR: --max-budget-usd must be a positive number");
    process.exit(1);
  }
  if (!workDir) workDir = path.join(REVIEW_RUNS_DIR, path.basename(ideaNote, path.extname(ideaNote)));
  return { ideaNote, workDir, maxRounds, maxBudgetUsd, resume };
}

// ─── Init Messages ──────────────────────────────────────────────────────────

async function buildQAInitMsg(): Promise<string> {
  const skillContent = await fs.readFile(QA_SKILL_PATH, "utf-8");
  const initLoop = "[LOOP: §INIT | await=INIT]";
  return `${skillContent}

---
本次执行语义：
${describeLoopSemantics("Question Agent", initLoop, "INIT")}
按 \`§1 / §INIT\` 的线性步骤执行：输出 "Question Agent 就绪，等待输入。" 和 \`[LOOP: §2 | await=START_REVIEW]\`，然后按最后一个步骤暂停。`;
}

async function buildAAInitMsg(ideaNoteContent: string, ideaNoteHint: string, ideaNotePath: string): Promise<string> {
  const skillContent = await fs.readFile(AA_SKILL_PATH, "utf-8");
  const initLoop = "[LOOP: §INIT | await=INIT]";
  return `${skillContent}

---
## 当前 Idea Note
**线索标题（来自文件名，可能不精确，需自行在 paper_secs/ 中确认完整论文标题和子目录）**：${ideaNoteHint}
**路径**：${ideaNotePath}
**内容**：
${ideaNoteContent.substring(0, 5000)}

---
本次执行语义：
${describeLoopSemantics("Answer Agent", initLoop, "INIT")}
按 \`§1 / §INIT\` 的线性步骤执行。必须一次性完成论文定位、从论文主文件 H1 确认 canonical paper_title、一级上下文和二级上下文三层获取，再输出 \`___AA_INIT_COMPLETE___\` 协议与 \`[LOOP: §2 | await=QA_QUESTION | ...]\`，然后按最后一个步骤暂停。`;
}

function buildAAIsolatedRecoveryPrompt(state: ConversationState, question: QAQuestionPayload): string {
  const history = state.qa_history
    .slice(-5)
    .map(item => `## Round ${item.round}\nQ: ${item.question}\nA: ${item.answer}`)
    .join("\n\n");
  const boundedHistory = history.length > 40_000 ? history.slice(-40_000) : history;

  return [
    "# Answer Agent isolated recovery",
    "",
    "你是接管中断轮次的新持久 Answer Agent。原 AA 会话在工具调用后被 CLI/适配器空终止。",
    "本会话禁止调用任何工具；只根据下方 canonical 信息、既有问答和当前问题作答。",
    "证据不足时必须写入 information_gaps，不得编造事实、数字或来源。",
    "本次及后续每次回答都只输出一个完整 ___AA_OUTPUT_START___ 协议块和对应 LOOP，不输出分析过程。",
    "",
    `canonical paper_title: ${state.paper_title || state.idea_note_title}`,
    `paper_subdir: ${state.paper_subdir || "<unknown>"}`,
    "本次执行语义：",
    describeLoopSemantics("Answer Agent", state.aa_next_entry, "QA_QUESTION"),
    "",
    "# Existing Q&A history",
    boundedHistory || "(none)",
    "",
    "# Current question",
    formatQAQuestionPayload(question),
    "",
    "# Required output",
    "___AA_OUTPUT_START___",
    `{ "round": ${question.round} }`,
    "___AA_SOURCES_START___",
    "___AA_SOURCES_END___",
    "___AA_GAPS_START___",
    "- <无法由现有历史可靠回答的缺口；没有则留空>",
    "___AA_GAPS_END___",
    "___AA_ANSWER_START___",
    "<直接回答当前问题>",
    "___AA_ANSWER_END___",
    "___AA_OUTPUT_END___",
    `[LOOP: §2 | await=QA_QUESTION | completed_round=${question.round} | recovery=no_tools]`,
  ].join("\n");
}

async function recoverAAWithFreshNoToolSession(
  oldSession: Session,
  workDir: string,
  state: ConversationState,
  question: QAQuestionPayload,
  maxBudgetUsd: number,
): Promise<{ session: Session; output: string }> {
  closeSession(oldSession);

  const sessionId = crypto.randomUUID();
  const recoveryBudget = Math.min(maxBudgetUsd, AA_RECOVERY_MAX_BUDGET_USD);
  const session = spawnSession(
    sessionId,
    workDir,
    "AA",
    "aa",
    recoveryBudget,
    false,
    QA_DISALLOWED_TOOLS,
    "",
  );
  state.aa_session_id = sessionId;
  state.aa_no_tools = true;
  await saveState(workDir, state);

  try {
    console.warn(`  AA switched to isolated no-tool recovery session ${sessionId} (budget cap $${recoveryBudget})`);
    const recoveryPrompt = buildAAIsolatedRecoveryPrompt(state, question);
    ioLog(workDir, "AA", "IN", recoveryPrompt);
    let output = await sendInitToSession(session, recoveryPrompt, "AA recovery", AGENT_TIMEOUT_MS);
    ioLog(workDir, "AA", "OUT", output);
    if (isCompleteProtocolResponse(output, "aa")) return { session, output };

    const initialFailure = describeProtocolResponseFailure(output, "aa");
    const initialResult = session.lastResultSummary;
    if (shouldUseIsolatedAARecovery(output, initialResult)) {
      throw new Error(
        `AA isolated no-tool recovery also returned an empty terminal result: `
        + `${initialFailure} (${summarizeTurnResult(initialResult)}).`,
      );
    }

    const repairPrompt = buildProtocolRepairPrompt("aa", formatQAQuestionPayload(question));
    console.warn("  AA isolated recovery returned no complete protocol; requesting one in-session repair");
    ioLog(workDir, "AA", "RETRY", repairPrompt);
    output = await sendToSession(session, repairPrompt, Math.min(AGENT_TIMEOUT_MS, 300_000));
    ioLog(workDir, "AA", "OUT", output);
    if (!isCompleteProtocolResponse(output, "aa")) {
      throw new Error(
        `AA isolated no-tool recovery protocol repair failed: initial=${initialFailure} `
        + `(${summarizeTurnResult(initialResult)}); repair=${describeProtocolResponseFailure(output, "aa")} `
        + `(${summarizeTurnResult(session.lastResultSummary)}).`,
      );
    }
    return { session, output };
  } catch (error) {
    closeSession(session);
    throw error;
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const { ideaNote, workDir: workDirFromArgs, maxRounds, maxBudgetUsd, resume } = parseArgs();

  // Resolve input: could be a file path OR a paper title
  const resolved = await resolveIdeaNote(ideaNote);
  const workDir = workDirFromArgs || path.join(REVIEW_RUNS_DIR, sanitizeDirName(resolved.displayTitle));

  console.log("══════════════════════════════════════════════════");
  console.log("  Idea Review Orchestrator — Dual-Session Broker");
  console.log("══════════════════════════════════════════════════");
  console.log(`  Input:  ${ideaNote}`);
  if (resolved.isAutoDiscovered) {
    console.log(`  Source: ${resolved.resolvedPath} (auto-discovered)`);
  }
  console.log(`  Title:  ${resolved.displayTitle}`);
  console.log(`  Run:    ${workDir}`);
  console.log("  Review: pending AA canonical-title confirmation");
  console.log(`  Max:    ${maxRounds} rounds${resume ? " (resume)" : ""}`);
  console.log(`  Budget: $${maxBudgetUsd} per agent session`);
  console.log("══════════════════════════════════════════════════\n");

  await ensureDir(workDir);

  // ── Load or init state ────────────────────────────────────────────────
  let state: ConversationState;
  if (resume) {
    const loaded = await loadState(workDir);
    if (!loaded) { console.error("No checkpoint found"); process.exit(1); }
    if (loaded.protocol_version !== PROTOCOL_VERSION) {
      console.error(`Checkpoint protocol ${loaded.protocol_version ?? "legacy"} cannot resume under blind-QA protocol ${PROTOCOL_VERSION}`);
      process.exit(1);
    }
    state = loaded;
  } else {
    state = {
      protocol_version: PROTOCOL_VERSION,
      idea_note_path: resolved.resolvedPath,
      idea_note_title: resolved.displayTitle,
      paper_title: "",
      paper_subdir: "",
      qa_session_id: crypto.randomUUID(), aa_session_id: crypto.randomUUID(),
      round: 0, qa_history: [], qa_loaded_references: [], qa_next_entry: "", aa_next_entry: "", final_judgment: null,
      started_at: now(), updated_at: now(),
    };
    await saveState(workDir, state);
  }

  // A terminated QA session must never be restarted. Older parser failures may
  // have saved the terminal LOOP before saving the final judgment; recover that
  // judgment from the raw stream and materialize the review without new spend.
  if (resume && state.qa_next_entry.includes("§TERMINATED")) {
    if (!state.final_judgment) {
      state.final_judgment = await recoverFinalJudgmentFromLogs(workDir);
      if (!state.final_judgment) {
        throw new Error("Terminated checkpoint has no recoverable ___JUDGMENT_COMPLETE___ in QA_raw.jsonl");
      }
      state.round = state.qa_history.length;
      await saveState(workDir, state);
      console.log("  Recovered final judgment from terminated QA raw log.");
    }
    const outputPath = await writeReview(state);
    console.log(`  Recovered review: ${outputPath}`);
    console.log(`  Rounds: ${state.round}`);
    return;
  }

  // ── Read idea note ────────────────────────────────────────────────────
  const ideaContent = await fs.readFile(resolved.resolvedPath, "utf-8");
  const ideaTitle = state.idea_note_title;

  // ── Init sessions ─────────────────────────────────────────────────────
  console.log("── Starting sessions ──\n");

  const qaSession = spawnSession(state.qa_session_id, workDir, "QA", "qa", maxBudgetUsd, resume);
  let aaSession = spawnSession(
    state.aa_session_id,
    workDir,
    "AA",
    "aa",
    state.aa_no_tools ? Math.min(maxBudgetUsd, AA_RECOVERY_MAX_BUDGET_USD) : maxBudgetUsd,
    resume,
    state.aa_no_tools ? QA_DISALLOWED_TOOLS : undefined,
    state.aa_no_tools ? "" : undefined,
  );
  console.log(`QA: ${qaSession.sessionId} (pid ${qaSession.proc.pid})`);
  console.log(`AA: ${aaSession.sessionId} (pid ${aaSession.proc.pid})\n`);

  try {
    let qaOutput: string;
    let startRound: number;
    let repairInterruptedAA = false;

    if (resume) {
      if (canAutoResumeInterruptedAA(state)) {
        const pendingQuestion = await recoverLastQAQuestionFromLogs(workDir, state.round);
        if (!pendingQuestion) {
          throw new Error(`Checkpoint cannot resume interrupted AA round ${state.round}: no recoverable QA question in QA_raw.jsonl`);
        }
        console.log(`── Repairing interrupted AA round ${state.round} ──`);
        qaOutput = `${formatQAQuestionPayload(pendingQuestion)}\n${state.qa_next_entry}`;
        startRound = state.round;
        repairInterruptedAA = true;
      } else if (canAutoResumeInterruptedQA(state)) {
        console.log("── Repairing interrupted QA turn ──");
        const resumePrompt = buildProtocolRepairPrompt("qa");
        ioLog(workDir, "QA", "RETRY", resumePrompt);
        qaOutput = await sendProtocolTurn(qaSession, resumePrompt, workDir, "QA", AGENT_TIMEOUT_MS);
        ioLog(workDir, "QA", "OUT", qaOutput);
        state.qa_next_entry = requireNextEntry(qaOutput, "QA resume action");
        qaOutput = await resolveQAReferenceRequests(
          qaSession,
          workDir,
          state,
          state.round + 1,
          qaOutput,
        );
        state.qa_next_entry = requireNextEntry(qaOutput, "QA resumed next action");
        await saveState(workDir, state);
        startRound = state.round + 1;
        console.log(`  Resumed at round ${startRound}: ${qaOutput.substring(0, 100)}...\n`);
      } else {
        throw new Error(
          "Checkpoint cannot auto-resume: expected either an interrupted AA answer or a saved AA answer awaiting QA processing",
        );
      }
    } else {
      // ── Step 1: Send skills, wait for ready ──────────────────────────
      console.log("── Sending skills ──");
      const qaInitMsg = await buildQAInitMsg();
      const aaInitMsg = await buildAAInitMsg(ideaContent, ideaTitle, ideaNote);

      ioLog(workDir, "QA", "IN", qaInitMsg.substring(0, 500) + "...[skill content]");
      const qaReady = await sendInitToSession(qaSession, qaInitMsg, "QA", AGENT_TIMEOUT_MS);
      ioLog(workDir, "QA", "OUT", qaReady);
      state.qa_next_entry = requireNextEntry(qaReady, "QA ready");
      console.log(`QA ready: ${qaReady.trim().substring(0, 80)}`);

      ioLog(workDir, "AA", "IN", aaInitMsg.substring(0, 500) + "...[skill content]");
      const aaReady = await sendInitToSession(aaSession, aaInitMsg, "AA", AGENT_TIMEOUT_MS);
      ioLog(workDir, "AA", "OUT", aaReady);
      state.aa_next_entry = requireNextEntry(aaReady, "AA ready");
      const aaInit = parseAAInitMarker(aaReady);
      if (!aaInit) {
        throw new Error("AA init protocol error: missing valid ___AA_INIT_COMPLETE___ canonical paper title");
      }
      state.paper_title = aaInit.paper_title;
      state.paper_subdir = aaInit.paper_subdir;
      await saveState(workDir, state);
      console.log(`AA ready: ${aaReady.trim().substring(0, 80)}\n`);
      console.log(`  Canonical title: ${state.paper_title}`);
      console.log(`  Review: ${reviewPathForTitle(state.paper_title)}\n`);

      // ── Step 2: Trigger QA's blind first-round overview ──────────────
      console.log("── Starting review ──");
      const startSignal = "START_REVIEW\n开始盲评提问。按你的 §2 / §R1 线性步骤输出固定首轮问题。";
      const qaStartInput = `${buildAnchor("Question Agent", state.qa_next_entry, "START_REVIEW", "")}\n${startSignal}`;
      ioLog(workDir, "QA", "IN", qaStartInput);
      qaOutput = await sendProtocolTurn(qaSession, qaStartInput, workDir, "QA", AGENT_TIMEOUT_MS);
      ioLog(workDir, "QA", "OUT", qaOutput);
      state.qa_next_entry = requireNextEntry(qaOutput, "QA round 1");
      console.log(`QA round 1: ${qaOutput.substring(0, 100)}...\n`);

      const firstMarker = parseQAMarker(qaOutput);
      if (firstMarker?.type !== "question"
          || firstMarker.q.round !== 1
          || firstMarker.q.question_category !== "五大类别总览") {
        throw new Error("QA protocol error: first round must be the blind 五大类别总览 question");
      }
      startRound = 1;
    }

    // ── Step 3-5: Main loop ────────────────────────────────────────────
    for (let round = startRound; round <= maxRounds; round++) {
      state.round = round;

      // Check if QA produced a judgment
      const qaMarker = parseQAMarker(qaOutput);
      if (qaMarker?.type === "judgment") {
        state.final_judgment = qaMarker.j;
        console.log(`\n═══ JUDGMENT at round ${round} ═══`);
        console.log(`  relevance=${qaMarker.j.relevance}, depth=${qaMarker.j.depth_value}`);
        break;
      }

      if (!qaMarker || qaMarker.type !== "question") {
        throw new Error("QA protocol error: missing valid ___QA_QUESTION___ payload");
      }
      if (qaMarker.q.round !== round) {
        throw new Error(`QA protocol error: expected question round ${round}, got ${qaMarker.q.round}`);
      }

      // Forward QA question to AA (with anchor)
      console.log(`── Round ${round}: QA → AA ──`);
      const aaRuntimeGuard = state.aa_no_tools
        ? "当前为 isolated recovery：禁止调用任何工具，直接使用既有问答回答；不足写入 information_gaps。"
        : "本轮最多调用 2 次工具；完成第 2 次调用后禁止继续检索，立即输出协议。";
      const aaAnchor = buildAnchor("Answer Agent", state.aa_next_entry, "QA_QUESTION", aaRuntimeGuard);
      const aaPayload = formatQAQuestionPayload(qaMarker.q);
      const aaInput = `${aaAnchor}\n${aaPayload}`;
      const aaTurnInput = repairInterruptedAA && round === startRound
        ? buildProtocolRepairPrompt("aa", aaInput)
        : aaInput;
      repairInterruptedAA = false;
      ioLog(workDir, "AA", "IN", aaTurnInput);
      let aaOutput: string;
      try {
        aaOutput = await sendProtocolTurn(aaSession, aaTurnInput, workDir, "AA", AGENT_TIMEOUT_MS);
      } catch (error) {
        if (!(error instanceof IsolatedAARecoveryRequiredError)) throw error;
        if (state.aa_no_tools) {
          throw new Error(`AA no-tool recovery session cannot be replaced again: ${error.message}`);
        }
        console.warn(`  ${error.message}`);
        console.warn("  Starting a fresh no-tool AA session to answer the interrupted round");
        const recovered = await recoverAAWithFreshNoToolSession(
          aaSession,
          workDir,
          state,
          qaMarker.q,
          maxBudgetUsd,
        );
        aaSession = recovered.session;
        aaOutput = recovered.output;
      }
      ioLog(workDir, "AA", "OUT", aaOutput);
      state.aa_next_entry = requireNextEntry(aaOutput, `AA round ${round}`);
      console.log(`AA: ${aaOutput.substring(0, 100)}...`);

      // Parse AA output
      const aaMarker = parseAAMarker(aaOutput);
      if (!aaMarker) {
        throw new Error(`AA protocol error: ${describeAAProtocolError(aaOutput)}`);
      }
      if (aaMarker.round !== round) {
        throw new Error(`AA protocol error: expected answer round ${round}, got ${aaMarker.round}`);
      }
      // Record Q&A
      state.qa_history.push({ round, question: qaMarker.q.question, answer: aaMarker.answer });
      await saveState(workDir, state);

      // Forward AA answer to QA (with anchor)
      console.log(`── Round ${round}: AA → QA ──`);
      const qaAnchor = buildAnchor("Question Agent", state.qa_next_entry, "AA_OUTPUT",
        `已加载 reference: [${state.qa_loaded_references.join(", ")}]`);
      const qaPayload2 = formatAAOutput(aaMarker);
      const qaInput2 = `${qaAnchor}\n${qaPayload2}`;
      ioLog(workDir, "QA", "IN", qaInput2);
      qaOutput = await sendProtocolTurn(qaSession, qaInput2, workDir, "QA", AGENT_TIMEOUT_MS);
      ioLog(workDir, "QA", "OUT", qaOutput);
      state.qa_next_entry = requireNextEntry(qaOutput, `QA after round ${round}`);
      qaOutput = await resolveQAReferenceRequests(qaSession, workDir, state, round + 1, qaOutput);
      state.qa_next_entry = requireNextEntry(qaOutput, `QA next action after round ${round}`);
      await saveState(workDir, state);
      console.log(`QA: ${qaOutput.substring(0, 100)}...\n`);

      // Check for judgment in this new QA output
      const qaMarker2 = parseQAMarker(qaOutput);
      if (qaMarker2?.type === "judgment") {
        state.final_judgment = qaMarker2.j;
        state.round = round;
        console.log(`\n═══ JUDGMENT at round ${round} ═══`);
        console.log(`  relevance=${qaMarker2.j.relevance}, depth=${qaMarker2.j.depth_value}`);
        break;
      }
    }

    // ── Save final review only after QA has produced a judgment ─────────
    if (state.final_judgment) {
      console.log(`\n── Saving review ──`);
      const outputPath = await writeReview(state);
      console.log(`  Output: ${outputPath}`);
      console.log(`  Rounds: ${state.round}`);
      console.log(`  Judgment: complete`);
    } else {
      console.warn(`\n[WARN] No final judgment after ${state.round} rounds.`);
      console.warn(`  Review not written; resumable run state retained at ${workDir}`);
    }

  } finally {
    console.log(`\n── Closing sessions ──`);
    await saveState(workDir, state);
    closeSession(qaSession);
    closeSession(aaSession);
    console.log(`  QA ${qaSession.sessionId}: closed`);
    console.log(`  AA ${aaSession.sessionId}: closed`);
  }

  console.log(`\n══════════════════════════════════════════════════\n`);
}

export {
  buildAnchor,
  buildSessionArgs,
  buildAAIsolatedRecoveryPrompt,
  buildProtocolRepairPrompt,
  canAutoResumeInterruptedAA,
  canAutoResumeInterruptedQA,
  formatAAInitPayload,
  formatAAOutput,
  formatQAQuestionPayload,
  parseAAInitMarker,
  parseAAMarker,
  parseQAMarker,
  parseFinalJudgmentFromRawJsonl,
  parseLastQAQuestionFromRawJsonl,
  describeAAProtocolError,
  describeLoopSemantics,
  isCompleteProtocolResponse,
  reviewPathForTitle,
  selectSessionOutput,
  shouldUseIsolatedAARecovery,
  summarizeApiRetry,
};

if (require.main === module) {
  main().catch(e => { console.error("FATAL:", e); process.exit(1); });
}
