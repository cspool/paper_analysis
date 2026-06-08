#!/usr/bin/env -S npx tsx
/**
 * Idea Review Orchestrator — Pure Dual-Session Message Broker
 *
 * Spawns TWO persistent Claude sessions (Question Agent + Answer Agent),
 * routes messages between them via stdin/stdout pipes.
 * All behavior is defined in the skill files — this script only:
 *   1. Starts sessions with skill content
 *   2. Waits for ready signals
 *   3. Forwards messages between sessions
 *   4. Detects completion marker
 *   5. Logs everything with timestamps
 *
 * Message markers (defined in skills):
 *   QA → ___QA_QUESTION___ or ___JUDGMENT_COMPLETE___
 *   AA → ___AA_OUTPUT_START___ ... ___AA_OUTPUT_END___
 *
 * Usage:
 *   npx tsx scripts/idea_review_orchestrator.ts --idea-note <path> [--max-rounds N] [--resume]
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
const REVIEW_SCHEDULING_DIR = path.join(VAULT_ROOT, "review_scheduling_timeline");
const CLAUDE_CMD = "claude";

const QA_SKILL_PATH = path.join(SKILLS_DIR, "idea_question", "SKILL.md");
const AA_SKILL_PATH = path.join(SKILLS_DIR, "idea_answer", "SKILL.md");

const DEFAULT_MAX_ROUNDS = 8;
const AGENT_TIMEOUT_MS = 600_000;

// ─── Types ──────────────────────────────────────────────────────────────────

interface Session {
  proc: ChildProcess;
  stdoutLines: readline.Interface;
  stdin: any;
  sessionId: string;
  label: string;
  pendingResolve: ((text: string) => void) | null;
  pendingReject: ((err: Error) => void) | null;
  accumulatedText: string[];
  logStream: NodeJS.WritableStream;
}

interface ConversationState {
  idea_note_path: string;
  idea_note_title: string;
  qa_session_id: string;
  aa_session_id: string;
  round: number;
  qa_history: { round: number; question: string; answer: string }[];
  final_judgment: any;
  started_at: number;
  updated_at: number;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function now(): number { return Math.floor(Date.now() / 1000); }
function ts(): string { return new Date().toISOString(); }

async function ensureDir(dir: string): Promise<void> {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

// ─── I/O Logging ────────────────────────────────────────────────────────────

function ioLog(logDir: string, label: string, direction: "IN" | "OUT", content: string): void {
  const entry = `[${ts()}] ${label} ${direction === "IN" ? "◄ IN" : "► OUT"}\n${content}\n---\n`;
  fs.appendFile(path.join(logDir, "io_log.txt"), entry).catch(() => {});
}

// ─── Session Management ─────────────────────────────────────────────────────

function spawnSession(sessionId: string, workDir: string, label: string): Session {
  const logPath = path.join(workDir, `${label}_raw.jsonl`);
  const logStream = createWriteStream(logPath, { flags: "a" });

  const proc = spawn(CLAUDE_CMD, [
    "--print", "--verbose",
    "--input-format", "stream-json",
    "--output-format", "stream-json",
    "--session-id", sessionId,
    "--permission-mode", "bypassPermissions",
    "--add-dir", VAULT_ROOT,
    "--max-budget-usd", "5",
  ], { cwd: VAULT_ROOT, stdio: ["pipe", "pipe", "pipe"] });

  const errLog = createWriteStream(path.join(workDir, `${label}_stderr.log`), { flags: "a" });
  proc.stderr?.pipe(errLog);

  const stdoutLines = readline.createInterface({ input: proc.stdout! });

  const session: Session = {
    proc, stdoutLines, stdin: proc.stdin!, sessionId, label,
    pendingResolve: null, pendingReject: null, accumulatedText: [], logStream,
  };

  stdoutLines.on("line", (line: string) => {
    logStream.write(line + "\n");
    try {
      const data = JSON.parse(line);
      if (data.type === "assistant") {
        for (const c of data.message?.content || []) {
          if (c.type === "text") session.accumulatedText.push(c.text);
        }
      } else if (data.type === "result") {
        if (session.pendingResolve) {
          const text = session.accumulatedText.join("");
          session.accumulatedText = [];
          const resolve = session.pendingResolve;
          session.pendingResolve = null; session.pendingReject = null;
          resolve(text);
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

  return session;
}

function sendToSession(session: Session, text: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      session.pendingResolve = null; session.pendingReject = null;
      reject(new Error(`${session.label} timeout after ${timeoutMs}ms`));
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

function closeSession(s: Session): void {
  try { s.stdin.end(); } catch { /* */ }
  try { s.stdoutLines.close(); } catch { /* */ }
  try { s.logStream.end(); } catch { /* */ }
  setTimeout(() => { try { s.proc.kill("SIGTERM"); } catch { /* */ } }, 3000);
}

// ─── Marker Parsing ─────────────────────────────────────────────────────────

function parseQAMarker(text: string): { type: "question"; q: any } | { type: "judgment"; j: any } | null {
  // Check for judgment (higher priority)
  const jMatch = text.match(/___JUDGMENT_COMPLETE___\s*([\s\S]*?)\s*___JUDGMENT_COMPLETE_END___/);
  if (jMatch) {
    try { return { type: "judgment", j: JSON.parse(jMatch[1].trim()) }; }
    catch { /* parse error */ }
  }
  // Check for question
  const qMatch = text.match(/___QA_QUESTION___\s*([\s\S]*?)\s*___QA_QUESTION_END___/);
  if (qMatch) {
    try { return { type: "question", q: JSON.parse(qMatch[1].trim()) }; }
    catch { /* parse error */ }
  }
  return null;
}

function parseAAMarker(text: string): { answer: string; sources: string[]; gaps: string[] } | null {
  const match = text.match(/___AA_OUTPUT_START___\s*([\s\S]*?)\s*___AA_OUTPUT_END___/);
  if (match) {
    try {
      const parsed = JSON.parse(match[1].trim());
      return { answer: parsed.answer || "", sources: parsed.sources || [], gaps: parsed.information_gaps || [] };
    } catch { /* parse error */ }
  }
  return null;
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

// ─── Output ─────────────────────────────────────────────────────────────────

async function writeReview(workDir: string, state: ConversationState): Promise<string> {
  const title = state.idea_note_title;
  const outDir = path.join(REVIEW_SCHEDULING_DIR, title);
  await ensureDir(outDir);
  const outputPath = path.join(outDir, `${title}_review.md`);
  let md = `# ${title}\n\n> ${ts()}\n> Rounds: ${state.round}\n> QA: \`${state.qa_session_id}\`  AA: \`${state.aa_session_id}\`\n\n`;

  if (state.final_judgment) {
    const j = state.final_judgment;
    md += `## 评判\n\n| 维度 | 评定 | 理由 |\n|------|------|------|\n`;
    md += `| 相关性 | ${j.relevance} | ${j.relevance_reason} |\n`;
    md += `| 参考价值 | ${j.reference_value} | ${j.reference_reason} |\n`;
    md += `| 深入价值 | ${j.depth_value} | ${j.depth_reason} |\n\n**总结**：${j.summary}\n\n`;
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
  let ideaNote = "", workDir = "", maxRounds = DEFAULT_MAX_ROUNDS, resume = false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--idea-note" && i + 1 < a.length) ideaNote = a[++i];
    else if (a[i] === "--work-dir" && i + 1 < a.length) workDir = a[++i];
    else if (a[i] === "--max-rounds" && i + 1 < a.length) maxRounds = parseInt(a[++i], 10);
    else if (a[i] === "--resume") resume = true;
  }
  if (!ideaNote) { console.error("ERROR: --idea-note <path> required"); process.exit(1); }
  if (!workDir) workDir = path.join(REVIEW_SCHEDULING_DIR, path.basename(ideaNote, path.extname(ideaNote)));
  return { ideaNote, workDir, maxRounds, resume };
}

// ─── Init Messages ──────────────────────────────────────────────────────────

async function buildInitMsg(agentRole: "qa" | "aa", ideaNoteContent: string, ideaNoteTitle: string, ideaNotePath: string): Promise<string> {
  const skillPath = agentRole === "qa" ? QA_SKILL_PATH : AA_SKILL_PATH;
  const skillContent = await fs.readFile(skillPath, "utf-8");

  return `${skillContent}

---
## 当前 Idea Note

**标题**：${ideaNoteTitle}
**路径**：${ideaNotePath}

**内容**：
${ideaNoteContent.substring(0, 5000)}

---

请确认你已理解以上内容。若你是 Question Agent，回复 "Question Agent 就绪，等待输入。"；若你是 Answer Agent，回复 "Answer Agent 就绪，等待输入。" 之后静默等待后续消息。`;
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const { ideaNote, workDir, maxRounds, resume } = parseArgs();

  console.log("══════════════════════════════════════════════════");
  console.log("  Idea Review Orchestrator — Dual-Session Broker");
  console.log("══════════════════════════════════════════════════");
  console.log(`  ${ideaNote}  →  ${workDir}  (max ${maxRounds} rounds${resume ? ", resume" : ""})`);
  console.log("══════════════════════════════════════════════════\n");

  await ensureDir(workDir);

  // ── Load or init state ────────────────────────────────────────────────
  let state: ConversationState;
  if (resume) {
    const loaded = await loadState(workDir);
    if (!loaded) { console.error("No checkpoint found"); process.exit(1); }
    state = loaded;
  } else {
    state = {
      idea_note_path: ideaNote,
      idea_note_title: path.basename(ideaNote, path.extname(ideaNote)),
      qa_session_id: crypto.randomUUID(), aa_session_id: crypto.randomUUID(),
      round: 0, qa_history: [], final_judgment: null,
      started_at: now(), updated_at: now(),
    };
    await saveState(workDir, state);
  }

  // ── Read idea note ────────────────────────────────────────────────────
  const ideaContent = await fs.readFile(ideaNote, "utf-8");
  const ideaTitle = state.idea_note_title;

  // ── Init sessions ─────────────────────────────────────────────────────
  console.log("── Starting sessions ──\n");

  const qaSession = spawnSession(state.qa_session_id, workDir, "QA");
  const aaSession = spawnSession(state.aa_session_id, workDir, "AA");
  console.log(`QA: ${qaSession.sessionId} (pid ${qaSession.proc.pid})`);
  console.log(`AA: ${aaSession.sessionId} (pid ${aaSession.proc.pid})\n`);

  try {
    // ── Step 1: Send skills, wait for ready ────────────────────────────
    console.log("── Sending skills ──");
    const qaInitMsg = await buildInitMsg("qa", ideaContent, ideaTitle, ideaNote);
    const aaInitMsg = await buildInitMsg("aa", ideaContent, ideaTitle, ideaNote);

    ioLog(workDir, "QA", "IN", qaInitMsg.substring(0, 500) + "...[skill content]");
    const qaReady = await sendToSession(qaSession, qaInitMsg, 120_000);
    ioLog(workDir, "QA", "OUT", qaReady);
    console.log(`QA ready: ${qaReady.trim().substring(0, 80)}`);

    ioLog(workDir, "AA", "IN", aaInitMsg.substring(0, 500) + "...[skill content]");
    const aaReady = await sendToSession(aaSession, aaInitMsg, 120_000);
    ioLog(workDir, "AA", "OUT", aaReady);
    console.log(`AA ready: ${aaReady.trim().substring(0, 80)}\n`);

    // ── Step 2: Trigger QA with "开始提问" ──────────────────────────────
    console.log("── Starting review ──");
    ioLog(workDir, "QA", "IN", "开始提问");
    let qaOutput = await sendToSession(qaSession, "开始提问", AGENT_TIMEOUT_MS);
    ioLog(workDir, "QA", "OUT", qaOutput);
    console.log(`QA round 1: ${qaOutput.substring(0, 100)}...\n`);

    // ── Step 3-5: Main loop ────────────────────────────────────────────
    for (let round = 1; round <= maxRounds; round++) {
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
        console.error(`[WARN] QA did not produce ___QA_QUESTION___ marker, attempting to continue...`);
        // Fallback: treat entire output as a question, forward to AA anyway
      }

      // Forward QA output → AA
      console.log(`── Round ${round}: QA → AA ──`);
      ioLog(workDir, "AA", "IN", qaOutput);
      const aaOutput = await sendToSession(aaSession, qaOutput, AGENT_TIMEOUT_MS);
      ioLog(workDir, "AA", "OUT", aaOutput);
      console.log(`AA: ${aaOutput.substring(0, 100)}...`);

      // Parse AA output
      const aaMarker = parseAAMarker(aaOutput);
      if (!aaMarker) {
        console.error(`[WARN] AA did not produce ___AA_OUTPUT___ marker`);
      }

      // Record Q&A
      const qText = qaMarker?.type === "question" ? qaOutput.replace(/___QA_QUESTION___[\s\S]*$/, "").trim() : qaOutput;
      const aText = aaMarker?.answer || aaOutput;
      state.qa_history.push({ round, question: qText, answer: aText });
      await saveState(workDir, state);

      // Forward AA output → QA
      console.log(`── Round ${round}: AA → QA ──`);
      ioLog(workDir, "QA", "IN", aaOutput);
      qaOutput = await sendToSession(qaSession, aaOutput, AGENT_TIMEOUT_MS);
      ioLog(workDir, "QA", "OUT", qaOutput);
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

    // ── Save final review ───────────────────────────────────────────────
    console.log(`\n── Saving review ──`);
    const outputPath = await writeReview(workDir, state);
    console.log(`  Output: ${outputPath}`);
    console.log(`  Rounds: ${state.round}`);
    console.log(`  Judgment: ${state.final_judgment ? "complete" : "none"}`);

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

main().catch(e => { console.error("FATAL:", e); process.exit(1); });
