#!/usr/bin/env -S npx tsx
/**
 * Learning Experiment from Notes — Autonomous Scheduler
 *
 * Multi-agent pipeline with worker pool pattern + checkpoint/resume + progress viz:
 *   Phase 1: Launch 6 Question Agents → build work pool of question entries
 *   Phase 2: 3 worker threads, each picks undone question → Answer Agent
 *   Phase 3: 2 worker threads, each picks undone layer → Horizon Summary Agent
 *   Phase 4: 1 Vertical Summary Agent → final summary.md
 *
 * Usage:
 *   npx tsx /data3/paper_analysis/scripts/learning_scheduler.ts --work-dir <dir> --user-input "..."
 */

import { spawn, ChildProcess, exec } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import { existsSync, createWriteStream } from "fs";

// ─── Configuration ───────────────────────────────────────────────────────────

const VAULT_ROOT = "/data3/paper_analysis";
const SKILLS_DIR = path.join(VAULT_ROOT, ".claude", "skills");
const CLAUDE_CMD = "claude";

const LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6"];

const LAYER_DEFS: Record<string, { name: string; scope: string; questions: string[] }> = {
  L1: {
    name: "算法 Pipeline",
    scope: "模型结构、算法优化（稀疏/量化/蒸馏/压缩）、推理算法流程、多算子并发的算法可行性",
    questions: [
      "Q1.1: <模型负载> 的推理计算流程是什么？（具体到伪代码或张量计算，含计算图和数据依赖）",
      "Q1.2: 算法层面有哪些方法可以加速推理？（稀疏gating、量化、蒸馏、KV-cache压缩、token合并、early-exit等）",
      "Q1.3: 这些加速方法的实现是什么？实验环境（<后端平台>平台、benchmark）是什么？",
      "Q1.4: 在<请求模式>场景下，<计算场景>的算法层面可行性？（哪些算子可并发、数据依赖分析、流水线并行机会）",
    ],
  },
  L2: {
    name: "Serving 调度",
    scope: "单/多请求调度、请求分解为计算单元、Serving框架支持、Dispatcher/SLO设计",
    questions: [
      "Q2.1: <请求模式>下，<模型负载> Serving 如何组织调度？",
      "Q2.2: 有哪些开源 Serving 框架支持这些模型？（vLLM、SGLang、TensorRT-LLM、TGI等）",
      "Q2.3: Serving 层如何将请求分解为可并发执行的计算单元？",
      "Q2.4: 这些 Serving 方法的实现和实验环境？",
    ],
  },
  L3: {
    name: "编译框架",
    scope: "IR表示、算子融合、图优化、自动调优、Codegen、多算子并发的编译支持",
    questions: [
      "Q3.1: <模型负载> 的编译 IR 表示是什么？",
      "Q3.2: 编译框架如何做算子融合和图优化来支持<计算场景>？",
      "Q3.3: 有哪些编译框架支持这些模型的多算子并发？（XLA、TVM、Triton、MLIR等）",
      "Q3.4: 这些编译方法的实现和实验环境？",
    ],
  },
  L4: {
    name: "Kernel 调度",
    scope: "算子实现、Tile切分、微算子并发、Memory Hierarchy、指令流水线、并发kernel",
    questions: [
      "Q4.1: 在<后端平台>上，kernel 如何切分和调度来支持<计算场景>？",
      "Q4.2: <计算场景>的指令编排和 pipeline 设计？",
      "Q4.3: 有哪些 kernel 调度框架/方法支持？",
      "Q4.4: 这些 kernel 调度方法的实现和实验环境？",
    ],
  },
  L5: {
    name: "硬件架构",
    scope: "数据流设计、计算单元、控制模块、访存架构、片上网络",
    questions: [
      "Q5.1: 硬件计算模块如何设计来支持<计算场景>？",
      "Q5.2: 控制模块和访存架构如何支持多算子并发？",
      "Q5.3: 有哪些硬件架构方法/模拟器评估过<模型负载>的<计算场景>？",
      "Q5.4: 这些硬件方法的实现和实验环境？",
    ],
  },
  L6: {
    name: "芯片设计",
    scope: "Chiplet设计、Wafer-Scale、PIM、DRAM/RERAM集成、NoC、芯片级并发支持",
    questions: [
      "Q6.1: 芯片级设计如何影响<模型负载>的推理并发？",
      "Q6.2: 有哪些芯片设计方法/模拟器评估过这类负载？",
      "Q6.3: 这些芯片方法的实现和实验环境？",
    ],
  },
};

const SIDE_EMPHASIS_MAP: Record<string, Record<string, string>> = {
  "硬件体系结构": { label: "硬件体系结构", primary: "硬件数据流 + 计算/控制模块设计 + 架构", secondary: "软件栈/算法" },
  "编译框架":     { label: "编译框架",     primary: "编译优化流程 + IR 转换 + Codegen",     secondary: "Kernel/硬件目标" },
  "Kernel 调度":  { label: "Kernel 调度",  primary: "Kernel 实现 + 指令编排 + Pipeline",   secondary: "编译/硬件" },
  "实验和实现":   { label: "实验和实现",   primary: "实现代码/框架 + 实验环境",            secondary: "创新方法" },
  "方法和创新":   { label: "方法和创新",   primary: "创新方法 + 算法设计",                 secondary: "实验环境" },
  "全栈均衡":     { label: "全栈均衡",     primary: "各内容均衡",                          secondary: "无" },
};

// ─── Types ───────────────────────────────────────────────────────────────────

interface UserInput {
  modelLoad: string;
  backend: string;
  requestMode: string;
  computeScenario: string;
  emphasis: string;
  constraints: string;
}

interface WorkEntry {
  questionId: string;
  layerId: string;
  isDone: boolean;
}

interface Checkpoint {
  phase: number;                        // 0=not started, 1-4
  phase1_done: boolean;
  question_pool: WorkEntry[];
  phase3_layers_done: string[];
  phase4_done: boolean;
  started_at: number;
  updated_at: number;
}

// ─── User Input Parsing ──────────────────────────────────────────────────────

function extractPatterns(text: string, patterns: [RegExp, string][]): string {
  const found = patterns.filter(([pat]) => pat.test(text)).map(([, l]) => l);
  return found.length > 0 ? found.join(", ") : "用户未指定";
}

function detectEmphasis(text: string): string {
  const ordered = Object.keys(SIDE_EMPHASIS_MAP).filter(k => k !== "全栈均衡").sort((a, b) => b.length - a.length);
  for (const label of ordered) { if (text.includes(label.slice(0, 3))) return label; }
  return "全栈均衡";
}

function parseUserInput(text: string): UserInput {
  return {
    modelLoad: extractPatterns(text, [
      [/MoE|mixture.of.experts/i, "MoE"], [/wavelet.?Diffusion|小波扩散/i, "wavelet-Diffusion"],
      [/DiT|Diffusion.Transformer/i, "DiT"], [/多模态|multimodal/i, "多模态"],
      [/Video|视频/i, "Video"], [/LLM|大语言模型|大模型/i, "LLM"],
    ]),
    backend: extractPatterns(text, [
      [/GPU|gpu|单.?GPU/i, "单GPU"], [/NPU|npu|昇腾|Ascend/i, "NPU"], [/加速器|accelerator/i, "加速器"],
    ]),
    requestMode: extractPatterns(text, [
      [/单请求|single.request|单.?batch/i, "单请求"], [/多请求|multi.request/i, "多请求"], [/在线|online|流式|stream/i, "在线"],
    ]),
    computeScenario: extractPatterns(text, [
      [/多算子.*并发|微算子.*并发|operator.concurr|micro.?operator/i, "多算子/微算子并发"],
      [/算子融合|operator.fusion|kernel.fusion/i, "算子融合"],
      [/pipeline.并行|流水线/i, "流水线并行"],
    ]),
    emphasis: detectEmphasis(text),
    constraints: "",
  };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${s % 60}s`;
}

function fillTemplate(template: string, ui: UserInput): string {
  return template
    .replace(/<模型负载>/g, ui.modelLoad)
    .replace(/<后端平台>/g, ui.backend)
    .replace(/<请求模式>/g, ui.requestMode)
    .replace(/<计算场景>/g, ui.computeScenario);
}

function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }

async function checkFileHasSignal(filepath: string, signal: string): Promise<boolean> {
  try { return (await fs.readFile(filepath, "utf-8")).includes(signal); }
  catch { return false; }
}

// ─── Skill Body Reader ───────────────────────────────────────────────────────

function stripFrontmatter(md: string): string {
  if (md.startsWith("---")) { const parts = md.split("---"); return parts.slice(2).join("---").trim(); }
  return md.trim();
}

async function readSkillBody(skillDir: string): Promise<string> {
  const p = path.join(SKILLS_DIR, skillDir, "SKILL.md");
  try { return stripFrontmatter(await fs.readFile(p, "utf-8")); }
  catch { return `# ${skillDir}`; }
}

// ─── Checkpoint / Resume ─────────────────────────────────────────────────────

function checkpointPath(workDir: string): string {
  return path.join(workDir, "progress.json");
}

async function loadCheckpoint(workDir: string): Promise<Checkpoint | null> {
  try {
    const raw = await fs.readFile(checkpointPath(workDir), "utf-8");
    return JSON.parse(raw) as Checkpoint;
  } catch { return null; }
}

async function saveCheckpoint(workDir: string, cp: Checkpoint): Promise<void> {
  cp.updated_at = Date.now();
  await fs.writeFile(checkpointPath(workDir), JSON.stringify(cp, null, 2), "utf-8");
}

// ─── Progress Visualization ──────────────────────────────────────────────────

// ANSI escape helpers
const C = { R: "\x1b[0m", G: "\x1b[32m", Y: "\x1b[33m", C: "\x1b[36m", B: "\x1b[34m", D: "\x1b[2m", W: "\x1b[37m", M: "\x1b[35m" };
const BOLD = "\x1b[1m";

function bar(filled: number, total: number, width: number, color: string, emptyColor: string = C.D): string {
  const blocks = Math.round((filled / Math.max(total, 1)) * width);
  let s = "";
  for (let i = 0; i < width; i++) s += i < blocks ? `${color}█${C.R}` : `${emptyColor}░${C.R}`;
  return s;
}

function pct(n: number, d: number): string {
  return d === 0 ? "  0%" : `${Math.round(n / d * 100)}%`.padStart(4);
}

async function printProgress(workDir: string, ui: UserInput, phaseLabel: string, cp: Checkpoint, startMs: number): Promise<void> {
  const elapsed = Date.now() - startMs;
  const pool = cp.question_pool;
  const totalQ = pool.length;
  const doneQ = pool.filter(e => e.isDone).length;
  const horizonDone = cp.phase3_layers_done.length;
  const runningAnswer = (await execCapture("pgrep -fc 'answer_Q[0-9]' 2>/dev/null")) || "0";
  const runningHorizon = (await execCapture("pgrep -fc 'horizon_[L]' 2>/dev/null")) || "0";

  // Count per-layer
  const layerStats: Record<string, { done: number; total: number }> = {};
  for (const lid of LAYER_ORDER) layerStats[lid] = { done: 0, total: 0 };
  for (const e of pool) {
    layerStats[e.layerId].total++;
    if (e.isDone) layerStats[e.layerId].done++;
  }

  const lines: string[] = [];
  lines.push("");
  lines.push(`${BOLD}${C.C}╔══════════════════════════════════════════════════╗${C.R}`);
  lines.push(`${BOLD}${C.C}║${C.R}  ${BOLD}Scheduler Progress${C.R}  ${C.D}elapsed ${formatDuration(elapsed)}  phase: ${C.Y}${phaseLabel}${C.R}`);
  lines.push(`${BOLD}${C.C}╠══════════════════════════════════════════════════╣${C.R}`);

  // Phase 1
  const p1done = cp.phase1_done ? 6 : LAYER_ORDER.filter(lid => existsSync(path.join(workDir, `${lid}_问题空间.md`))).length;
  lines.push(`${BOLD}${C.C}║${C.R} P1 Question [${bar(p1done, 6, 18, C.G)}] ${p1done}/6  ${cp.phase1_done ? `${C.G}DONE${C.R}` : `${C.Y}...${C.R}`}`);

  // Phase 2
  const p2Label = doneQ === totalQ && totalQ > 0 ? `${C.G}DONE${C.R}` : `${C.Y}RUN${C.R}`;
  lines.push(`${BOLD}${C.C}║${C.R} P2 Answer   [${bar(doneQ, totalQ, 18, C.Y)}] ${String(doneQ).padStart(2)}/${String(totalQ).padEnd(2)} ${pct(doneQ, totalQ)} ${p2Label} ${C.D}(${runningAnswer.trim()} act)${C.R}`);

  // Phase 3
  const p3Label = cp.phase3_layers_done.length === 6 ? `${C.G}DONE${C.R}` : horizonDone > 0 ? `${C.C}RUN${C.R}` : `${C.D}WAIT${C.R}`;
  lines.push(`${BOLD}${C.C}║${C.R} P3 Horizon  [${bar(horizonDone, 6, 18, C.C)}] ${horizonDone}/6  ${p3Label} ${C.D}(${runningHorizon.trim()} act)${C.R}`);

  // Phase 4
  const p4Label = cp.phase4_done ? `${C.G}DONE ✓${C.R}` : cp.phase3_layers_done.length === 6 ? `${C.C}WAIT${C.R}` : `${C.D}WAIT${C.R}`;
  lines.push(`${BOLD}${C.C}║${C.R} P4 Vertical ${p4Label}`);

  // Per-layer detail for Phase 2
  if (totalQ > 0 && doneQ < totalQ) {
    lines.push(`${BOLD}${C.C}╠══════════════════════════════════════════════════╣${C.R}`);
    for (const lid of LAYER_ORDER) {
      const s = layerStats[lid];
      if (s.total === 0) continue;
      const dotBar = Array.from({length: s.total}, (_, i) => i < s.done ? `${C.G}●${C.R}` : `${C.D}○${C.R}`).join("");
      const check = s.done === s.total ? ` ${C.G}✓${C.R}` : "";
      lines.push(`${BOLD}${C.C}║${C.R} ${lid} ${LAYER_DEFS[lid].name.padEnd(10)} [${dotBar}] ${s.done}/${s.total}${check}`);
    }
  }

  // Model info
  const emphasis = SIDE_EMPHASIS_MAP[ui.emphasis] ?? SIDE_EMPHASIS_MAP["全栈均衡"];
  lines.push(`${BOLD}${C.C}╠══════════════════════════════════════════════════╣${C.R}`);
  lines.push(`${BOLD}${C.C}║${C.R} ${C.D}模型:${C.R} ${ui.modelLoad}  ${C.D}后端:${C.R} ${ui.backend}  ${C.D}侧重:${C.R} ${emphasis.label}`);
  lines.push(`${BOLD}${C.C}╚══════════════════════════════════════════════════╝${C.R}`);

  // Clear screen and print
  process.stdout.write("\x1b[2J\x1b[H" + lines.join("\n") + "\n");
}

async function execCapture(cmd: string): Promise<string> {
  return new Promise(resolve => {
    exec(cmd, { timeout: 3000 }, (_err: any, stdout: string) => resolve((stdout || "").trim()));
  });
}

// ─── Claude Subprocess Spawn ─────────────────────────────────────────────────

// Track all spawned child processes for cleanup on SIGINT/SIGTERM/timeout
const activeProcs = new Set<ChildProcess>();

function cleanupAll(): void {
  for (const proc of activeProcs) {
    try { proc.kill("SIGKILL"); } catch { /* already dead */ }
  }
  activeProcs.clear();
}

// Register signal handlers — kill all children before exiting
let cleanupRegistered = false;
function ensureCleanup(): void {
  if (cleanupRegistered) return;
  cleanupRegistered = true;
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => {
      console.log(`\n  [scheduler] ${sig} received, killing ${activeProcs.size} child processes...`);
      cleanupAll();
      process.exit(128 + (sig === "SIGINT" ? 2 : sig === "SIGTERM" ? 15 : 1));
    });
  }
}

function spawnClaude(prompt: string, workDir: string, logPrefix: string): ChildProcess {
  ensureCleanup();
  const logFile = path.join(workDir, `${logPrefix}.log`);
  const cmd = [
    CLAUDE_CMD, "-p", prompt,
    "--output-format", "stream-json",
    "--verbose", "--include-partial-messages",
    "--permission-mode", "acceptEdits",
    "--add-dir", VAULT_ROOT,
  ];
  const proc = spawn(cmd[0], cmd.slice(1), {
    cwd: VAULT_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const logStream = createWriteStream(logFile);
  proc.stdout?.pipe(logStream);
  proc.stderr?.pipe(logStream);

  activeProcs.add(proc);
  proc.on("close", () => activeProcs.delete(proc));
  proc.on("error", () => activeProcs.delete(proc));
  return proc;
}

function waitForProc(proc: ChildProcess, timeoutMs: number): Promise<boolean> {
  return new Promise(resolve => {
    const timer = setTimeout(() => {
      // Kill orphan on timeout
      try { proc.kill("SIGKILL"); } catch { /* already dead */ }
      activeProcs.delete(proc);
      resolve(false);
    }, timeoutMs);
    proc.on("close", () => { clearTimeout(timer); resolve(true); });
  });
}

// ─── Phase 1: Question Agents ────────────────────────────────────────────────

async function buildQuestionPrompt(lid: string, workDir: string, ui: UserInput): Promise<string> {
  const skillBody = await readSkillBody("learning-experiment-from-notes-question");
  const layer = LAYER_DEFS[lid];
  const qs = layer.questions.map((q, i) => `- ${lid[1]}.${i + 1}: ${fillTemplate(q, ui)}`);
  const outFile = path.join(workDir, `${lid}_问题空间.md`);

  return `${skillBody}\n\n---\n\n`
    + `你是 Question Agent。\n`
    + `层编号: ${lid}, 层名称: ${layer.name}, 覆盖: ${layer.scope}\n\n`
    + `用户要素: 模型=${ui.modelLoad}, 后端=${ui.backend}, 请求=${ui.requestMode}, 计算=${ui.computeScenario}, 侧重=${ui.emphasis}\n\n`
    + `该层预置问题:\n${qs.join("\n")}\n\n`
    + `## 输出格式\n`
    + `写入 ${outFile}，每个问题存储格式:\n`
    + `\`\`\`\n`
    + `## <question_id>\n`
    + `**层**: ${lid}\n`
    + `**问题**: <问题文本>\n`
    + `**预关键词**: <逗号分隔>\n`
    + `\`\`\`\n`
    + `所有问题写完后末尾加 [QUESTION_AGENT_DONE] ${lid}`;
}

async function phase1_questions(workDir: string, ui: UserInput, cp: Checkpoint, timeoutMs: number): Promise<WorkEntry[]> {
  console.log(`\n${"=".repeat(60)}\nPhase 1: Spawning 6 Question Agents\n${"=".repeat(60)}\n`);

  // Resume: check which layers already have DONE signal
  const alreadyDone: string[] = [];
  const needSpawn: string[] = [];
  for (const lid of LAYER_ORDER) {
    const f = path.join(workDir, `${lid}_问题空间.md`);
    if (existsSync(f) && await checkFileHasSignal(f, `[QUESTION_AGENT_DONE] ${lid}`)) {
      alreadyDone.push(lid);
      console.log(`  ${lid} 问题空间已存在 DONE 信号，跳过`);
    } else {
      needSpawn.push(lid);
    }
  }

  if (alreadyDone.length > 0) {
    console.log(`  恢复: ${alreadyDone.length}/6 layers already done, ${needSpawn.length} remaining`);
  }

  // Spawn only for unfinished layers
  if (needSpawn.length > 0) {
    const procs: { lid: string; proc: ChildProcess }[] = [];
    for (const lid of needSpawn) {
      const prompt = await buildQuestionPrompt(lid, workDir, ui);
      procs.push({ lid, proc: spawnClaude(prompt, workDir, `question_${lid}`) });
    }

    const start = Date.now();
    while ((Date.now() - start) < timeoutMs) {
      let ready = alreadyDone.length;
      for (const lid of LAYER_ORDER) {
        const f = path.join(workDir, `${lid}_问题空间.md`);
        if (await checkFileHasSignal(f, `[QUESTION_AGENT_DONE] ${lid}`)) ready++;
      }
      cp.phase = 1;
      await printProgress(workDir, ui, "Phase 1", cp, cp.started_at);
      console.log(`  [${formatDuration(Date.now() - start)}] ${ready}/6 question spaces ready`);
      if (ready === 6) break;
      await sleep(15_000);
    }
  }

  // Build work pool from question spaces (NEVER scan answer files for resume —
  // only progress.json is authoritative for completion status)
  const pool: WorkEntry[] = [];
  for (const lid of LAYER_ORDER) {
    const f = path.join(workDir, `${lid}_问题空间.md`);
    if (!existsSync(f)) continue;
    const content = await fs.readFile(f, "utf-8");
    const matches = content.match(/Q\d\.\d+/g);
    if (matches) {
      for (const qid of [...new Set(matches)]) {
        pool.push({ questionId: qid, layerId: lid, isDone: false });
      }
    }
  }

  cp.phase1_done = true;
  cp.question_pool = pool;
  await saveCheckpoint(workDir, cp);

  console.log(`  Work pool: ${pool.length} questions across ${new Set(pool.map(e => e.layerId)).size} layers\n`);
  return pool;
}

// ─── Phase 2: Answer Agents (3 workers) ──────────────────────────────────────

async function buildAnswerPrompt(qid: string, lid: string, workDir: string, ui: UserInput): Promise<string> {
  const skillBody = await readSkillBody("learning-experiment-from-notes-answer");
  const layer = LAYER_DEFS[lid];
  const emphasis = SIDE_EMPHASIS_MAP[ui.emphasis] ?? SIDE_EMPHASIS_MAP["全栈均衡"];
  const qsFile = path.join(workDir, `${lid}_问题空间.md`);
  const outFile = path.join(workDir, `${qid}_${lid}_answer.md`);

  return `${skillBody}\n\n---\n\n`
    + `你是 Answer Agent。模仿 obsidian-keyword-explain 搜索 vault 回答单问题。\n\n`
    + `- 问题 ID: ${qid}\n- 层: ${lid} ${layer.name}\n- 模型: ${ui.modelLoad}\n- 后端: ${ui.backend}\n- 侧重: ${ui.emphasis}\n\n`
    + `## 输入\n`
    + `问题空间: ${qsFile}\n`
    + `输出文件: ${outFile}\n\n`
    + `## 侧重配置\n${JSON.stringify(emphasis, null, 2)}\n\n`
    + `## 流程\n`
    + `1. 读取 ${qsFile} 获取 ${qid} 的问题文本和预关键词\n`
    + `2. 对关键词进行语义分割（仿 obsidian-keyword-explain Step 1）\n`
    + `3. 四目录搜索（paper_secs/knowledge_notes/experiment_notes/idea_notes）\n`
    + `4. 去重后通过 obsidian_get_note 读取笔记\n`
    + `5. 按侧重组织答案，每个方法含: 笔记证据 + 是什么 + 方法细节 + 实现 + 实验环境\n`
    + `6. 末尾标注所用上下文的 vault path 来源\n`
    + `7. 写入 ${outFile}\n`
    + `8. 末尾输出 [ANSWER_AGENT_DONE] ${qid}`;
}

async function phase2_answers(workDir: string, ui: UserInput, pool: WorkEntry[], cp: Checkpoint, timeoutPerWorker: number): Promise<void> {
  const remaining = pool.filter(e => !e.isDone).length;
  console.log(`\n${"=".repeat(60)}\nPhase 2: Answer Agents (3 workers, ${remaining}/${pool.length} remaining)\n${"=".repeat(60)}\n`);

  if (remaining === 0) {
    console.log("  All questions already answered (resume). Skipping Phase 2.\n");
    return;
  }

  const NUM_WORKERS = 3;
  let progressTimer: ReturnType<typeof setInterval> | null = null;

  // Periodic progress display
  progressTimer = setInterval(async () => {
    await printProgress(workDir, ui, "Phase 2", cp, cp.started_at);
  }, 30_000);

  async function worker(workerId: number) {
    while (true) {
      const idx = pool.findIndex(e => !e.isDone);
      if (idx === -1) break;

      const entry = pool[idx];
      entry.isDone = true; // Claim it (saved to checkpoint below before spawn)
      cp.phase = 2;
      await saveCheckpoint(workDir, cp);

      const af = path.join(workDir, `${entry.questionId}_${entry.layerId}_answer.md`);
      const prompt = await buildAnswerPrompt(entry.questionId, entry.layerId, workDir, ui);
      const proc = spawnClaude(prompt, workDir, `answer_${entry.questionId}_${entry.layerId}`);
      const exitOk = await waitForProc(proc, timeoutPerWorker);

      // Verify: process must exit cleanly AND output must contain DONE signal
      const verified = exitOk && existsSync(af) && await checkFileHasSignal(af, `[ANSWER_AGENT_DONE] ${entry.questionId}`);
      if (!verified) {
        entry.isDone = false; // Undo claim — will be retried
        console.log(`  [W${workerId}] ✗ ${entry.questionId} (${entry.layerId}) — agent failed or no DONE signal, will retry`);
      } else {
        await saveCheckpoint(workDir, cp);
      }

      const left = pool.filter(e => !e.isDone).length;
      const status = verified ? "✓" : "↻";
      console.log(`  [W${workerId}] ${status} ${entry.questionId} (${entry.layerId}) | remaining: ${left}`);

      await printProgress(workDir, ui, "Phase 2", cp, cp.started_at);
    }
  }

  const workers = [];
  for (let i = 0; i < NUM_WORKERS; i++) workers.push(worker(i));
  await Promise.all(workers);

  if (progressTimer) clearInterval(progressTimer);

  const done = pool.filter(e => e.isDone).length;
  console.log(`  Phase 2 done: ${done}/${pool.length} answers\n`);
}

// ─── Phase 3: Horizon Summary Agents (2 workers) ─────────────────────────────

async function buildHorizonSummaryPrompt(lid: string, workDir: string, ui: UserInput): Promise<string> {
  const skillBody = await readSkillBody("learning-experiment-from-notes-horizon");
  const emphasis = SIDE_EMPHASIS_MAP[ui.emphasis] ?? SIDE_EMPHASIS_MAP["全栈均衡"];
  const layer = LAYER_DEFS[lid];
  const qsFile = path.join(workDir, `${lid}_问题空间.md`);
  const outFile = path.join(workDir, `${lid}_horizon_summary.md`);

  const answers = (await fs.readdir(workDir))
    .filter(f => f.startsWith("Q") && f.includes(`_${lid}_answer.md`))
    .map(f => path.join(workDir, f));

  return `${skillBody}\n\n---\n\n`
    + `你是 Horizon Summary Agent。负责单层 ${lid} ${layer.name} 的分类总结。\n\n`
    + `## 输入\n`
    + `问题空间: ${qsFile}\n`
    + `答案文件 (${answers.length} 个):\n${answers.map(f => `- ${f}`).join("\n")}\n\n`
    + `## 输出\n${outFile}\n\n`
    + `## 流程\n`
    + `1. 读取问题空间和所有答案文件\n`
    + `2. 按实验环境/方法类别对该层所有方法进行分类\n`
    + `3. 输出分类表 + 方法摘要\n`
    + `4. 末尾输出 [HORIZON_SUMMARY_DONE] ${lid}`;
}

async function phase3_horizon(workDir: string, ui: UserInput, cp: Checkpoint, timeoutPerWorker: number): Promise<void> {
  console.log(`\n${"=".repeat(60)}\nPhase 3: Horizon Summary Agents (2 workers, 6 layers)\n${"=".repeat(60)}\n`);

  // Determine which layers still need horizon summary (ONLY from checkpoint, not file scan)
  cp.phase3_layers_done = [...new Set(cp.phase3_layers_done)];
  const layersTodo: string[] = [];
  for (const lid of LAYER_ORDER) {
    if (!cp.phase3_layers_done.includes(lid)) {
      layersTodo.push(lid);
    }
  }
  if (cp.phase3_layers_done.length > 0) {
    console.log(`  恢复: ${cp.phase3_layers_done.length}/6 horizon summaries in checkpoint, ${layersTodo.length} remaining`);
  }

  if (layersTodo.length === 0) {
    console.log("  All horizon summaries already done (resume). Skipping Phase 3.\n");
    return;
  }

  const NUM_WORKERS = 2;

  let progressTimer = setInterval(async () => {
    await printProgress(workDir, ui, "Phase 3", cp, cp.started_at);
  }, 30_000);

  async function worker(workerId: number) {
    while (layersTodo.length > 0) {
      const lid = layersTodo.shift()!;
      const prompt = await buildHorizonSummaryPrompt(lid, workDir, ui);
      const proc = spawnClaude(prompt, workDir, `horizon_${lid}`);
      await waitForProc(proc, timeoutPerWorker);

      const hf = path.join(workDir, `${lid}_horizon_summary.md`);
      const verified = existsSync(hf) && await checkFileHasSignal(hf, `[HORIZON_SUMMARY_DONE] ${lid}`);
      if (verified) {
        cp.phase3_layers_done.push(lid);
        cp.phase3_layers_done = [...new Set(cp.phase3_layers_done)];
        cp.phase = 3;
        await saveCheckpoint(workDir, cp);
      }

      const status = verified ? "✓" : "✗";
      console.log(`  [W${workerId}] ${status} horizon ${lid} (${LAYER_DEFS[lid].name}) | remaining: ${layersTodo.length}`);
      await printProgress(workDir, ui, "Phase 3", cp, cp.started_at);
    }
  }

  const workers = [];
  for (let i = 0; i < NUM_WORKERS; i++) workers.push(worker(i));
  await Promise.all(workers);

  clearInterval(progressTimer);
  console.log(`  Phase 3 done: ${cp.phase3_layers_done.length}/6 horizon summaries\n`);
}

// ─── Phase 4: Vertical Summary Agent ─────────────────────────────────────────

async function buildVerticalSummaryPrompt(workDir: string, ui: UserInput): Promise<string> {
  const skillBody = await readSkillBody("learning-experiment-from-notes-vertical");
  const emphasis = SIDE_EMPHASIS_MAP[ui.emphasis] ?? SIDE_EMPHASIS_MAP["全栈均衡"];

  const horizonFiles = LAYER_ORDER
    .map(lid => path.join(workDir, `${lid}_horizon_summary.md`))
    .filter(p => existsSync(p));

  const outFile = path.join(workDir, "summary.md");

  return `${skillBody}\n\n---\n\n`
    + `你是 Vertical Summary Agent。负责跨层垂向梳理。\n\n`
    + `## 输入 (${horizonFiles.length} 个 horizon summary):\n`
    + `${horizonFiles.map(f => `- ${f}`).join("\n")}\n\n`
    + `## 输出\n${outFile}\n\n`
    + `## 流程\n`
    + `1. 读取所有 horizon summary 文件\n`
    + `2. 识别跨层 (L1→L2→L3→L4→L5→L6) 可串联的垂向方法组合\n`
    + `3. 输出: 侧重声明 + 全栈关系图(mermaid) + 垂向组合逐层分析 + 端到端数据流 + 方法总结表 + 学习路径 + 证据索引\n`
    + `4. 末尾输出 [VERTICAL_SUMMARY_DONE]`;
}

async function phase4_vertical(workDir: string, ui: UserInput, cp: Checkpoint, timeoutMs: number): Promise<void> {
  console.log(`\n${"=".repeat(60)}\nPhase 4: Vertical Summary Agent\n${"=".repeat(60)}\n`);

  const vf = path.join(workDir, "summary.md");

  // Only trust checkpoint for resume — never scan the file
  if (cp.phase4_done) {
    console.log("  Phase 4 already done (checkpoint). Skipping.\n");
    const content = existsSync(vf) ? await fs.readFile(vf, "utf-8") : "(summary file missing)";
    console.log(`\n${"=".repeat(60)}\nFinal Report (${Math.round(content.length / 1024)}KB)\n${"=".repeat(60)}\n`);
    console.log(content);
    return;
  }

  const prompt = await buildVerticalSummaryPrompt(workDir, ui);
  const proc = spawnClaude(prompt, workDir, "vertical_summary");
  const ok = await waitForProc(proc, timeoutMs);

  if (ok && existsSync(vf) && await checkFileHasSignal(vf, "[VERTICAL_SUMMARY_DONE]")) {
    cp.phase4_done = true;
    cp.phase = 4;
    await saveCheckpoint(workDir, cp);
    const content = await fs.readFile(vf, "utf-8");
    console.log(`  Phase 4 done: ✓`);
    console.log(`\n${"=".repeat(60)}\nFinal Report (${Math.round(content.length / 1024)}KB)\n${"=".repeat(60)}\n`);
    console.log(content);
  } else {
    console.log(`  Phase 4 done: ✗`);
  }
}

// ─── Main Pipeline ───────────────────────────────────────────────────────────

async function run(workDir: string, userInputText: string): Promise<number> {
  const totalStart = Date.now();
  const ui = parseUserInput(userInputText);

  await fs.mkdir(workDir, { recursive: true });
  const plan = {
    work_dir: workDir,
    user_input: { model_load: ui.modelLoad, backend: ui.backend, request_mode: ui.requestMode, compute_scenario: ui.computeScenario, emphasis: ui.emphasis, raw: userInputText },
  };
  await fs.writeFile(path.join(workDir, "dispatch.json"), JSON.stringify(plan, null, 2), "utf-8");

  console.log(`\nWork dir: ${workDir}`);
  console.log(`Model: ${ui.modelLoad} | Backend: ${ui.backend} | Request: ${ui.requestMode} | Compute: ${ui.computeScenario} | Emphasis: ${ui.emphasis}`);

  // ── Load or create checkpoint ──
  let cp = await loadCheckpoint(workDir);
  if (cp) {
    console.log(`\n${C.Y}═══ 检测到 checkpoint，恢复中... ═══${C.R}`);
    console.log(`  上次运行到 Phase ${cp.phase}, ${cp.question_pool?.filter(e => e.isDone).length ?? 0}/${cp.question_pool?.length ?? 0} answers done`);
  } else {
    cp = {
      phase: 0,
      phase1_done: false,
      question_pool: [],
      phase3_layers_done: [],
      phase4_done: false,
      started_at: Date.now(),
      updated_at: Date.now(),
    };
    await saveCheckpoint(workDir, cp);
  }

  // Phase 1: Question Agents → build work pool
  if (!cp.phase1_done) {
    const pool = await phase1_questions(workDir, ui, cp, 7_200_000);
    if (pool.length === 0) { console.error("Phase 1 failed: no question spaces"); return 1; }
  } else {
    console.log("Phase 1 already done (checkpoint). Skipping to Phase 2.");
  }

  // Phase 2: Answer Agents (3 workers)
  const undoneAnswers = cp.question_pool.filter(e => !e.isDone).length;
  if (undoneAnswers > 0) {
    await phase2_answers(workDir, ui, cp.question_pool, cp, 3_600_000 * 2);
  } else {
    console.log("Phase 2 already done (checkpoint). Skipping to Phase 3.");
  }

  // Phase 3: Horizon Summary (2 workers)
  const undoneHorizons = LAYER_ORDER.filter(lid => !cp.phase3_layers_done.includes(lid)).length;
  if (undoneHorizons > 0 || cp.phase3_layers_done.length < 6) {
    await phase3_horizon(workDir, ui, cp, 1_800_000);
  } else {
    console.log("Phase 3 already done (checkpoint). Skipping to Phase 4.");
  }

  // Phase 4: Vertical Summary
  if (!cp.phase4_done) {
    await phase4_vertical(workDir, ui, cp, 1_800_000);
  } else {
    console.log("Phase 4 already done (checkpoint). Pipeline complete!");
    const vf = path.join(workDir, "summary.md");
    if (existsSync(vf)) {
      const content = await fs.readFile(vf, "utf-8");
      console.log(`\n${"=".repeat(60)}\nFinal Report (${Math.round(content.length / 1024)}KB)\n${"=".repeat(60)}\n`);
      console.log(content);
    }
  }

  console.log(`\nTotal: ${formatDuration(Date.now() - totalStart)}`);
  return 0;
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

const DEFAULT_ROOT = "/data3/paper_analysis/learning_outputs";

function parseArgs() {
  const args = process.argv.slice(2);
  const parsed: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      const key = args[i].slice(2);
      parsed[key] = args[i + 1] && !args[i + 1].startsWith("--") ? args[++i] : "true";
    }
  }
  return { workDir: parsed["work-dir"] || "", userInput: parsed["user-input"] || "" };
}

/** Generate a human-readable subdirectory name from timestamp + input keywords */
function makeSubdirName(ui: UserInput): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const parts: string[] = [ts];
  const tok = (s: string) => s.replace(/[\/, ]+/g, "").slice(0, 20);
  if (ui.modelLoad && ui.modelLoad !== "用户未指定") parts.push(tok(ui.modelLoad));
  if (ui.backend && ui.backend !== "用户未指定") parts.push(tok(ui.backend));
  if (ui.computeScenario && ui.computeScenario !== "用户未指定") parts.push(tok(ui.computeScenario));
  if (ui.emphasis && ui.emphasis !== "全栈均衡") parts.push(tok(ui.emphasis));
  return parts.join("__");
}

async function main() {
  const opts = parseArgs();

  if (!opts.userInput) {
    if (!process.stdin.isTTY) {
      const chunks: Buffer[] = [];
      for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
      opts.userInput = Buffer.concat(chunks).toString().trim();
    }
    // Fallback: read from existing dispatch.json (resume without re-entering input)
    if (!opts.userInput && opts.workDir) {
      const dispatchFile = path.join(opts.workDir, "dispatch.json");
      if (existsSync(dispatchFile)) {
        try {
          const dispatch = JSON.parse(await fs.readFile(dispatchFile, "utf-8"));
          // New format has raw; old format has parsed fields — reconstruct
          opts.userInput = dispatch?.user_input?.raw ?? "";
          if (!opts.userInput && dispatch?.user_input) {
            const u = dispatch.user_input;
            opts.userInput = [u.model_load, u.backend, u.request_mode, u.compute_scenario, u.emphasis].filter(Boolean).join(" ");
          }
          if (opts.userInput) console.log(`  从 dispatch.json 恢复 user-input`);
        } catch { /* ignore */ }
      }
    }
    if (!opts.userInput) { console.error("Error: --user-input required (no dispatch.json to recover from)"); process.exit(1); }
  }

  // Resolve actual work directory:
  //   - If --work-dir points to an existing run (has dispatch.json), use it directly (resume)
  //   - Otherwise, auto-create a timestamped subdirectory under root (or --work-dir if given)
  let actualWorkDir: string;
  const rootOrGiven = opts.workDir || DEFAULT_ROOT;
  if (existsSync(path.join(rootOrGiven, "dispatch.json"))) {
    actualWorkDir = rootOrGiven;
    console.log(`  恢复已存在的 run: ${actualWorkDir}`);
  } else {
    const ui = parseUserInput(opts.userInput);
    const subdir = makeSubdirName(ui);
    actualWorkDir = path.join(rootOrGiven, subdir);
  }

  process.exit(await run(actualWorkDir, opts.userInput));
}

main().catch(err => { console.error("Scheduler crashed:", err); process.exit(1); });
