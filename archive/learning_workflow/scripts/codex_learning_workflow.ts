#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { runAnchorStage } from "./codex_learning_workflow/anchor_stage.ts";
import {
  CanonicalStore,
  createInitialState,
} from "./codex_learning_workflow/canonical_store.ts";
import { runDirectionStage } from "./codex_learning_workflow/direction_stage.ts";
import { runDoctor } from "./codex_learning_workflow/doctor.ts";
import { renderRun } from "./codex_learning_workflow/renderer.ts";
import { ROLE_EFFORT } from "./codex_learning_workflow/role_profiles.ts";
import { RuntimeManager } from "./codex_learning_workflow/runtime_manager.ts";
import type { RunConfig, RunState } from "./codex_learning_workflow/types.ts";
import { validateRunState } from "./codex_learning_workflow/validators.ts";

interface ParsedArgs {
  command: string;
  options: Map<string, string[]>;
  positionals: string[];
}

function parseArgs(argv: string[]): ParsedArgs {
  const command = argv[0] ?? "help";
  const options = new Map<string, string[]>();
  const positionals: string[] = [];
  for (let index = 1; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      positionals.push(token);
      continue;
    }
    const equals = token.indexOf("=");
    const key = equals >= 0 ? token.slice(2, equals) : token.slice(2);
    let value = equals >= 0 ? token.slice(equals + 1) : "true";
    if (equals < 0 && argv[index + 1] && !argv[index + 1].startsWith("--")) {
      value = argv[index + 1];
      index += 1;
    }
    const existing = options.get(key) ?? [];
    existing.push(value);
    options.set(key, existing);
  }
  return { command, options, positionals };
}

function option(args: ParsedArgs, name: string, fallback?: string): string | undefined {
  return args.options.get(name)?.at(-1) ?? fallback;
}

function numberOption(
  args: ParsedArgs,
  name: string,
  fallback: number,
  minimum = 1,
): number {
  const raw = option(args, name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`--${name} must be an integer >= ${minimum}`);
  }
  return value;
}

function projectRoot(): string {
  return path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
}

function runIdNow(): string {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function defaultWorkDir(root: string): string {
  return path.join(root, "learning_outputs_codex", runIdNow());
}

function codexSourceRoot(): string {
  return process.env.CODEX_HOME
    ? path.resolve(process.env.CODEX_HOME)
    : path.join(os.homedir(), ".codex");
}

function buildConfig(args: ParsedArgs, topic: string): RunConfig {
  const root = projectRoot();
  const sourceRoot = codexSourceRoot();
  const model = option(args, "model", "gpt-5.6-sol");
  if (model !== "gpt-5.6-sol") {
    throw new Error("this workflow intentionally permits only --model gpt-5.6-sol");
  }
  return {
    protocolVersion: 1,
    topic,
    constraints: args.options.get("constraint") ?? [
      "最终产出是可进一步实验探索的方向或参考",
      "尽可能同时提高吞吐并降低延迟",
    ],
    model: "gpt-5.6-sol",
    roleEffort: { ...ROLE_EFFORT },
    protocolRepairEffort: "low",
    maxAnchors: numberOption(args, "max-anchors", 30),
    noNewAnchorStop: numberOption(args, "no-new-anchor-stop", 2),
    maxStage1Rounds: numberOption(args, "max-stage1-rounds", 6),
    maxStage1Tasks: numberOption(args, "max-stage1-tasks", 24),
    evidenceTasksPerRound: numberOption(args, "evidence-tasks-per-round", 4),
    anchorEvidenceConcurrency: numberOption(args, "anchor-evidence-concurrency", 3),
    curatorConcurrency: numberOption(args, "curator-concurrency", 1),
    maxDirectionsPerAnchor: numberOption(args, "max-directions-per-anchor", 2),
    maxPlannerTurnsPerAnchor: numberOption(args, "max-planner-turns", 5),
    maxReviewRoundsPerDirection: numberOption(args, "max-review-rounds", 12),
    directionConcurrency: numberOption(args, "direction-concurrency", 2),
    maxTotalTurns: numberOption(args, "max-total-turns", 600),
    maxPersistentTurns: numberOption(args, "max-persistent-turns", 24),
    turnTimeoutMs: numberOption(args, "turn-timeout-ms", 1_200_000, 1_000),
    requestTimeoutMs: numberOption(args, "request-timeout-ms", 90_000, 1_000),
    startupTimeoutMs: numberOption(args, "startup-timeout-ms", 180_000, 1_000),
    vaultRoot: root,
    evidenceRoots: [
      "paper_secs",
      "knowledge_notes",
      "experiment_notes",
      "idea_notes",
      "human_notes",
      "learning_outputs",
      "review_notes",
    ],
    projectRoot: root,
    skillRoot: path.join(root, ".codex/skills"),
    sourceConfigPath: path.join(sourceRoot, "config.toml"),
    sourceAuthPath: path.join(sourceRoot, "auth.json"),
    createdAt: new Date().toISOString(),
  };
}

function requireWorkDir(args: ParsedArgs): string {
  const value = option(args, "work-dir");
  if (!value) throw new Error("--work-dir is required");
  return path.resolve(value);
}

function printHelp(): void {
  process.stdout.write(`
Codex Learning Workflow (Node 22 native TypeScript)

Usage:
  node scripts/codex_learning_workflow.ts doctor [--work-dir DIR]
  node scripts/codex_learning_workflow.ts init --topic TOPIC [--work-dir DIR]
  node scripts/codex_learning_workflow.ts run --work-dir DIR [--resume]
       [--stop-after anchor-explore|direction-plan|direction-review]
  node scripts/codex_learning_workflow.ts status --work-dir DIR
  node scripts/codex_learning_workflow.ts validate --work-dir DIR
  node scripts/codex_learning_workflow.ts render --work-dir DIR

Important init limits:
  --max-anchors 30
  --max-stage1-rounds 6
  --max-stage1-tasks 24
  --evidence-tasks-per-round 4
  --max-directions-per-anchor 2
  --max-planner-turns 5
  --max-review-rounds 12
  --max-total-turns 600

All semantic roles use gpt-5.6-sol. Effort is fixed by role; ultra and subagents are forbidden.
`.trimStart());
}

async function doctorCommand(args: ParsedArgs): Promise<void> {
  const root = projectRoot();
  const requested = option(args, "work-dir");
  const workDir = requested
    ? path.resolve(requested)
    : fs.mkdtempSync(path.join(os.tmpdir(), "codex-learning-doctor-"));
  const store = new CanonicalStore(workDir);
  store.ensureLayout();
  const config = store.exists()
    ? store.load().config
    : buildConfig(args, "doctor capability probe");
  const state = store.exists()
    ? store.load()
    : createInitialState(`doctor-${runIdNow()}`, config);
  const result = await runDoctor(state, workDir);
  process.stdout.write(`${JSON.stringify({
    ok: true,
    workDir,
    projectRoot: root,
    manifest: result.manifest,
    warnings: result.warnings,
  }, null, 2)}\n`);
}

function initCommand(args: ParsedArgs): void {
  const topic = option(args, "topic");
  if (!topic?.trim()) throw new Error("--topic is required and must be nonempty");
  const root = projectRoot();
  const workDir = path.resolve(option(args, "work-dir", defaultWorkDir(root))!);
  const config = buildConfig(args, topic.trim());
  const state = createInitialState(path.basename(workDir), config);
  const store = new CanonicalStore(workDir);
  store.initialize(state);
  process.stdout.write(`${JSON.stringify({
    initialized: true,
    runId: state.runId,
    workDir,
    config: {
      topic: config.topic,
      model: config.model,
      roleEffort: config.roleEffort,
      maxAnchors: config.maxAnchors,
    },
  }, null, 2)}\n`);
}

async function runCommand(args: ParsedArgs): Promise<void> {
  const workDir = requireWorkDir(args);
  const store = new CanonicalStore(workDir);
  const state = store.load();
  const stopRaw = option(args, "stop-after");
  if (stopRaw && !["anchor-explore", "direction-plan", "direction-review"].includes(stopRaw)) {
    throw new Error("invalid --stop-after value");
  }
  const stopAfter = (stopRaw ?? null) as "anchor-explore" | "direction-plan" | "direction-review" | null;
  const doctor = await runDoctor(state, workDir);
  state.provider = doctor.manifest;
  for (const warning of doctor.warnings) {
    store.recordEvent("doctor_warning", { warning });
  }
  store.save(state, "doctor_passed");

  const runtime = new RuntimeManager(state, workDir);
  let stopping = false;
  const stopRuntime = async (signal: string): Promise<void> => {
    if (stopping) return;
    stopping = true;
    store.recordEvent("signal", { signal });
    store.save(state, `signal_${signal}`);
    await runtime.stop();
  };
  const onSigint = () => void stopRuntime("SIGINT");
  const onSigterm = () => void stopRuntime("SIGTERM");
  process.once("SIGINT", onSigint);
  process.once("SIGTERM", onSigterm);
  await runtime.start();
  try {
    await runAnchorStage(state, store, runtime);
    if (stopAfter === "anchor-explore") {
      process.stdout.write(`${JSON.stringify({ stoppedAfter: stopAfter, workDir }, null, 2)}\n`);
      return;
    }
    await runDirectionStage(
      state,
      store,
      runtime,
      stopAfter === "direction-plan" || stopAfter === "direction-review" ? stopAfter : null,
    );
    if (stopAfter) {
      process.stdout.write(`${JSON.stringify({ stoppedAfter: stopAfter, workDir }, null, 2)}\n`);
      return;
    }
  } finally {
    process.removeListener("SIGINT", onSigint);
    process.removeListener("SIGTERM", onSigterm);
    await runtime.stop();
  }

  const report = validateRunState(state);
  state.validation = {
    ...report,
    checkedAt: new Date().toISOString(),
  };
  store.saveValidatedState(state);
  if (!report.ok) {
    throw new Error(`workflow validation failed:\n${report.errors.join("\n")}`);
  }
  renderRun(state, workDir);
  process.stdout.write(`${JSON.stringify({
    complete: true,
    workDir,
    anchors: state.stage1.acceptedAnchorIds.length,
    directions: state.directions.filter((direction) => direction.status === "accepted").length,
    reviews: state.reviews.length,
    validationWarnings: report.warnings,
    final: path.join(workDir, "final.md"),
  }, null, 2)}\n`);
}

function statusCommand(args: ParsedArgs): void {
  const workDir = requireWorkDir(args);
  const state = new CanonicalStore(workDir).load();
  process.stdout.write(`${JSON.stringify({
    runId: state.runId,
    status: state.status,
    topic: state.config.topic,
    stage1: state.stage1,
    stage2: {
      status: state.stage2.status,
      planning: Object.fromEntries(
        Object.entries(state.stage2.anchorPlanning).map(([id, progress]) => [id, progress.status]),
      ),
      review: Object.fromEntries(
        Object.entries(state.stage2.directionReview).map(([id, progress]) => [id, progress.status]),
      ),
    },
    counts: {
      claims: state.claims.length,
      anchors: state.stage1.acceptedAnchorIds.length,
      entries: state.entries.length,
      edges: state.edges.length,
      directions: state.directions.length,
      reviews: state.reviews.length,
      turns: state.usage.turns,
    },
    validation: state.validation,
  }, null, 2)}\n`);
}

function validateCommand(args: ParsedArgs, shouldRender = false): void {
  const workDir = requireWorkDir(args);
  const store = new CanonicalStore(workDir);
  const state = store.load();
  const report = validateRunState(state);
  state.validation = {
    ...report,
    checkedAt: new Date().toISOString(),
  };
  store.saveValidatedState(state);
  if (shouldRender && report.ok) renderRun(state, workDir);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!report.ok) process.exitCode = 2;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  switch (args.command) {
    case "doctor":
      await doctorCommand(args);
      break;
    case "init":
      initCommand(args);
      break;
    case "run":
      await runCommand(args);
      break;
    case "status":
      statusCommand(args);
      break;
    case "validate":
      validateCommand(args, false);
      break;
    case "render":
      validateCommand(args, true);
      break;
    case "help":
    case "--help":
    case "-h":
      printHelp();
      break;
    default:
      throw new Error(`unknown command: ${args.command}`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});

