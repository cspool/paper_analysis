#!/usr/bin/env node
import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { parseCatchConfig, sourceSpecFromUrl } from "./paper_catch/config.ts";
import { PaperCatchController } from "./paper_catch/controller.ts";
import { PaperCatchStore, timestampId } from "./paper_catch/store.ts";
import type { ControllerOptions, PaperCatchRun, ReportManifest } from "./paper_catch/types.ts";

const scriptPath = fileURLToPath(import.meta.url);
const projectRoot = resolve(dirname(scriptPath), "..");
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    config: { type: "string" },
    "output-dir": { type: "string" },
    "batch-size": { type: "string" },
    "lookback-days": { type: "string" },
    model: { type: "string" },
    "codex-bin": { type: "string" },
    "no-search": { type: "boolean", default: false },
    "max-attempts": { type: "string" },
    "codex-timeout-ms": { type: "string" },
  },
});

const command = positionals[0];

try {
  switch (command) {
    case "start":
      startCommand();
      break;
    case "run":
      await runCommand(false);
      break;
    case "scan":
      await runCommand(true);
      break;
    case "status":
      statusCommand();
      break;
    case "doctor":
      doctorCommand();
      break;
    case "validate":
      validateCommand();
      break;
    default:
      usage();
      process.exitCode = 2;
  }
} catch (error) {
  process.stderr.write(`ERROR: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}

function options(scanOnly = false): ControllerOptions {
  return {
    projectRoot,
    configPath: resolve(values.config ?? resolve(projectRoot, "human_notes/Catch_Paper_Urls.md")),
    outputDir: resolve(values["output-dir"] ?? resolve(projectRoot, "paper_catch")),
    batchSize: positiveInteger(values["batch-size"], 20, "batch-size"),
    lookbackDays: positiveNumber(values["lookback-days"], 7, "lookback-days"),
    model: values.model ?? null,
    codexBin: values["codex-bin"] ?? "codex",
    useWebSearch: !values["no-search"],
    maxAttemptsPerInvocation: positiveInteger(values["max-attempts"], 2, "max-attempts"),
    codexTimeoutMs: positiveInteger(values["codex-timeout-ms"], 900_000, "codex-timeout-ms"),
    scanOnly,
  };
}

async function runCommand(scanOnly: boolean): Promise<void> {
  const config = options(scanOnly);
  const store = new PaperCatchStore(config.outputDir);
  store.initialize();
  const livePid = liveLockPid(store);
  if (livePid !== null) {
    throw new Error(`paper catch is already running with pid ${livePid}`);
  }
  store.acquireLock();
  try {
    const outcome = await new PaperCatchController(store, config).run();
    print(outcome);
    process.exitCode = outcome.status === "PAUSED" ? 2 : 0;
  } finally {
    store.releaseLock();
  }
}

function startCommand(): void {
  const config = options(false);
  const store = new PaperCatchStore(config.outputDir);
  store.initialize();
  const livePid = liveLockPid(store);
  if (livePid !== null) {
    throw new Error(`paper catch is already running with pid ${livePid}`);
  }
  const logRef = `logs/job-${timestampId()}.log`;
  const logPath = store.absolute(logRef);
  const descriptor = openSync(logPath, "a", 0o600);
  const forwarded = process.argv.slice(3);
  const child = spawn(process.execPath, [scriptPath, "run", ...forwarded], {
    cwd: projectRoot,
    detached: true,
    stdio: ["ignore", descriptor, descriptor],
    env: process.env,
  });
  closeSync(descriptor);
  child.unref();
  store.writeJson(".state/last_start.json", {
    pid: child.pid,
    startedAt: new Date().toISOString(),
    logRef,
    command: [process.execPath, scriptPath, "run", ...forwarded],
  });
  print({ status: "started", pid: child.pid, log: logPath });
}

function statusCommand(): void {
  const config = options(false);
  const store = new PaperCatchStore(config.outputDir);
  store.initialize();
  const activeId = store.activeRunId();
  const active = activeId && store.exists(`.runs/${activeId}/run.json`)
    ? store.readRun(activeId)
    : null;
  let lock: Record<string, unknown> | null = null;
  let lockLive = false;
  if (store.exists(".state/run.lock")) {
    try {
      lock = store.readJson<Record<string, unknown>>(".state/run.lock");
      if (typeof lock.pid === "number") {
        try {
          process.kill(lock.pid, 0);
          lockLive = true;
        } catch {
          lockLive = false;
        }
      }
    } catch {
      lock = { malformed: true };
    }
  }
  print({
    running: lockLive,
    lock,
    activeRun: active,
    latestReport: store.latestReport(),
    lastStart: store.exists(".state/last_start.json")
      ? store.readJson(".state/last_start.json")
      : null,
  });
}

function doctorCommand(): void {
  const config = options(false);
  const parsed = parseCatchConfig(config.configPath);
  const sources = parsed.urls.map(sourceSpecFromUrl);
  const git = commandVersion("git", ["--version"]);
  const codex = commandVersion(config.codexBin, ["--version"]);
  const template = resolve(config.outputDir, "PAPER_ENTRY_TEMPLATE.md");
  const report = {
    valid: git.ok && codex.ok && existsSync(template),
    config: {
      path: parsed.configPath,
      hash: parsed.configHash,
      sourceCount: sources.length,
      sources,
      interest: parsed.interest,
    },
    template: { path: template, exists: existsSync(template) },
    git,
    codex,
    defaults: {
      batchSize: config.batchSize,
      lookbackDays: config.lookbackDays,
      useWebSearch: config.useWebSearch,
      codexTimeoutMs: config.codexTimeoutMs,
    },
  };
  print(report);
  if (!report.valid) process.exitCode = 1;
}

function validateCommand(): void {
  const config = options(false);
  const store = new PaperCatchStore(config.outputDir);
  store.initialize();
  const latest = store.latestReport();
  const errors: string[] = [];
  if (!latest) errors.push("no timestamped report exists");
  let manifest: ReportManifest | null = null;
  let run: PaperCatchRun | null = null;
  if (latest?.manifestRef) {
    manifest = store.readJson<ReportManifest>(latest.manifestRef);
    if (!store.exists(manifest.reportRef)) errors.push(`missing report: ${manifest.reportRef}`);
    if (!store.exists(manifest.aggregateRef)) errors.push(`missing aggregate: ${manifest.aggregateRef}`);
    if (!store.exists(manifest.candidatesRef)) errors.push(`missing candidates: ${manifest.candidatesRef}`);
    if (!store.exists(manifest.sourceSnapshotsRef)) errors.push(`missing source snapshot: ${manifest.sourceSnapshotsRef}`);
    for (const ref of manifest.batchResultRefs) {
      if (!store.exists(ref)) errors.push(`missing batch result: ${ref}`);
    }
    if (store.exists(`.runs/${manifest.runId}/run.json`)) {
      run = store.readRun(manifest.runId);
      if (run.status !== "COMPLETED") errors.push(`report run is not COMPLETED: ${run.status}`);
      if (run.reportRef !== manifest.reportRef) errors.push("run/report manifest reference mismatch");
    } else {
      errors.push(`missing run record: ${manifest.runId}`);
    }
  } else if (latest) {
    errors.push(`missing report manifest for ${latest.reportRef}`);
  }
  print({ valid: errors.length === 0, latest, manifest, run, errors });
  if (errors.length > 0) process.exitCode = 1;
}

function commandVersion(command: string, args: string[]): { ok: boolean; output: string } {
  const result = spawnSync(command, args, { encoding: "utf8", timeout: 30_000 });
  return {
    ok: result.status === 0,
    output: `${result.stdout ?? ""}${result.stderr ?? ""}`.trim(),
  };
}

function liveLockPid(store: PaperCatchStore): number | null {
  if (!store.exists(".state/run.lock")) return null;
  try {
    const lock = store.readJson<{ pid?: number }>(".state/run.lock");
    if (typeof lock.pid !== "number") return null;
    try {
      process.kill(lock.pid, 0);
      return lock.pid;
    } catch {
      return null;
    }
  } catch {
    return null;
  }
}

function positiveInteger(value: string | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`);
  return parsed;
}

function positiveNumber(value: string | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${name} must be positive`);
  return parsed;
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function usage(): void {
  process.stderr.write(`Usage:
  node scripts/paper_catch.ts doctor [options]
  node scripts/paper_catch.ts scan [options]
  node scripts/paper_catch.ts start [options]
  node scripts/paper_catch.ts run [options]
  node scripts/paper_catch.ts status [options]
  node scripts/paper_catch.ts validate [options]

Commands:
  start     Launch a detached background run; manually invoke this on your schedule.
  run       Run/resume in the foreground. An unfinished batch run is resumed automatically.
  scan      Fetch Git updates and freeze candidate batches without calling Codex or publishing.
  status    Show lock, active run, latest report, and background log.
  doctor    Check config, template, Git, and Codex CLI.
  validate  Audit the latest published report and all referenced batch artifacts.

Options:
  --config PATH              Default: human_notes/Catch_Paper_Urls.md
  --output-dir PATH          Default: paper_catch
  --batch-size N             Default: 20
  --lookback-days N          First run only; default: 7
  --model MODEL              Optional Codex model override
  --codex-bin PATH           Default: codex
  --no-search                Disable live web search in Codex batch sessions
  --max-attempts N           Fresh attempts per batch per invocation; default: 2
  --codex-timeout-ms N       Hard timeout per batch session; default: 900000
`);
}
