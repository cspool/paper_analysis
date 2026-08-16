#!/usr/bin/env node
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { DirectionExperimentController } from "./direction_experiment_loop/controller.ts";
import {
  initializeDirectionExperimentRun,
} from "./direction_experiment_loop/setup.ts";
import { DirectionExperimentStore } from "./direction_experiment_loop/store.ts";
import { validateDirectionExperimentRun } from "./direction_experiment_loop/validation.ts";
import { LiveConsoleRenderer } from "./simple_semantic_loop/refactor/live_console.ts";
import { CodexAppServerRuntime } from "./simple_semantic_loop/refactor/runtime.ts";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    "direction-result": { type: "string" },
    "source-work-dir": { type: "string" },
    "work-dir": { type: "string" },
    model: { type: "string" },
    "max-cycles": { type: "string" },
    "additional-cycles": { type: "string" },
    "decision-idle-timeout-ms": { type: "string" },
    "decision-hard-timeout-ms": { type: "string" },
    "judge-idle-timeout-ms": { type: "string" },
    "judge-hard-timeout-ms": { type: "string" },
    "lab-idle-timeout-ms": { type: "string" },
    "lab-hard-timeout-ms": { type: "string" },
    "interrupt-grace-ms": { type: "string" },
    yolo: { type: "boolean", default: false },
    quiet: { type: "boolean", default: false },
    json: { type: "boolean", default: false },
  },
});

try {
  switch (positionals[0]) {
    case "init":
      initCommand();
      break;
    case "run":
      await runCommand(false);
      break;
    case "resume":
      await runCommand(true);
      break;
    case "status":
      statusCommand();
      break;
    case "events":
      eventsCommand();
      break;
    case "validate":
      validateCommand();
      break;
    case "render":
      renderCommand();
      break;
    case "pause":
      pauseCommand();
      break;
    case "cancel":
      cancelCommand();
      break;
    default:
      usage();
      process.exitCode = 2;
  }
} catch (error) {
  process.stderr.write(`ERROR: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}

function initCommand(): void {
  if (!values["direction-result"] || !values["work-dir"]) {
    throw new Error("init requires --direction-result and --work-dir");
  }
  const run = initializeDirectionExperimentRun({
    projectRoot,
    workDir: resolve(values["work-dir"]),
    directionResultPath: resolve(values["direction-result"]),
    sourceWorkDir: values["source-work-dir"]
      ? resolve(values["source-work-dir"])
      : undefined,
    model: values.model,
    maxCycles: positiveInteger("max-cycles"),
    decisionIdleTimeoutMs: positiveInteger("decision-idle-timeout-ms"),
    decisionHardTimeoutMs: positiveInteger("decision-hard-timeout-ms"),
    judgeIdleTimeoutMs: positiveInteger("judge-idle-timeout-ms"),
    judgeHardTimeoutMs: positiveInteger("judge-hard-timeout-ms"),
    labIdleTimeoutMs: positiveInteger("lab-idle-timeout-ms"),
    labHardTimeoutMs: positiveInteger("lab-hard-timeout-ms"),
    interruptGraceMs: positiveInteger("interrupt-grace-ms"),
  });
  print({
    status: "initialized",
    formatVersion: run.formatVersion,
    runId: run.runId,
    workDir: resolve(values["work-dir"]),
    directionId: run.source.directionId,
    directionRevision: run.source.directionRevision,
    parentAnchorId: run.source.parentAnchorId,
    maxCycles: run.budgets.initialLabCycles,
    experimentPolicyRef: run.inputs.experimentPolicy.path,
    nextRole: "EXPERIMENT_DECISION",
  });
}

async function runCommand(resume: boolean): Promise<void> {
  const workDir = requiredWorkDir();
  const store = new DirectionExperimentStore(workDir);
  const renderer = values.quiet
    ? null
    : new LiveConsoleRenderer(process.stderr, "direction-loop");
  const sandbox = values.yolo ? "danger-full-access" : "read-only";
  if (values.yolo) {
    process.stderr.write(
      "WARNING: --yolo grants fresh Decision/Judge Turns and persistent Lab Goals danger-full-access. " +
        "Tool effects can occur before output review; only Script records control the loop.\n",
    );
  }
  const runtime = new CodexAppServerRuntime({
    sandbox,
    onLiveEvent: renderer ? (event) => renderer.handle(event) : undefined,
  });
  const requestSignalPause = (signal: "SIGINT" | "SIGTERM") => {
    store.writePauseRequest(`operator requested pause via ${signal}`, signal);
    process.stderr.write(`[direction-loop] ${signal} received; requesting checkpoint and graceful Goal interruption\n`);
  };
  const onSigint = () => requestSignalPause("SIGINT");
  const onSigterm = () => requestSignalPause("SIGTERM");
  process.on("SIGINT", onSigint);
  process.on("SIGTERM", onSigterm);
  try {
    const outcome = await new DirectionExperimentController(store, runtime).run(
      resume,
      positiveInteger("additional-cycles"),
    );
    print(outcome);
    process.exitCode = outcome.workflowOutcome === "FINISHED" ? 0 : 2;
  } finally {
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigterm);
    await runtime.close();
    renderer?.finish();
  }
}

function statusCommand(): void {
  const store = new DirectionExperimentStore(requiredWorkDir());
  const state = store.readState();
  const activeGoal = state.activeGoalRecordRef && store.exists(state.activeGoalRecordRef)
    ? store.readJson<Record<string, unknown>>(state.activeGoalRecordRef)
    : null;
  const activeInvocation = state.activeLabInvocationRef && store.exists(state.activeLabInvocationRef)
    ? store.readJson<Record<string, unknown>>(state.activeLabInvocationRef)
    : null;
  const checkpoint = state.latestCheckpointRef && store.exists(state.latestCheckpointRef)
    ? store.readJson<Record<string, unknown>>(state.latestCheckpointRef)
    : null;
  print({
    run: store.readRun(),
    state,
    nextRole: state.node === "JUDGE"
      ? "EVIDENCE_JUDGE"
      : state.node === "LAB_GOAL"
      ? "DIRECTION_LAB_GOAL"
      : state.node === "DECISION"
      ? "EXPERIMENT_DECISION"
      : null,
    activeGoal,
    activeInvocation,
    checkpoint,
    history: store.readText("history.jsonl").split("\n").filter(Boolean).map(
      (line) => JSON.parse(line),
    ),
    validation: validateDirectionExperimentRun(store.workDir),
  });
}

function eventsCommand(): void {
  const store = new DirectionExperimentStore(requiredWorkDir());
  const text = store.exists("events.jsonl") ? store.readText("events.jsonl") : "";
  if (values.json) {
    print(text.split("\n").filter(Boolean).map((line) => JSON.parse(line)));
  } else process.stdout.write(text);
}

function validateCommand(): void {
  const report = validateDirectionExperimentRun(requiredWorkDir());
  print(report);
  if (!report.valid) process.exitCode = 1;
}

function renderCommand(): void {
  const store = new DirectionExperimentStore(requiredWorkDir());
  if (!store.exists("final/report.md")) {
    throw new Error("final/report.md does not exist; only a terminal Judge decision renders it");
  }
  print({ status: "already_rendered", finalPath: store.absolute("final/report.md") });
}

function pauseCommand(): void {
  const store = new DirectionExperimentStore(requiredWorkDir());
  const state = store.readState();
  if (state.lifecycle === "FINISHED") throw new Error("finished run is immutable");
  if (state.lifecycle === "PAUSED") {
    print({
      lifecycle: state.lifecycle,
      evidenceScope: state.evidenceScope,
      reason: state.reason,
      pauseKind: state.pauseKind,
    });
    return;
  }
  if (!store.exists(".run.lock")) {
    try {
      store.acquireLock();
      const current = store.readState();
      store.writeState({
        ...current,
        revision: current.revision + 1,
        lifecycle: "PAUSED",
        reason: "operator requested pause while Controller was not active",
        pauseKind: "OPERATOR_REQUESTED",
        activeLabInvocationRef: null,
      }, "RUN_PAUSED");
      print({
        status: "paused",
        lifecycle: "PAUSED",
        node: current.node,
        reason: "Controller was not active; state paused directly",
      });
      return;
    } catch (error) {
      if (!String(error).includes("run is already locked")) throw error;
    } finally {
      store.releaseLock();
    }
  }
  store.writePauseRequest("operator requested pause", "CLI");
  print({
    status: "pause_requested",
    lifecycle: state.lifecycle,
    node: state.node,
    requestRef: "control/pause-request.json",
    reason: "the active Controller will interrupt Lab, then adopt result or index checkpoint",
  });
}

function cancelCommand(): void {
  const store = new DirectionExperimentStore(requiredWorkDir());
  store.acquireLock();
  try {
    const state = store.readState();
    if (state.lifecycle === "FINISHED") throw new Error("finished run is immutable");
    store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle: "FAILED",
      node: null,
      reason: "operator cancelled run",
      pauseKind: null,
    }, "RUN_CANCELLED");
    print({
      lifecycle: "FAILED",
      evidenceScope: state.evidenceScope,
      reason: "operator cancelled run",
    });
  } finally {
    store.releaseLock();
  }
}

function requiredWorkDir(): string {
  if (!values["work-dir"]) throw new Error("--work-dir is required");
  const workDir = resolve(values["work-dir"]);
  if (!existsSync(resolve(workDir, "run.json"))) {
    throw new Error(`run.json does not exist in ${workDir}; run init first`);
  }
  return workDir;
}

function positiveInteger(name:
  | "max-cycles"
  | "additional-cycles"
  | "decision-idle-timeout-ms"
  | "decision-hard-timeout-ms"
  | "judge-idle-timeout-ms"
  | "judge-hard-timeout-ms"
  | "lab-idle-timeout-ms"
  | "lab-hard-timeout-ms"
  | "interrupt-grace-ms"
): number | undefined {
  const raw = values[name];
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`--${name} must be a positive integer`);
  }
  return value;
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function usage(): void {
  process.stderr.write(`Usage:
  node scripts/direction_experiment_loop.ts init --direction-result FILE --work-dir DIR [--source-work-dir DIR] [--model MODEL] [--max-cycles N] [--decision-idle-timeout-ms N] [--decision-hard-timeout-ms N] [--judge-idle-timeout-ms N] [--judge-hard-timeout-ms N] [--lab-idle-timeout-ms N] [--lab-hard-timeout-ms N] [--interrupt-grace-ms N]
  node scripts/direction_experiment_loop.ts run --work-dir DIR [--yolo] [--quiet]
  node scripts/direction_experiment_loop.ts resume --work-dir DIR [--additional-cycles N] [--yolo] [--quiet]
  node scripts/direction_experiment_loop.ts status --work-dir DIR
  node scripts/direction_experiment_loop.ts events --work-dir DIR [--json]
  node scripts/direction_experiment_loop.ts validate --work-dir DIR
  node scripts/direction_experiment_loop.ts render --work-dir DIR
  node scripts/direction_experiment_loop.ts pause --work-dir DIR
  node scripts/direction_experiment_loop.ts cancel --work-dir DIR
`);
}
