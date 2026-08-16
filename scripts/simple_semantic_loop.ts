#!/usr/bin/env node
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { RefactoredSemanticLoopController } from "./simple_semantic_loop/refactor/controller.ts";
import { LiveConsoleRenderer } from "./simple_semantic_loop/refactor/live_console.ts";
import {
  CodexAppServerRuntime,
} from "./simple_semantic_loop/refactor/runtime.ts";
import {
  continueRunFromFinished,
  initializeRun,
} from "./simple_semantic_loop/refactor/run_setup.ts";
import { authorizeRuntimeRecovery } from "./simple_semantic_loop/refactor/recovery.ts";
import {
  readObservationSummary,
  writeCheckpoint,
} from "./simple_semantic_loop/refactor/observations.ts";
import { FileLoopStore } from "./simple_semantic_loop/refactor/store.ts";
import {
  CURRENT_FORMAT_VERSION,
  type StateFile,
} from "./simple_semantic_loop/refactor/types.ts";
import {
  runDoctor,
  validateRun,
} from "./simple_semantic_loop/refactor/validation.ts";
import {
  computeRemainingRequirements,
} from "./simple_semantic_loop/refactor/workflow.ts";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    topic: { type: "string" },
    objective: { type: "string" },
    acceptance: { type: "string", multiple: true },
    "from-work-dir": { type: "string" },
    "work-dir": { type: "string" },
    model: { type: "string" },
    "max-rounds": { type: "string" },
    "max-exp-goals": { type: "string" },
    "additional-rounds": { type: "string" },
    "idle-timeout-ms": { type: "string" },
    "hard-timeout-ms": { type: "string" },
    "interrupt-grace-ms": { type: "string" },
    "exp-idle-timeout-ms": { type: "string" },
    "exp-hard-timeout-ms": { type: "string" },
    "exp-interrupt-grace-ms": { type: "string" },
    "recovery-token": { type: "string" },
    "reset-budgets": { type: "boolean", default: false },
    "no-provider": { type: "boolean", default: false },
    yolo: { type: "boolean", default: false },
    quiet: { type: "boolean", default: false },
    json: { type: "boolean", default: false },
  },
});

const command = positionals[0];

try {
  switch (command) {
    case "doctor":
      await doctorCommand();
      break;
    case "init":
      initCommand();
      break;
    case "continue":
      continueCommand();
      break;
    case "run":
      await runCommand(false);
      break;
    case "resume":
      await runCommand(true);
      break;
    case "recover-runtime":
      await recoverRuntimeCommand();
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
    case "checkpoint":
      checkpointCommand();
      break;
    case "pause":
      lifecycleCommand("PAUSED", "operator requested pause");
      break;
    case "cancel":
      lifecycleCommand("FAILED", "operator cancelled run");
      break;
    default:
      usage();
      process.exitCode = 2;
  }
} catch (error) {
  process.stderr.write(
    `ERROR: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
}

async function doctorCommand(): Promise<void> {
  const model = values.model ?? "gpt-5.6-sol";
  const renderer = values.quiet
    ? null
    : new LiveConsoleRenderer(process.stderr);
  const runtime = values["no-provider"]
    ? null
    : new CodexAppServerRuntime({
        sandbox: "read-only",
        onLiveEvent: renderer
          ? (event) => renderer.handle(event)
          : undefined,
      });
  const report = await runDoctor(projectRoot, runtime, model);
  renderer?.finish();
  print(report);
  if (!report.valid) process.exitCode = 1;
}

function initCommand(): void {
  if (!values.topic || !values["work-dir"]) {
    throw new Error("init requires --topic and --work-dir");
  }
  const maxRounds = values["max-rounds"]
    ? Number(values["max-rounds"])
    : undefined;
  const run = initializeRun({
    projectRoot,
    workDir: resolve(values["work-dir"]),
    topic: values.topic,
    objective: values.objective,
    acceptanceCriteria: values.acceptance,
    model: values.model,
    maxRounds,
    maxExperimentGoals: optionalNonNegativeInteger("max-exp-goals"),
    idleTimeoutMs: optionalPositiveInteger("idle-timeout-ms"),
    hardTimeoutMs: optionalPositiveInteger("hard-timeout-ms"),
    interruptGraceMs: optionalPositiveInteger("interrupt-grace-ms"),
    experimentIdleTimeoutMs: optionalPositiveInteger("exp-idle-timeout-ms"),
    experimentHardTimeoutMs: optionalPositiveInteger("exp-hard-timeout-ms"),
    experimentInterruptGraceMs: optionalPositiveInteger(
      "exp-interrupt-grace-ms",
    ),
  });
  const goal = new FileLoopStore(resolve(values["work-dir"]))
    .readJson<{ topic: string }>("workflow_goal.json");
  print({
    status: "initialized",
    runId: run.runId,
    workDir: resolve(values["work-dir"]),
    topic: goal.topic,
    model: run.model,
    maxRounds: run.budgets.maxRounds,
    maxExperimentGoals: run.budgets.maxExperimentGoals,
    experimentGoalTokenBudget: null,
    timeoutProfiles: run.budgets.timeoutProfiles,
  });
}

function continueCommand(): void {
  if (!values["from-work-dir"] || !values["work-dir"]) {
    throw new Error("continue requires --from-work-dir and --work-dir");
  }
  const run = continueRunFromFinished({
    projectRoot,
    sourceWorkDir: resolve(values["from-work-dir"]),
    workDir: resolve(values["work-dir"]),
    model: values.model,
    maxRounds: optionalPositiveInteger("max-rounds"),
    maxExperimentGoals: optionalNonNegativeInteger("max-exp-goals"),
    idleTimeoutMs: optionalPositiveInteger("idle-timeout-ms"),
    hardTimeoutMs: optionalPositiveInteger("hard-timeout-ms"),
    interruptGraceMs: optionalPositiveInteger("interrupt-grace-ms"),
    experimentIdleTimeoutMs: optionalPositiveInteger("exp-idle-timeout-ms"),
    experimentHardTimeoutMs: optionalPositiveInteger("exp-hard-timeout-ms"),
    experimentInterruptGraceMs: optionalPositiveInteger(
      "exp-interrupt-grace-ms",
    ),
    resetBudgets: values["reset-budgets"],
  });
  const store = new FileLoopStore(resolve(values["work-dir"]));
  print({
    status: "continued",
    runId: run.runId,
    sourceRunId: run.continuation?.sourceRunId,
    sourceWorkDir: run.continuation?.sourceWorkDir,
    sourceLifecycle: run.continuation?.sourceLifecycle ?? "FINISHED",
    budgetReset: run.continuation?.budgetReset ?? false,
    workDir: resolve(values["work-dir"]),
    nextRole: store.readState().sequence[0]?.role ?? null,
    maxRounds: run.budgets.maxRounds,
    maxExperimentGoals: run.budgets.maxExperimentGoals,
    timeoutProfiles: run.budgets.timeoutProfiles,
  });
}

async function recoverRuntimeCommand(): Promise<void> {
  if (!values["recovery-token"]) {
    throw new Error("recover-runtime requires --recovery-token");
  }
  const store = new FileLoopStore(requiredWorkDir());
  const authorization = authorizeRuntimeRecovery(store, {
    token: values["recovery-token"],
    timeoutOverride: {
      ...(optionalPositiveInteger("idle-timeout-ms") !== undefined
        ? { idleTimeoutMs: optionalPositiveInteger("idle-timeout-ms") }
        : {}),
      ...(optionalPositiveInteger("hard-timeout-ms") !== undefined
        ? { hardTimeoutMs: optionalPositiveInteger("hard-timeout-ms") }
        : {}),
      ...(optionalPositiveInteger("interrupt-grace-ms") !== undefined
        ? { interruptGraceMs: optionalPositiveInteger("interrupt-grace-ms") }
        : {}),
    },
  });
  if (authorization.status === "ALREADY_CONSUMED") {
    print(authorization);
    return;
  }
  await runCommand(false);
}

async function runCommand(resume: boolean): Promise<void> {
  const workDir = requiredWorkDir();
  const store = new FileLoopStore(workDir);
  const run = store.readRun();
  requireCurrentWritableFormat(store);
  const renderer = values.quiet
    ? null
    : new LiveConsoleRenderer(process.stderr);
  const sandbox = values.yolo ? "danger-full-access" : "read-only";
  if (values.yolo) {
    process.stderr.write(
      "WARNING: --yolo uses approvalPolicy=never and sandbox=danger-full-access for fresh Agent Turns. " +
        "Tool effects can occur before output validation; Agent output remains untrusted and only Script records are workflow authority.\n",
    );
  }
  const runtime = new CodexAppServerRuntime({
    sandbox,
    onLiveEvent: renderer
      ? (event) => renderer.handle(event)
      : undefined,
  });
  const controller = new RefactoredSemanticLoopController(
    store,
    runtime,
  );
  const additionalRounds = optionalPositiveInteger("additional-rounds");
  if (!resume && additionalRounds !== undefined) {
    throw new Error("--additional-rounds is only valid with resume");
  }
  const outcome = await controller.run(resume, additionalRounds);
  renderer?.finish();
  print(outcome);
  process.exitCode = outcome.workflowOutcome === "FINISHED" ? 0 : 2;
}

function statusCommand(): void {
  const store = new FileLoopStore(requiredWorkDir());
  const state = store.readState();
  const run = store.readRun();
  const remainingRequirements = computeRemainingRequirements(
    store,
    state,
    true,
  );
  print({
    run,
    workflowGoal: store.readJson("workflow_goal.json"),
    state,
    controller: controllerStatusProjection(
      store,
      state,
      run.budgets.maxRounds,
      remainingRequirements,
    ),
    remainingRequirements,
    objects: store.readObjects(),
    observations: readObservationSummary(store),
    experiments: store.experimentRefs().map((ref) => ({
      experimentRef: ref,
      ...store.readExperiment(ref),
    })),
    latestRuntimeFailure: latestRuntimeFailure(store),
    runtimeRecoveryEligible:
      state.lifecycle === "FAILED" &&
      state.failureKind === "RUNTIME_RETRY_EXHAUSTED",
  });
}

function latestRuntimeFailure(store: FileLoopStore): unknown {
  const failed = store.turnRefs()
    .map((ref) => ({ ref, turn: store.readTurn(ref) }))
    .filter(({ turn }) => turn.turnState === "RUNTIME_FAILED")
    .sort((left, right) =>
      left.turn.startedAt.localeCompare(right.turn.startedAt)
    )
    .at(-1);
  if (!failed) return null;
  return {
    turnRef: failed.ref,
    role: failed.turn.role,
    failureKind: failed.turn.runtimeFailureKind,
    outputCapture: failed.turn.outputCapture,
    providerThreadId: failed.turn.providerThreadId,
    providerTurnId: failed.turn.providerTurnId,
    runtimeRef: failed.turn.runtimeRef,
    partialOutputRef: failed.turn.partialOutputRef,
    rawOutputRef: failed.turn.rawOutputRef,
  };
}

function eventsCommand(): void {
  const store = new FileLoopStore(requiredWorkDir());
  const text = store.exists("events.jsonl")
    ? store.readText("events.jsonl")
    : "";
  if (values.json) {
    print(
      text
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line)),
    );
  } else {
    process.stdout.write(text);
  }
}

function validateCommand(): void {
  const report = validateRun(requiredWorkDir());
  print(report);
  if (!report.valid) process.exitCode = 1;
}

function renderCommand(): void {
  const store = new FileLoopStore(requiredWorkDir());
  if (!store.exists("final/report.md")) {
    throw new Error(
      "final/report.md does not exist; rendering occurs only after FINISH_WORKFLOW and Script checks",
    );
  }
  print({
    status: "already_rendered_by_controller",
    finalPath: store.absolute("final/report.md"),
  });
}

function checkpointCommand(): void {
  const store = new FileLoopStore(requiredWorkDir());
  store.acquireLock();
  try {
    requireCurrentWritableFormat(store);
    const state = store.readState();
    const checkpointRef = writeCheckpoint(
      store,
      state,
      state.reason ?? "operator requested checkpoint",
    );
    print({ status: "checkpoint_written", checkpointRef });
  } finally {
    store.releaseLock();
  }
}

function lifecycleCommand(
  lifecycle: Extract<StateFile["lifecycle"], "PAUSED" | "FAILED">,
  reason: string,
): void {
  const store = new FileLoopStore(requiredWorkDir());
  store.acquireLock();
  try {
    requireCurrentWritableFormat(store);
    const state = store.readState();
    if (state.lifecycle === "FINISHED") {
      throw new Error("completed run is immutable");
    }
    if (lifecycle === "PAUSED" && state.lifecycle === "PAUSED") {
      print({
        lifecycle: state.lifecycle,
        stateRevision: state.revision,
        reason: state.reason,
        pauseKind: state.pauseKind ?? null,
      });
      return;
    }
    store.writeState({
      ...state,
      revision: state.revision + 1,
      lifecycle,
      reason,
      pauseKind: lifecycle === "PAUSED" ? "OPERATOR_REQUESTED" : null,
      failureKind: lifecycle === "FAILED" ? "NON_RECOVERABLE" : null,
      runtimeRecovery: null,
      node: null,
    }, lifecycle === "PAUSED" ? "RUN_PAUSED" : "RUN_CANCELLED");
    store.writeJson("final/outcome.json", {
      workflowOutcome: lifecycle === "PAUSED" ? "PAUSED" : "FAILED",
      reportRef: null,
      reason,
    });
    print({ lifecycle, stateRevision: state.revision + 1, reason });
  } finally {
    store.releaseLock();
  }
}

function requireCurrentWritableFormat(store: FileLoopStore): void {
  const run = store.readRun();
  const state = store.readState();
  if (
    run.formatVersion !== CURRENT_FORMAT_VERSION ||
    state.formatVersion !== CURRENT_FORMAT_VERSION
  ) {
    throw new Error(
      `formatVersion ${run.formatVersion}/${state.formatVersion} is read-only; this command requires formatVersion ${CURRENT_FORMAT_VERSION}`,
    );
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

function optionalPositiveInteger(
  name:
    | "idle-timeout-ms"
    | "hard-timeout-ms"
    | "interrupt-grace-ms"
    | "exp-idle-timeout-ms"
    | "exp-hard-timeout-ms"
    | "exp-interrupt-grace-ms"
    | "additional-rounds"
    | "max-rounds",
): number | undefined {
  const raw = values[name];
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`--${name} must be a positive integer`);
  }
  return value;
}

function optionalNonNegativeInteger(
  name: "max-exp-goals",
): number | undefined {
  const raw = values[name];
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`--${name} must be a non-negative integer`);
  }
  return value;
}

function controllerStatusProjection(
  store: FileLoopStore,
  state: StateFile,
  defaultRoundGrant: number,
  remainingRequirements: string[],
): Record<string, unknown> {
  const authorizedThroughRound =
    state.roundBudget?.authorizedThroughRound ?? defaultRoundGrant;
  return {
    pauseKind: state.pauseKind ?? null,
    nextRole: state.sequence[0]?.role ?? null,
    nextAction: inferNextAction(store, state, remainingRequirements),
    finishEligibility: {
      mechanicallyEligible: remainingRequirements.length === 0,
      semanticDecisionRequired: state.lifecycle !== "FINISHED",
      blockedBy: remainingRequirements,
    },
    roundAuthorization: {
      initialRoundBudget: defaultRoundGrant,
      authorizedThroughRound,
      remainingAuthorizedRounds:
        state.lifecycle === "FINISHED" || state.lifecycle === "FAILED"
          ? 0
          : Math.max(0, authorizedThroughRound - state.round + 1),
      defaultAdditionalRoundsOnResume: defaultRoundGrant,
      lastAuthorizationRef:
        state.roundBudget?.lastAuthorizationRef ?? null,
    },
  };
}

function inferNextAction(
  store: FileLoopStore,
  state: StateFile,
  remainingRequirements: string[],
): string | null {
  const step = state.sequence[0];
  if (!step) return null;
  if (step.role === "EXP_GOAL") return "RUN_EXPERIMENT";
  if (step.bindingRef && store.exists(step.bindingRef)) {
    return store.readJson<{ action?: string }>(step.bindingRef).action ?? null;
  }
  if (step.role === "DECISION") return "DECIDE_NEXT_BRANCH";
  if (step.role === "REVIEWER") {
    if (step.mode === "ANCHOR_REASSESS") return "REVIEW_ANCHOR";
    if (step.mode === "POST_EXP_REVIEW") {
      const latest = state.latestExperimentResultRef &&
          store.exists(state.latestExperimentResultRef)
        ? store.readJson<{ directionWork?: string | null }>(
          state.latestExperimentResultRef,
        )
        : null;
      return latest?.directionWork ? "REVIEW_DIRECTION" : "REVIEW_ANCHOR";
    }
    const kind = state.pending?.objectKind ?? state.preReview?.objectKind;
    return kind ? `REVIEW_${kind}` : "SELECT_REVIEW_TARGET";
  }
  const requirement = remainingRequirements[0] ?? null;
  if (requirement === "ANCHOR_REQUIRED") return "CREATE_ANCHOR";
  if (requirement?.startsWith("ANCHOR_REVIEW_PASS_REQUIRED:")) {
    return "DEEPEN_ANCHOR";
  }
  if (requirement?.startsWith("DIRECTION_REQUIRED:")) {
    return "CREATE_DIRECTION";
  }
  if (requirement?.startsWith("DIRECTION_REVIEW_PASS_REQUIRED:")) {
    return "DEEPEN_DIRECTION";
  }
  return "CREATE_ANCHOR_CONVERGENCE_PROBE";
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function usage(): void {
  process.stderr.write(`Usage:
  node scripts/simple_semantic_loop.ts doctor [--model MODEL] [--no-provider] [--quiet]
  node scripts/simple_semantic_loop.ts init --topic TOPIC --work-dir DIR [--objective TEXT] [--acceptance TEXT ...] [--model MODEL] [--max-rounds N] [--max-exp-goals N] [--idle-timeout-ms N] [--hard-timeout-ms N] [--interrupt-grace-ms N] [--exp-idle-timeout-ms N] [--exp-hard-timeout-ms N] [--exp-interrupt-grace-ms N]
  node scripts/simple_semantic_loop.ts continue --from-work-dir SOURCE_DIR --work-dir NEW_DIR [--reset-budgets] [--model MODEL] [--max-rounds N] [--max-exp-goals N] [--idle-timeout-ms N] [--hard-timeout-ms N] [--interrupt-grace-ms N] [--exp-idle-timeout-ms N] [--exp-hard-timeout-ms N] [--exp-interrupt-grace-ms N]
  node scripts/simple_semantic_loop.ts run --work-dir DIR [--yolo] [--quiet]
  node scripts/simple_semantic_loop.ts resume --work-dir DIR [--additional-rounds N] [--yolo] [--quiet]
  node scripts/simple_semantic_loop.ts recover-runtime --work-dir DIR --recovery-token TOKEN [--idle-timeout-ms N] [--hard-timeout-ms N] [--interrupt-grace-ms N] [--yolo] [--quiet]
  node scripts/simple_semantic_loop.ts status --work-dir DIR
  node scripts/simple_semantic_loop.ts events --work-dir DIR [--json]
  node scripts/simple_semantic_loop.ts validate --work-dir DIR
  node scripts/simple_semantic_loop.ts render --work-dir DIR
  node scripts/simple_semantic_loop.ts checkpoint --work-dir DIR
  node scripts/simple_semantic_loop.ts pause --work-dir DIR
  node scripts/simple_semantic_loop.ts cancel --work-dir DIR
`);
}
