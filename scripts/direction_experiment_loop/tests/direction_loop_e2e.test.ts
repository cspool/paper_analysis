import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { ScriptedTurnRuntime } from "../../simple_semantic_loop/refactor/runtime.ts";
import type {
  GoalDispatch,
  RawGoalResult,
  RawTurnResult,
  TurnDispatch,
  TurnRuntime,
} from "../../simple_semantic_loop/refactor/types.ts";
import { DirectionExperimentController } from "../controller.ts";
import { parseDecisionResult, parseJudgeResult } from "../protocol.ts";
import { initializeDirectionExperimentRun } from "../setup.ts";
import { DirectionExperimentStore } from "../store.ts";
import { validateDirectionExperimentRun } from "../validation.ts";

const projectRoot = resolve(import.meta.dirname, "../../..");

test("v7 runs Decision → atomic Lab → Judge → Decision with frozen run-local Skills", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "experiment-run");
  const run = initializeDirectionExperimentRun({
    projectRoot,
    workDir,
    directionResultPath: fixture.directionPath,
    maxCycles: 2,
  });
  assert.equal(run.formatVersion, 7);
  assert.equal(run.budgets.initialLabCycles, 2);
  assert.match(run.skills.lab.path, /^inputs\/skills\//);
  assert.equal(existsSync(resolve(workDir, run.skills.lab.path)), true);

  const turns: TurnDispatch[] = [];
  const goals: GoalDispatch[] = [];
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      turns.push(dispatch);
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        return decisionCount === 1
          ? decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE")
          : terminal(
            "COMPLETE_SUPPORT",
            "LOCAL_SINGLE_GPU_PERFORMANCE",
            "reviewed local paired evidence is sufficient",
          );
      }
      return judgment(
        "VALID_POSITIVE",
        "LOCAL_SINGLE_GPU_PERFORMANCE",
        "paired gain is valid",
        "NONE",
      );
    },
    (dispatch) => {
      goals.push(dispatch);
      dispatch.onRuntimeEvent?.({
        type: "goal_raw_event",
        at: new Date().toISOString(),
        event: { method: "turn/diff/updated", params: { large: "raw-provider-payload" } },
      });
      dispatch.onRuntimeEvent?.({
        type: "goal_message_completed",
        at: new Date().toISOString(),
        providerTurnId: "lab-turn",
        itemId: "message-1",
        phase: "commentary",
        text: "large commentary body stored only in provider raw",
      });
      writeLabResult(dispatch, "ordinary atomic completion with paired evidence");
      return {
        goalStatus: "complete",
        finalText: "Atomic result committed.",
        providerThreadId: "lab-thread",
        providerTurnIds: ["lab-turn"],
        tokensUsed: 321,
        timeUsedSeconds: 17,
      };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.deepEqual(turns.map((item) => item.role), [
    "EXPERIMENT_DECISION",
    "EVIDENCE_JUDGE",
    "EXPERIMENT_DECISION",
  ]);
  assert.deepEqual(turns.map((item) => item.effort), ["max", "high", "max"]);
  assert.equal(goals.length, 1);
  assert.match(goals[0]!.prompt, /Stop Gate/);
  assert.match(goals[0]!.prompt, /Isolated mutable source/);
  assert.equal(goals[0]!.tokenBudget, null);
  const contract = store.readJson<{
    formatVersion: number;
    contractRevision: number;
    stopConditions: string[];
    estimatedMinutes: number;
  }>("contracts/contract-1/contract.json");
  assert.equal(contract.formatVersion, 2);
  assert.equal(contract.contractRevision, 1);
  assert.equal(contract.stopConditions.length, 2);
  assert.equal(contract.estimatedMinutes, 120);
  assert.deepEqual(readHistory(store).map((entry) => entry.kind), [
    "DECISION",
    "EXPERIMENT",
    "JUDGMENT",
    "DECISION",
  ]);
  const experiment = readHistory(store)[1]!;
  assert.equal(experiment.adoptedAfterInterruption, false);
  assert.equal((experiment.invocationRefs as string[]).length, 1);
  const invocation = store.readJson<{
    runtimeRef: string;
    providerRawRef: string;
  }>((experiment.invocationRefs as string[])[0]!);
  assert.doesNotMatch(store.readText(invocation.runtimeRef), /large commentary body/);
  assert.match(store.readText(invocation.runtimeRef), /textSha256/);
  assert.match(store.readText(invocation.providerRawRef), /large commentary body/);
  assert.match(store.readText(invocation.providerRawRef), /raw-provider-payload/);
  assert.equal(validateDirectionExperimentRun(workDir).valid, true);
});

test("RUN_JUDGE re-reviews existing design/evidence without scheduling Lab", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "review-only-run");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  const turns: TurnDispatch[] = [];
  const goals: GoalDispatch[] = [];
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      turns.push(dispatch);
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        return decisionCount === 1
          ? JSON.stringify({
            decision: "RUN_JUDGE",
            evidenceScope: "DESIGN_AUDIT_ONLY",
            reason: "the source design needs an independent baseline audit",
            experimentContract: null,
            reviewFocus: "Check whether the strongest simple baseline is specified.",
          })
          : terminal(
            "RETURN_TO_LEARNING",
            "DESIGN_AUDIT_ONLY",
            "the core claim needs a different lever",
          );
      }
      return judgment(
        "INVALID",
        "DESIGN_AUDIT_ONLY",
        "no executable paired comparison exists",
        "strong baseline missing",
      );
    },
    (dispatch) => {
      goals.push(dispatch);
      return {};
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(goals.length, 0);
  assert.deepEqual(turns.map((item) => item.role), [
    "EXPERIMENT_DECISION",
    "EVIDENCE_JUDGE",
    "EXPERIMENT_DECISION",
  ]);
  assert.match(turns[0]!.prompt, /Lab runtime envelope/);
  assert.equal(validateDirectionExperimentRun(workDir).valid, true);
});

test("Decision corrects malformed or over-envelope contracts against frozen snapshots", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "decision-retry-run");
  initializeDirectionExperimentRun({
    projectRoot,
    workDir,
    directionResultPath: fixture.directionPath,
    labHardTimeoutMs: 3_600_000,
  });
  const prompts: string[] = [];
  let calls = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    prompts.push(dispatch.prompt);
    calls += 1;
    return calls === 1
      ? decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE", 120)
      : terminal(
        "RETURN_TO_LEARNING",
        "DESIGN_AUDIT_ONLY",
        "the experiment cannot fit the available carrier honestly",
      );
  });
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(prompts.length, 2);
  assert.equal(
    extractLine(prompts[0]!, "- State snapshot:"),
    extractLine(prompts[1]!, "- State snapshot:"),
  );
  assert.match(prompts[1]!, /must not exceed current Lab envelope/);
  assert.match(store.readText("events.jsonl"), /EXPERIMENT_DECISION_OUTPUT_INVALID/);
});

test("Judge rejects scheduling-shaped output and retries only judgment protocol", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "judge-retry-run");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  let decisionCount = 0;
  let judgeCount = 0;
  const judgePrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        return decisionCount === 1
          ? decisionRunLab("WEAKENED_PROXY_MECHANISM")
          : terminal(
            "COMPLETE_REJECT",
            "WEAKENED_PROXY_MECHANISM",
            "valid equivalence removes the scoped gain",
          );
      }
      judgeCount += 1;
      judgePrompts.push(dispatch.prompt);
      return judgeCount === 1
        ? '{"decision":"RUN_LAB","reason":"old protocol"}'
        : judgment(
          "VALID_NEGATIVE",
          "WEAKENED_PROXY_MECHANISM",
          "simple baseline is equivalent",
          "NONE",
        );
    },
    (dispatch) => {
      writeLabResult(dispatch, "valid early-stop equivalence evidence");
      return { goalStatus: "complete", finalText: "result committed" };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(judgePrompts.length, 2);
  assert.match(judgePrompts[1]!, /上一次 final_answer 未通过评判协议检查/);
  assert.match(store.readText("events.jsonl"), /EVIDENCE_JUDGE_OUTPUT_INVALID/);
});

test("a prepared atomic contract pauses at cycle budget and resumes explicitly", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "budget-run");
  initializeDirectionExperimentRun({
    projectRoot,
    workDir,
    directionResultPath: fixture.directionPath,
    maxCycles: 1,
  });
  let decisionCount = 0;
  let judgeCount = 0;
  let goalCount = 0;
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        if (decisionCount <= 2) {
          return decisionRunLab(
            decisionCount === 1
              ? "WEAKENED_PROXY_MECHANISM"
              : "LOCAL_SINGLE_GPU_PERFORMANCE",
          );
        }
        return terminal(
          "COMPLETE_REJECT",
          "LOCAL_SINGLE_GPU_PERFORMANCE",
          "the local comparison is negative",
        );
      }
      judgeCount += 1;
      return judgeCount === 1
        ? judgment(
          "INCONCLUSIVE",
          "WEAKENED_PROXY_MECHANISM",
          "proxy is too narrow",
          "real local result",
        )
        : judgment(
          "VALID_NEGATIVE",
          "LOCAL_SINGLE_GPU_PERFORMANCE",
          "local paired comparison fails",
          "NONE",
        );
    },
    (dispatch) => {
      goalCount += 1;
      writeLabResult(dispatch, `cycle ${goalCount} result`);
      return { goalStatus: "complete", finalText: `cycle ${goalCount} done` };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const first = await new DirectionExperimentController(store, runtime).run();
  assert.equal(first.workflowOutcome, "PAUSED");
  assert.equal(store.readState().pauseKind, "CYCLE_BUDGET_EXHAUSTED");
  assert.equal(store.readState().cycle, 2);
  assert.equal(goalCount, 1);

  const second = await new DirectionExperimentController(store, runtime).run(true, 1);
  assert.equal(second.workflowOutcome, "FINISHED");
  assert.equal(goalCount, 2);
  assert.equal(validateDirectionExperimentRun(workDir).valid, true);
});

test("checkpoint-only timeout resumes same Goal thread in a new invocation", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "lab-resume-run");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  let decisionCount = 0;
  let goalCount = 0;
  const resumeIds: Array<string | null> = [];
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        return decisionCount === 1
          ? decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE")
          : terminal(
            "COMPLETE_SUPPORT",
            "LOCAL_SINGLE_GPU_PERFORMANCE",
            "reviewed local result is sufficient",
          );
      }
      return judgment(
        "VALID_POSITIVE",
        "LOCAL_SINGLE_GPU_PERFORMANCE",
        "valid paired result",
        "NONE",
      );
    },
    (dispatch) => {
      goalCount += 1;
      resumeIds.push(dispatch.resumeThreadId);
      if (goalCount === 1) {
        writeCheckpoint(dispatch, ["calibration-shard-01"]);
        return {
          goalStatus: "paused",
          finalText: "checkpoint saved",
          providerThreadId: "persistent-lab-thread",
          providerTurnIds: ["turn-1"],
          failureKind: "HARD_TIMEOUT",
          error: "HARD_TIMEOUT after watchdog",
        };
      }
      writeLabResult(dispatch, "resumed only missing shards and committed result");
      return {
        goalStatus: "complete",
        finalText: "experiment completed",
        providerThreadId: "persistent-lab-thread",
        providerTurnIds: ["turn-1", "turn-2"],
      };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const first = await new DirectionExperimentController(store, runtime).run();
  assert.equal(first.workflowOutcome, "PAUSED");
  assert.equal(store.readState().pauseKind, "LAB_GOAL_PAUSED");
  assert.ok(store.readState().latestCheckpointRef);

  const second = await new DirectionExperimentController(store, runtime).run(true);
  assert.equal(second.workflowOutcome, "FINISHED");
  assert.deepEqual(resumeIds, [null, "persistent-lab-thread"]);
  const experiment = readHistory(store).find((entry) => entry.kind === "EXPERIMENT")!;
  assert.equal((experiment.invocationRefs as string[]).length, 2);
  assert.match(store.readText("events.jsonl"), /LAB_GOAL_RESUMED/);
  assert.equal(validateDirectionExperimentRun(workDir).valid, true);
});

for (const providerStatus of ["paused", "blocked"] as const) {
  test(`provider ${providerStatus} with final result is adopted and sent to Judge`, async () => {
    const fixture = makeSourceFixture();
    const workDir = resolve(fixture.root, `${providerStatus}-result-run`);
    initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
    let decisionCount = 0;
    const runtime = new ScriptedTurnRuntime(
      (dispatch) => {
        if (dispatch.role === "EXPERIMENT_DECISION") {
          decisionCount += 1;
          return decisionCount === 1
            ? decisionRunLab("WEAKENED_PROXY_MECHANISM")
            : terminal(
              "COMPLETE_REJECT",
              "WEAKENED_PROXY_MECHANISM",
              "reviewed scoped negative is sufficient",
            );
        }
        return judgment(
          "VALID_NEGATIVE",
          "WEAKENED_PROXY_MECHANISM",
          "early-stop evidence is valid",
          "NONE",
        );
      },
      (dispatch) => {
        writeLabResult(dispatch, `${providerStatus} after an independently reviewable early stop`);
        return {
          goalStatus: providerStatus,
          finalText: "result committed before provider exit",
          failureKind: providerStatus === "paused" ? "HARD_TIMEOUT" : null,
          error: providerStatus === "paused" ? "watchdog" : "external block after result",
        };
      },
    );
    const store = new DirectionExperimentStore(workDir);
    const outcome = await new DirectionExperimentController(store, runtime).run();
    assert.equal(outcome.workflowOutcome, "FINISHED");
    const experiment = readHistory(store).find((entry) => entry.kind === "EXPERIMENT")!;
    assert.equal(experiment.adoptedAfterInterruption, true);
    assert.match(store.readText("events.jsonl"), /LAB_RESULT_ADOPTED_AFTER_INTERRUPTION/);
  });
}

test("03 all-100 calibration closes through result → Judge without confirmation", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "all-100-regression");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  let decisionCount = 0;
  let goalCount = 0;
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      if (dispatch.role === "EXPERIMENT_DECISION") {
        decisionCount += 1;
        return decisionCount === 1
          ? decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE")
          : terminal(
            "COMPLETE_REJECT",
            "LOCAL_SINGLE_GPU_PERFORMANCE",
            "the exact frozen selector is degenerate",
          );
      }
      return judgment(
        "VALID_NEGATIVE",
        "LOCAL_SINGLE_GPU_PERFORMANCE",
        "pi2=100% x11 and G0=100% validly hit the predeclared stop within this policy",
        "broader policy family remains outside scope",
      );
    },
    (dispatch) => {
      goalCount += 1;
      writeLabResult(
        dispatch,
        "stop condition hit: pi2 = 100% x 11; G0 = 100%; mechanism audit valid; confirmation not started",
      );
      return { goalStatus: "complete", finalText: "calibration result committed" };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(goalCount, 1);
  assert.equal(store.readState().finalDecision, "COMPLETE_REJECT");
});

test("tampered cycle binding prevents result adoption", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "binding-tamper-run");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  let judgeCalls = 0;
  const runtime = new ScriptedTurnRuntime(
    (dispatch) => {
      if (dispatch.role === "EVIDENCE_JUDGE") judgeCalls += 1;
      return decisionRunLab("WEAKENED_PROXY_MECHANISM");
    },
    (dispatch) => {
      writeLabResult(dispatch, "result whose cycle binding was tampered");
      const bindingPath = promptPath(dispatch.prompt, "- Cycle binding:");
      writeFileSync(bindingPath, `${readFileSync(bindingPath, "utf8")}\n`, "utf8");
      return { goalStatus: "complete", finalText: "done" };
    },
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "PAUSED");
  assert.equal(judgeCalls, 0);
  assert.match(outcome.reason!, /cycle binding hash mismatch/);
});

test("complete provider exit without result or checkpoint pauses abnormally", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "missing-artifact-run");
  initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
  const runtime = new ScriptedTurnRuntime(
    () => decisionRunLab("DESIGN_AUDIT_ONLY"),
    () => ({ goalStatus: "complete", finalText: "forgot files" }),
  );
  const store = new DirectionExperimentStore(workDir);
  const outcome = await new DirectionExperimentController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "PAUSED");
  assert.match(outcome.reason!, /neither an adoptable result nor a valid checkpoint/);
  assert.equal(readHistory(store).some((entry) => entry.kind === "EXPERIMENT"), false);
});

test("lock-free operator pause interrupts Lab and indexes its checkpoint", async () => {
  const fixture = makeSourceFixture();
  const workDir = resolve(fixture.root, "operator-pause-run");
  const run = initializeDirectionExperimentRun({
    projectRoot,
    workDir,
    directionResultPath: fixture.directionPath,
  });
  const store = new DirectionExperimentStore(workDir);
  store.writeJson("run.json", {
    ...run,
    budgets: { ...run.budgets, controlPollMs: 10 },
  });
  const runtime = new PauseAwareRuntime(store);
  const outcome = await new DirectionExperimentController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "PAUSED");
  assert.equal(store.readState().pauseKind, "OPERATOR_REQUESTED");
  assert.ok(store.readState().latestCheckpointRef);
  assert.equal(runtime.interrupted, true);
});

test("contract and Agent responsibilities use minimal v7 protocol", () => {
  const parsed = parseDecisionResult(decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE"), {
    maxEstimatedMinutes: 300,
  });
  assert.equal(parsed.errors.length, 0);
  assert.deepEqual(parsed.value?.experimentContract?.stopConditions, [
    "stop when the calibrated selector degenerates to one constant policy",
    "stop when the intended causal trigger never occurs under valid conditions",
  ]);
  assert.ok(parseDecisionResult(decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE", 301), {
    maxEstimatedMinutes: 300,
  }).errors.length > 0);
  assert.ok(parseDecisionResult(JSON.stringify({
    decision: "RUN_LAB",
    evidenceScope: "DESIGN_AUDIT_ONLY",
    reason: "missing stop conditions",
    experimentContract: {
      objective: "x",
      comparison: "x",
      conditions: "x",
      stopConditions: [],
      estimatedMinutes: 1,
      allowedWeakening: [],
      forbiddenWeakening: [],
      completionEvidence: "x",
    },
    reviewFocus: null,
  })).errors.length > 0);
  assert.equal(parseJudgeResult(judgment(
    "INCONCLUSIVE",
    "WEAKENED_PROXY_MECHANISM",
    "usable but narrow",
    "real task evidence",
  )).errors.length, 0);

  const lab = readFileSync(resolve(projectRoot, ".codex/skills/direction-lab-goal/SKILL.md"), "utf8");
  const judge = readFileSync(resolve(projectRoot, ".codex/skills/direction-evidence-judge/SKILL.md"), "utf8");
  const decision = readFileSync(resolve(projectRoot, ".codex/skills/direction-experiment-decision/SKILL.md"), "utf8");
  assert.match(lab, /stop conditions/i);
  assert.match(judge, /early-stop/i);
  assert.match(decision, /atomic contract/i);
  assert.doesNotMatch(judge, /RUN_LAB|COMPLETE_SUPPORT|COMPLETE_REJECT/);
});

test("formats v2-v6 are audit-only under v7", () => {
  for (const version of [2, 3, 4, 5, 6]) {
    const fixture = makeSourceFixture();
    const workDir = resolve(fixture.root, `old-v${version}`);
    initializeDirectionExperimentRun({ projectRoot, workDir, directionResultPath: fixture.directionPath });
    const store = new DirectionExperimentStore(workDir);
    const run = store.readJson<Record<string, unknown>>("run.json");
    const state = store.readJson<Record<string, unknown>>("state.json");
    store.writeJson("run.json", { ...run, formatVersion: version });
    store.writeJson("state.json", { ...state, formatVersion: version });
    const report = validateDirectionExperimentRun(workDir);
    assert.equal(report.valid, true);
    assert.match(report.warnings.join("; "), new RegExp(`format v${version} is audit-only`));
  }
});

class PauseAwareRuntime implements TurnRuntime {
  readonly store: DirectionExperimentStore;
  interrupted = false;
  private goalResolve: ((value: RawGoalResult) => void) | null = null;

  constructor(store: DirectionExperimentStore) {
    this.store = store;
  }

  async run(dispatch: TurnDispatch): Promise<RawTurnResult> {
    return turnResult(
      dispatch.role === "EXPERIMENT_DECISION"
        ? decisionRunLab("LOCAL_SINGLE_GPU_PERFORMANCE")
        : judgment("INCONCLUSIVE", "DESIGN_AUDIT_ONLY", "not reached", "x"),
    );
  }

  async runGoal(dispatch: GoalDispatch): Promise<RawGoalResult> {
    writeCheckpoint(dispatch, ["pilot-shard-01"]);
    setTimeout(() => this.store.writePauseRequest("test operator pause", "CLI"), 20);
    return await new Promise<RawGoalResult>((resolveGoal) => {
      this.goalResolve = resolveGoal;
    });
  }

  async interruptGoal(): Promise<void> {
    this.interrupted = true;
    this.goalResolve?.({
      goalStatus: "paused",
      finalText: "checkpoint saved",
      providerThreadId: "pause-thread",
      providerTurnIds: ["pause-turn"],
      tokensUsed: 0,
      timeUsedSeconds: 1,
      failureKind: "OPERATOR_INTERRUPT",
      error: "operator requested pause",
    });
  }
}

function decisionRunLab(evidenceScope: string, estimatedMinutes = 120): string {
  return JSON.stringify({
    decision: "RUN_LAB",
    evidenceScope,
    reason: "one atomic contract resolves the next decisive uncertainty",
    experimentContract: {
      objective: "Determine whether task-aware cache selection adds value beyond length-aware selection.",
      comparison: "Run parent B0, calibrated simple C_length, and M1 differing only in the cache selector.",
      conditions: "Same carrier, model, real subset, paired trace, seeds, metrics, and guards.",
      stopConditions: [
        "stop when the calibrated selector degenerates to one constant policy",
        "stop when the intended causal trigger never occurs under valid conditions",
      ],
      estimatedMinutes,
      allowedWeakening: ["smaller real model", "shorter paired trace"],
      forbiddenWeakening: ["real task labels", "cache-state causal interface"],
      completionEvidence: "For a stop, save calibration raw, mechanism trace, statistics, frozen policy, exclusions, and the narrow observation; otherwise save the same atomic-stage evidence without downstream phases.",
    },
    reviewFocus: null,
  });
}

function terminal(decision: string, evidenceScope: string, reason: string): string {
  return JSON.stringify({
    decision,
    evidenceScope,
    reason,
    experimentContract: null,
    reviewFocus: null,
  });
}

function judgment(
  assessment: string,
  evidenceScope: string,
  reason: string,
  remainingUncertainty: string,
): string {
  return JSON.stringify({ assessment, evidenceScope, reason, remainingUncertainty });
}

function writeLabResult(dispatch: GoalDispatch, observation: string): void {
  const path = promptPath(dispatch.prompt, "- Required final result:");
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, [
    "# Atomic Lab Result",
    "",
    observation,
    "",
    "- baseline/variant binding: valid",
    "- completed shards: all declared atomic-stage shards",
    "- partial excluded: none",
  ].join("\n"), "utf8");
}

function writeCheckpoint(dispatch: GoalDispatch, completedUnits: string[]): void {
  const checkpointPath = promptPath(dispatch.prompt, "- Checkpoint:");
  const bindingPath = promptPath(dispatch.prompt, "- Cycle binding:");
  const binding = JSON.parse(readFileSync(bindingPath, "utf8")) as {
    cycle: number;
    contractRevision: number;
    contractHash: string;
  };
  writeFileSync(checkpointPath, `${JSON.stringify({
    cycle: binding.cycle,
    contractRevision: binding.contractRevision,
    contractHash: binding.contractHash,
    phase: "CALIBRATION",
    completedUnits,
    validatedArtifacts: [],
    lastProgressAt: new Date().toISOString(),
    activeCommand: null,
    resumeAction: "run only missing shards",
    partialExcludedRefs: [],
  }, null, 2)}\n`, "utf8");
}

function promptPath(prompt: string, prefix: string): string {
  const line = extractLine(prompt, prefix);
  assert.notEqual(line, "MISSING");
  return line.slice(prefix.length).trim();
}

function extractLine(prompt: string, prefix: string): string {
  return prompt.split("\n").find((line) => line.startsWith(prefix)) ?? "MISSING";
}

function turnResult(text: string): RawTurnResult {
  return {
    status: "completed",
    text,
    providerThreadId: "scripted-thread",
    providerTurnId: "scripted-turn",
    usage: {
      inputTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      reasoningOutputTokens: 0,
      totalTokens: 0,
    },
    toolEvents: [],
    rawEvents: [],
    compacted: false,
    outputCapture: "COMPLETE",
    partialText: "",
    failureKind: null,
    interruptError: null,
    lastActivityAt: new Date().toISOString(),
    incrementalEventsPersisted: false,
    error: null,
    elapsedMs: 1,
  };
}

function readHistory(store: DirectionExperimentStore): Array<Record<string, unknown>> {
  return store.readText("history.jsonl").split("\n").filter(Boolean).map(
    (line) => JSON.parse(line) as Record<string, unknown>,
  );
}

function makeSourceFixture(): { root: string; directionPath: string } {
  const root = mkdtempSync(resolve(tmpdir(), "direction-source-"));
  mkdirSync(resolve(root, "objects"), { recursive: true });
  mkdirSync(resolve(root, "results"), { recursive: true });
  mkdirSync(resolve(root, "tasks/direction-task"), { recursive: true });
  mkdirSync(resolve(root, "notes"), { recursive: true });
  json(resolve(root, "run.json"), {
    formatVersion: 8,
    runId: "source-run-1",
    projectRoot: root,
    skills: { worker: { name: "fixture", sha256: "fixture" } },
  });
  writeFileSync(resolve(root, "notes/evidence.md"), "# Baseline\nEvidence.\n", "utf8");
  json(resolve(root, "results/anchor.json"), {
    workOutcome: "READY_FOR_REVIEW",
    content: {
      name: "Bound anchor",
      baseline: "P0 baseline",
      performanceTension: "Observed P99 headroom",
      scope6L: { L1: "DAG", L2: "queue", L3: null, L4: null, L5: "GPU", L6: null },
      constraints: ["same resources"],
    },
    evidence: [{ sourceRef: "notes/evidence.md#Baseline", supports: "baseline" }],
  });
  json(resolve(root, "results/direction.json"), {
    workOutcome: "READY_FOR_REVIEW",
    content: {
      name: "Cache-aware slack",
      mechanism: "Use task and cache state to select a token budget.",
      baselineChange: "Add only the cache-aware selector over the parent baseline.",
      expectedEffects: [{ metric: "P99", effect: "lower", conditions: "mixed cache" }],
      tradeoffs: ["control overhead"],
      failureConditions: ["calibrated simple policy is equivalent"],
      measurementPlan: ["calibrate, freeze, trace, and run paired ablation"],
    },
    evidence: [{ sourceRef: "notes/evidence.md#Baseline", supports: "mechanism context" }],
  });
  json(resolve(root, "results/review.json"), {
    reviewVerdict: "PASS",
    summary: "Bounded and falsifiable.",
    findings: [],
    queryGaps: [],
  });
  json(resolve(root, "tasks/direction-task/turn_task.json"), {
    action: "CREATE_DIRECTION",
    inputs: { boundAnchor: "results/anchor.json" },
  });
  json(resolve(root, "objects/index.json"), {
    revision: 2,
    activeAnchorIds: ["anchor-1"],
    anchors: {
      "anchor-1": {
        objectId: "anchor-1",
        latestRevision: 1,
        revisions: { "1": { revision: 1, workRef: "results/anchor.json" } },
        directionIds: ["direction-1"],
        rejected: false,
      },
    },
    directions: {
      "direction-1": {
        objectId: "direction-1",
        parentAnchorId: "anchor-1",
        latestRevision: 1,
        revisions: {
          "1": {
            revision: 1,
            workTaskRef: "tasks/direction-task/turn_task.json",
            workRef: "results/direction.json",
            reviewRef: "results/review.json",
            reviewVerdict: "PASS",
          },
        },
        rejected: false,
      },
    },
  });
  return { root, directionPath: resolve(root, "results/direction.json") };
}

function json(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
