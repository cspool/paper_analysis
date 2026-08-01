import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { test } from "node:test";
import { RefactoredSemanticLoopController } from "../refactor/controller.ts";
import { authorizeRuntimeRecovery } from "../refactor/recovery.ts";
import { ScriptedTurnRuntime } from "../refactor/runtime.ts";
import { initializeRun } from "../refactor/run_setup.ts";
import { FileLoopStore } from "../refactor/store.ts";
import type {
  RawTurnResult,
  TurnDispatch,
  TurnTask,
} from "../refactor/types.ts";
import { validateRun } from "../refactor/validation.ts";

const projectRoot = resolve(import.meta.dirname, "../../..");

test("partial timeout is persisted and retried from the same frozen Task", async () => {
  const workDir = newRun("partial-timeout");
  const store = new FileLoopStore(workDir);
  const workerPrompts: string[] = [];
  let firstWorker = true;
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      workerPrompts.push(dispatch.prompt);
      if (firstWorker) {
        firstWorker = false;
        const at = new Date().toISOString();
        dispatch.onRuntimeEvent?.({
          type: "provider_started",
          at,
          threadId: "partial-thread",
          providerTurnId: "partial-provider-turn",
        });
        dispatch.onRuntimeEvent?.({
          type: "output_delta",
          at,
          itemId: "partial-item",
          delta: '{"leakedPartialMarker":true,',
        });
        return timeoutResult(
          "PARTIAL",
          '{"leakedPartialMarker":true,',
        );
      }
      return JSON.stringify(workResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });

  const outcome = await new RefactoredSemanticLoopController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "FINISHED");
  const workers = store.turnRefs()
    .map((ref) => ({ ref, turn: store.readTurn(ref) }))
    .filter(({ turn }) => turn.role === "WORKER")
    .sort((a, b) => a.turn.startedAt.localeCompare(b.turn.startedAt));
  assert.equal(workers[0]!.turn.turnState, "RUNTIME_FAILED");
  assert.equal(workers[0]!.turn.outputCapture, "PARTIAL");
  assert.equal(workers[0]!.turn.rawOutputRef, null);
  assert.ok(workers[0]!.turn.partialOutputRef);
  assert.equal(
    store.readText(workers[0]!.turn.partialOutputRef!),
    '{"leakedPartialMarker":true,',
  );
  assert.equal(workers[1]!.turn.taskBindingRef, workers[0]!.turn.taskBindingRef);
  assert.equal(workers[1]!.turn.retryOf, workers[0]!.ref);
  assert.match(workerPrompts[1]!, /\[RUNTIME_RETRY\]/);
  assert.match(workerPrompts[1]!, /HARD_TIMEOUT/);
  assert.match(workerPrompts[1]!, /不得从断点续写/);
  assert.equal(
    store.readText(workers[0]!.turn.runtimeRef).includes("output_delta"),
    true,
  );
  assert.doesNotMatch(
    store.readText("observations/research_memory.json"),
    /leakedPartialMarker/,
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("a unique completed final message survives provider timeout without a retry", async () => {
  const workDir = newRun("complete-timeout");
  const store = new FileLoopStore(workDir);
  let anchorWorkerCalls = 0;
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      const task = taskFromPrompt(dispatch.prompt);
      const text = JSON.stringify(workResult(task));
      if (task.action === "CREATE_ANCHOR") {
        anchorWorkerCalls += 1;
        dispatch.onRuntimeEvent?.({
          type: "message_completed",
          at: new Date().toISOString(),
          itemId: "complete-final",
          phase: "final_answer",
          text,
        });
        return {
          ...timeoutResult("COMPLETE", text),
          text,
        };
      }
      return text;
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });

  const outcome = await new RefactoredSemanticLoopController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(anchorWorkerCalls, 1);
  const captured = store.turnRefs()
    .map((ref) => store.readTurn(ref))
    .find((turn) =>
      turn.role === "WORKER" && turn.providerStatus === "timeout"
    );
  assert.ok(captured);
  assert.equal(captured.outputCapture, "COMPLETE");
  assert.equal(captured.runtimeFailureKind, "HARD_TIMEOUT");
  assert.equal(captured.turnState, "COMMITTED");
  assert.ok(captured.rawOutputRef && store.exists(captured.rawOutputRef));
  assert.equal(validateRun(workDir).valid, true);
});

test("Decision runtime retry reuses the frozen DecisionContext", async () => {
  const workDir = newRun("decision-context-retry");
  const store = new FileLoopStore(workDir);
  const decisionPrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionPrompts.push(dispatch.prompt);
    if (decisionPrompts.length === 1) return timeoutResult("NONE", "");
    return decisionPrompts.length === 2
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });

  const outcome = await new RefactoredSemanticLoopController(store, runtime).run();
  assert.equal(outcome.workflowOutcome, "FINISHED");
  const firstPath = decisionPrompts[0]!.match(/^本次决策上下文：(.+)$/m)?.[1];
  const retryPath = decisionPrompts[1]!.match(/^本次决策上下文：(.+)$/m)?.[1];
  assert.ok(firstPath);
  assert.equal(retryPath, firstPath);
  assert.match(decisionPrompts[1]!, /\[RUNTIME_RETRY\]/);
  assert.equal(validateRun(workDir).valid, true);
});

test("runtime retry exhaustion supports one idempotent explicit recovery", async () => {
  const workDir = newRun("explicit-recovery");
  const store = new FileLoopStore(workDir);
  const failing = new ScriptedTurnRuntime(() => timeoutResult("NONE", ""));
  const failed = await new RefactoredSemanticLoopController(store, failing).run();
  assert.equal(failed.workflowOutcome, "FAILED");
  assert.equal(store.readState().failureKind, "RUNTIME_RETRY_EXHAUSTED");
  const sourceTurnRef = store.turnRefs()
    .map((ref) => ({ ref, turn: store.readTurn(ref) }))
    .filter(({ turn }) => turn.turnState === "RUNTIME_FAILED")
    .sort((a, b) => a.turn.startedAt.localeCompare(b.turn.startedAt))
    .at(-1)!.ref;
  const oldTurnBytes = store.readText(sourceTurnRef);
  const token = "operator-recovery-token-1";
  const first = authorizeRuntimeRecovery(store, {
    token,
    timeoutOverride: { hardTimeoutMs: 1_200_000 },
  });
  assert.equal(first.status, "AUTHORIZED");
  const duplicateBeforeUse = authorizeRuntimeRecovery(store, { token });
  assert.equal(duplicateBeforeUse.status, "ALREADY_AUTHORIZED");

  let decisionCount = 0;
  const succeeding = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const recovered = await new RefactoredSemanticLoopController(
    store,
    succeeding,
  ).run();
  assert.equal(recovered.workflowOutcome, "FINISHED");
  assert.equal(store.readText(sourceTurnRef), oldTurnBytes);
  const recoveryTurn = store.turnRefs()
    .map((ref) => store.readTurn(ref))
    .find((turn) => turn.recoveryRef === first.recoveryRef);
  assert.ok(recoveryTurn);
  assert.equal(recoveryTurn.retryOf, sourceTurnRef);
  assert.equal(recoveryTurn.timeoutProfile.hardTimeoutMs, 1_200_000);
  const duplicateAfterUse = authorizeRuntimeRecovery(store, { token });
  assert.equal(duplicateAfterUse.status, "ALREADY_CONSUMED");
  assert.doesNotMatch(store.readText(first.recoveryRef), new RegExp(token));
  assert.equal(validateRun(workDir).valid, true);
});

function timeoutResult(
  capture: RawTurnResult["outputCapture"],
  partialText: string,
): Partial<RawTurnResult> {
  return {
    status: "timeout",
    text: "",
    outputCapture: capture,
    partialText,
    failureKind: "HARD_TIMEOUT",
    interruptError: null,
    lastActivityAt: new Date().toISOString(),
    incrementalEventsPersisted: true,
    error: "HARD_TIMEOUT after test budget",
  };
}

function newRun(prefix: string): string {
  const workDir = mkdtempSync(resolve(tmpdir(), `simple-loop-${prefix}-`));
  initializeRun({
    projectRoot,
    workDir,
    topic: "任意性能研究 Topic",
    objective: "从本地多维知识库形成可验证的性能优化潜力",
    maxRounds: 5,
    model: "scripted-model",
  });
  return workDir;
}

function taskFromPrompt(prompt: string): TurnTask {
  const match = prompt.match(/^本次任务：(.+)$/m);
  assert.ok(match);
  return JSON.parse(readFileSync(match[1]!, "utf8")) as TurnTask;
}

function workResult(task: TurnTask): unknown {
  if (task.action.includes("ANCHOR")) {
    return {
      workOutcome: "READY_FOR_REVIEW",
      content: {
        name: "条件化执行 Anchor",
        scenario: "目标工作负载的受控在线阶段",
        baseline: "所有请求执行完整路径",
        performanceTension: "低价值工作增加尾延迟",
        scope6L: {
          L1: "条件执行",
          L2: null,
          L3: null,
          L4: null,
          L5: null,
          L6: null,
        },
        constraints: ["固定公平比较变量"],
      },
      evidence: [{ sourceRef: "idea_notes/example.md#a", supports: "baseline" }],
      unresolved: [],
    };
  }
  return {
    workOutcome: "READY_FOR_REVIEW",
    content: {
      name: "按需执行 Direction",
      mechanism: "轻量判别后激活完整路径",
      baselineChange: "从无条件执行改为一个条件门控",
      expectedEffects: [{ metric: "延迟", effect: "降低", conditions: "误判受控" }],
      tradeoffs: ["判别开销"],
      failureConditions: ["所有请求均需完整路径"],
      measurementPlan: ["先复现 baseline，再冻结条件进行配对比较"],
    },
    evidence: [{ sourceRef: "idea_notes/example.md#d", supports: "mechanism" }],
    unresolved: [],
  };
}

function passReview(): unknown {
  return {
    reviewVerdict: "PASS",
    summary: "满足当前对象合同。",
    findings: [],
    queryGaps: [],
  };
}
