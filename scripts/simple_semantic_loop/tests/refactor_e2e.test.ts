import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { test } from "node:test";
import { RefactoredSemanticLoopController } from "../refactor/controller.ts";
import { ScriptedTurnRuntime } from "../refactor/runtime.ts";
import { initializeRun } from "../refactor/run_setup.ts";
import { writeCheckpoint } from "../refactor/observations.ts";
import { FileLoopStore } from "../refactor/store.ts";
import type {
  StateFile,
  TurnDispatch,
  TurnFile,
  TurnTask,
} from "../refactor/types.ts";
import { validateRun } from "../refactor/validation.ts";
import { createWorkerBinding } from "../refactor/workflow.ts";

const projectRoot = resolve(import.meta.dirname, "../../..");

test("E2E follows Worker → Reviewer → Decision and closes Anchor plus Direction", async () => {
  const workDir = newRun("e2e");
  const dispatches: TurnDispatch[] = [];
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    dispatches.push(dispatch);
    if (dispatch.role === "DECISION") {
      decisionCount += 1;
      return decisionCount === 1
        ? "decision = RUN_WORKER\nguidance = 增加 Anchor（该 guidance 故意与机械 requirement 冲突）。"
        : "decision = FINISH_WORKFLOW";
    }
    const task = taskFromPrompt(dispatch.prompt);
    return JSON.stringify(
      dispatch.role === "WORKER" ? workerResult(task) : passReview(),
    );
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.deepEqual(outcome, {
    workflowOutcome: "FINISHED",
    reportRef: "final/report.md",
    reason: null,
  });
  assert.equal(store.readState().lifecycle, "FINISHED");
  assert.equal(
    dispatches.map((dispatch) => dispatch.role).join(","),
    "WORKER,REVIEWER,DECISION,WORKER,REVIEWER,DECISION",
  );
  assert.ok(
    dispatches
      .filter((dispatch) => dispatch.role === "DECISION")
      .every((dispatch) => dispatch.effort === "max"),
  );
  assert.equal(
    taskFromPrompt(
      dispatches.filter((dispatch) => dispatch.role === "WORKER")[1]!.prompt,
    ).action,
    "CREATE_DIRECTION",
    "T01 requirement wins; Script must not parse contradictory guidance",
  );
  assert.ok(
    dispatches
      .filter((dispatch) => dispatch.role !== "DECISION")
      .every((dispatch) => dispatch.effort === "high"),
  );
  assert.ok(
    dispatches.every((dispatch) => dispatch.outputSchema === null),
    "no Agent Turn receives a full content outputSchema",
  );
  assert.equal(store.readRun().formatVersion, 7);
  assert.equal(store.readState().formatVersion, 7);
  assert.ok(
    turns(store)
      .filter((turn) => turn.resultRef)
      .every((turn) => turn.controlRef && store.exists(turn.controlRef)),
    "every accepted Agent result has a stored core control projection",
  );
  const report = readFileSync(resolve(workDir, "final/report.md"), "utf8");
  assert.match(report, /Anchor 1/);
  assert.match(report, /Direction 1\.1/);
  assert.match(report, /idea_notes\/example\.md#baseline/);
  assert.equal(validateRun(workDir).valid, true);

  const context = latestDecisionContext(store);
  assert.deepEqual(Object.keys(context).sort(), [
    "committedResults",
    "goalRef",
    "observationRef",
    "pendingResults",
    "remainingRequirementsAfterPendingCommit",
  ]);
  const decisionTurns = turns(store).filter((turn) => turn.role === "DECISION");
  const firstContext = store.readJson<Record<string, unknown>>(
    decisionTurns[0]!.decisionContextRef!,
  );
  const observation = store.readJson<{
    stateRevision: number;
    researchMemoryRef: string;
    trajectoryRef: string;
    trajectoryTail: unknown[];
    branchEffects: Array<{
      decision: string;
      nextAction: string;
      sequence: string[];
    }>;
  }>(firstContext.observationRef as string);
  assert.match(
    observation.researchMemoryRef,
    /^contexts\/[^/]+\/research_memory_snapshot\.json$/,
  );
  assert.match(
    observation.trajectoryRef,
    /^contexts\/[^/]+\/progress_trajectory_snapshot\.jsonl$/,
  );
  const frozenMemory = store.readJson<{
    sourceStateRevision: number;
    accepted: unknown[];
  }>(observation.researchMemoryRef);
  assert.equal(frozenMemory.sourceStateRevision, observation.stateRevision);
  assert.equal(frozenMemory.accepted.length, 0);
  const frozenTrajectory = store.readJsonLines<unknown>(
    observation.trajectoryRef,
  );
  assert.deepEqual(observation.trajectoryTail, frozenTrajectory.slice(-5));
  const frozenMemoryBytes = store.readText(observation.researchMemoryRef);
  const frozenTrajectoryBytes = store.readText(observation.trajectoryRef);
  writeCheckpoint(store, store.readState(), "snapshot immutability check");
  assert.equal(store.readText(observation.researchMemoryRef), frozenMemoryBytes);
  assert.equal(store.readText(observation.trajectoryRef), frozenTrajectoryBytes);
  assert.deepEqual(
    observation.branchEffects.find((effect) => effect.decision === "RUN_WORKER"),
    {
      decision: "RUN_WORKER",
      nextRole: "WORKER",
      nextAction: "CREATE_DIRECTION",
      targetRef: turns(store)
        .find((turn) => turn.role === "WORKER")!.resultRef,
      sequence: ["WORKER", "REVIEWER", "DECISION"],
    },
  );
  const memory = store.readJson<{
    accepted: Array<{ objectKind: string; workRef: string; reviewRef: string }>;
  }>("observations/research_memory.json");
  assert.equal(memory.accepted.filter((item) => item.objectKind === "ANCHOR").length, 1);
  assert.equal(memory.accepted.filter((item) => item.objectKind === "DIRECTION").length, 1);
  const trajectory = store.readJsonLines<{ decisionTurnRef: string }>(
    "observations/progress_trajectory.jsonl",
  );
  assert.equal(trajectory.length, 2);
  assert.equal(new Set(trajectory.map((item) => item.decisionTurnRef)).size, 2);
  assert.equal(store.exists("observations/checkpoints/round-2.md"), true);
  assert.equal(validateRun(workDir).valid, true);
});

test("RUN_REVIEWER branch is Reviewer → Worker → Reviewer → Decision", async () => {
  const workDir = newRun("reviewer-branch");
  const dispatches: TurnDispatch[] = [];
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    dispatches.push(dispatch);
    if (dispatch.role === "DECISION") {
      decisionCount += 1;
      if (decisionCount === 1) return "decision = RUN_WORKER";
      if (decisionCount === 2) {
        return "decision = RUN_REVIEWER\nguidance = 从新的反例角度审阅。";
      }
      return "decision = FINISH_WORKFLOW";
    }
    const task = taskFromPrompt(dispatch.prompt);
    return JSON.stringify(
      dispatch.role === "WORKER" ? workerResult(task) : passReview(),
    );
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(
    dispatches.map((dispatch) => dispatch.role).join(","),
    [
      "WORKER",
      "REVIEWER",
      "DECISION",
      "WORKER",
      "REVIEWER",
      "DECISION",
      "REVIEWER",
      "WORKER",
      "REVIEWER",
      "DECISION",
    ].join(","),
  );
  const actions = dispatches
    .filter((dispatch) => dispatch.role !== "DECISION")
    .map((dispatch) => taskFromPrompt(dispatch.prompt).action);
  assert.deepEqual(actions, [
    "CREATE_ANCHOR",
    "REVIEW_ANCHOR",
    "CREATE_DIRECTION",
    "REVIEW_DIRECTION",
    "REVIEW_DIRECTION",
    "DEEPEN_DIRECTION",
    "REVIEW_DIRECTION",
  ]);
  const decisionTurns = turns(store).filter((turn) => turn.role === "DECISION");
  const branchContext = store.readJson<Record<string, unknown>>(
    decisionTurns[1]!.decisionContextRef!,
  );
  const observation = store.readJson<{
    branchEffects: Array<{
      decision: string;
      targetRef: string | null;
    }>;
  }>(branchContext.observationRef as string);
  const preview = observation.branchEffects.find(
    (effect) => effect.decision === "RUN_REVIEWER",
  );
  const preReviewTask = taskFromPrompt(
    dispatches.filter((dispatch) => dispatch.role === "REVIEWER")[2]!.prompt,
  );
  const revisedPairTask = taskFromPrompt(
    dispatches.filter((dispatch) => dispatch.role === "REVIEWER")[3]!.prompt,
  );
  const reviewerTurns = turns(store).filter(
    (turn) => turn.role === "REVIEWER" && turn.resultRef,
  );
  assert.equal(
    revisedPairTask.inputs.previousReview,
    reviewerTurns[2]!.resultRef,
  );
  assert.equal(preview?.targetRef, preReviewTask.inputs.reviewTarget);
  assert.equal(validateRun(workDir).valid, true);
});

test("pre-review REJECT becomes canonical and Worker creates a reviewed replacement", async () => {
  const workDir = newRun("pre-review-reject");
  const dispatches: TurnDispatch[] = [];
  let decisionCalls = 0;
  let reviewerCalls = 0;
  let directionWorkerCalls = 0;
  let rejectedReviewRef: string | null = null;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    dispatches.push(dispatch);
    if (dispatch.role === "DECISION") {
      decisionCalls += 1;
      if (decisionCalls === 1) return "decision = RUN_WORKER";
      if (decisionCalls === 2) return "decision = RUN_REVIEWER";
      return "decision = FINISH_WORKFLOW";
    }
    const task = taskFromPrompt(dispatch.prompt);
    if (dispatch.role === "WORKER") {
      const result = workerResult(task) as {
        content: { name: string };
      };
      if (
        task.action === "CREATE_DIRECTION" ||
        task.action === "DEEPEN_DIRECTION"
      ) {
        directionWorkerCalls += 1;
        result.content.name =
          directionWorkerCalls === 1 ? "旧 Direction" : "替代 Direction";
      }
      return JSON.stringify(result);
    }
    reviewerCalls += 1;
    if (reviewerCalls === 3) {
      return JSON.stringify({
        reviewVerdict: "REJECT",
        summary: "该 Direction 的机制与 Anchor 不相容，不能在同一对象内修复。",
        findings: [
          {
            severity: "BLOCKING",
            issue: "机制离开绑定 Anchor",
            basis: "reviewTarget 的主要 baseline change 不作用于 boundAnchor",
            expected: "在同一 Anchor 内创建替代 Direction",
          },
        ],
        queryGaps: [],
      });
    }
    return JSON.stringify(passReview());
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  const contentTasks = dispatches
    .filter((dispatch) => dispatch.role !== "DECISION")
    .map((dispatch) => taskFromPrompt(dispatch.prompt));
  assert.deepEqual(
    contentTasks.map((task) => task.action),
    [
      "CREATE_ANCHOR",
      "REVIEW_ANCHOR",
      "CREATE_DIRECTION",
      "REVIEW_DIRECTION",
      "REVIEW_DIRECTION",
      "CREATE_DIRECTION",
      "REVIEW_DIRECTION",
    ],
  );

  const reviewerTurns = turns(store).filter(
    (turn) => turn.role === "REVIEWER" && turn.resultRef,
  );
  rejectedReviewRef = reviewerTurns[2]!.resultRef;
  const replacementTask = contentTasks[5]!;
  assert.equal(replacementTask.inputs.latestReview, rejectedReviewRef);

  const directions = Object.values(store.readObjects().directions);
  assert.equal(directions.length, 2);
  assert.equal(directions.filter((direction) => direction.rejected).length, 1);
  const rejected = directions.find((direction) => direction.rejected)!;
  const rejectedRevision =
    rejected.revisions[String(rejected.latestRevision)]!;
  assert.equal(replacementTask.inputs.currentWork, rejectedRevision.workRef);
  assert.equal(rejectedRevision.reviewRef, rejectedReviewRef);
  assert.equal(rejectedRevision.reviewVerdict, "REJECT");

  const report = store.readText("final/report.md");
  assert.doesNotMatch(report, /旧 Direction/);
  assert.match(report, /替代 Direction/);
  assert.match(store.readText("events.jsonl"), /PRE_REVIEW_COMMITTED/);
  assert.equal(validateRun(workDir).valid, true);
});

test("D01 keeps the latest committed REJECT result visible to Decision", async () => {
  const workDir = newRun("rejected-history");
  let reviewerCalls = 0;
  let decisionCalls = 0;
  const decisionContexts: Array<Record<string, unknown>> = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") {
      reviewerCalls += 1;
      if (reviewerCalls === 1) {
        return JSON.stringify({
          reviewVerdict: "REJECT",
          summary: "该对象不能在原绑定内修复。",
          findings: [
            {
              severity: "BLOCKING",
              issue: "Anchor 静默离开用户 Topic",
              basis: "scenario 与 workflow_goal.topic 不相交",
              expected: "丢弃该对象并创建新的 Anchor",
            },
          ],
          queryGaps: [],
        });
      }
      return JSON.stringify(passReview());
    }
    decisionCalls += 1;
    decisionContexts.push(decisionContextFromPrompt(dispatch.prompt));
    return decisionCalls < 3
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  const secondContext = decisionContexts[1] as {
    committedResults: Array<{ review: string }>;
  };
  assert.equal(secondContext.committedResults.length, 1);
  assert.equal(
    store.readJson<{ reviewVerdict: string }>(
      secondContext.committedResults[0]!.review,
    ).reviewVerdict,
    "REJECT",
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("resume can authorize multiple additional rounds in one lease", async () => {
  const workDir = newRun("round-pause", 1);
  let decisionCalls = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      const task = taskFromPrompt(dispatch.prompt);
      if (
        task.action === "CREATE_ANCHOR" &&
        decisionCalls >= 2
      ) {
        return JSON.stringify({
          workOutcome: "BLOCKED_NO_RESULT",
          content: null,
          evidence: [],
          unresolved: ["有界扩展未发现非重复且有证据支持的新 Anchor。"],
        });
      }
      return JSON.stringify(workerResult(task));
    }
    if (dispatch.role === "REVIEWER") {
      const task = taskFromPrompt(dispatch.prompt);
      const target = store.readJson<{ workOutcome: string }>(
        task.inputs.reviewTarget!,
      );
      return JSON.stringify(
        target.workOutcome === "BLOCKED_NO_RESULT"
          ? {
            reviewVerdict: "REJECT",
            summary: "有界探测可信地未形成新对象。",
            findings: [{
              severity: "BLOCKING",
              issue: "没有新 Anchor 可提交",
              basis: "候选均重复或缺少支持",
              expected: "保留为负面收敛证据",
            }],
            queryGaps: [],
          }
          : passReview(),
      );
    }
    decisionCalls += 1;
    return decisionCalls < 3
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const firstOutcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(firstOutcome.workflowOutcome, "PAUSED");
  const paused = store.readState();
  assert.equal(paused.round, 2);
  assert.equal(paused.pauseKind, "ROUND_BUDGET_EXHAUSTED");
  assert.equal(paused.roundBudget?.authorizedThroughRound, 1);
  assert.deepEqual(
    paused.sequence.map((step) => step.role),
    ["WORKER", "REVIEWER", "DECISION"],
  );
  assert.equal(paused.pending, null);
  assert.equal(store.exists("rounds/2.json"), true);
  assert.equal(validateRun(workDir).valid, true);

  const resumedOutcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run(true, 2);
  assert.equal(resumedOutcome.workflowOutcome, "FINISHED");
  const finished = store.readState();
  assert.equal(finished.round, 3);
  assert.equal(finished.roundBudget?.authorizedThroughRound, 3);
  assert.match(
    finished.roundBudget?.lastAuthorizationRef ?? "",
    /^authorizations\/rounds\/round-authorization-/,
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("existing literals preserve a quiet Anchor expansion as semantic evidence", async () => {
  const workDir = newRun("quiet-expansion");
  let decisionCalls = 0;
  const actions: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "DECISION") {
      decisionCalls += 1;
      return decisionCalls < 3
        ? "decision = RUN_WORKER"
        : "decision = FINISH_WORKFLOW";
    }
    const task = taskFromPrompt(dispatch.prompt);
    actions.push(task.action);
    if (dispatch.role === "WORKER") {
      if (actions.filter((action) => action === "CREATE_ANCHOR").length > 1) {
        return JSON.stringify({
          workOutcome: "BLOCKED_NO_RESULT",
          content: null,
          evidence: [],
          unresolved: [
            "受限检索覆盖现有 Anchor 之外的候选区域；可见路线均与已接受对象重复或缺少来源支撑，不能诚实形成新 Anchor。",
          ],
        });
      }
      return JSON.stringify(workerResult(task));
    }
    const target = store.readJson<{ workOutcome: string }>(
      task.inputs.reviewTarget!,
    );
    if (target.workOutcome === "BLOCKED_NO_RESULT") {
      return JSON.stringify({
        reviewVerdict: "REJECT",
        summary: "本轮没有可提交的新 Anchor；负面结论只供 Decision 判断信息增益。",
        findings: [
          {
            severity: "BLOCKING",
            issue: "没有形成可提交的 Anchor 对象",
            basis: "受限搜索记录显示候选重复或无来源支撑",
            expected: "不提交该候选；由 Decision 结合全局 Goal 判断下一分支",
          },
        ],
        queryGaps: [],
      });
    }
    return JSON.stringify(passReview());
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.deepEqual(actions, [
    "CREATE_ANCHOR",
    "REVIEW_ANCHOR",
    "CREATE_DIRECTION",
    "REVIEW_DIRECTION",
    "CREATE_ANCHOR",
    "REVIEW_ANCHOR",
  ]);
  const convergenceTask = turns(store)
    .filter((turn) => turn.role === "WORKER")
    .map((turn) => store.readJson<TurnTask>(
      store.readJson<{ taskRef: string }>(turn.taskBindingRef!).taskRef,
    ))
    .find((task) =>
      task.action === "CREATE_ANCHOR" && Boolean(task.inputs.researchMemory)
    );
  assert.ok(convergenceTask?.inputs.researchMemory);
  assert.match(convergenceTask.objective, /收敛探测/);
  assert.equal(store.exists(convergenceTask.inputs.researchMemory), true);
  const index = store.readObjects();
  assert.equal(Object.keys(index.anchors).length, 2);
  assert.equal(Object.values(index.anchors).filter((item) => item.rejected).length, 1);
  assert.equal(index.activeAnchorIds.length, 1);
  assert.equal(validateRun(workDir).valid, true);
});

test("premature FINISH_WORKFLOW is rejected by the mechanical allowed set", async () => {
  const workDir = newRun("premature-finish");
  const decisionPrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionPrompts.push(dispatch.prompt);
    return "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FAILED");
  assert.match(outcome.reason!, /output-correction retry budget/);
  assert.equal(Object.keys(store.readObjects().anchors).length, 0);
  assert.ok(
    decisionPrompts.every(
      (prompt) =>
        !prompt.match(
          /\[ALLOWED_DECISIONS\][\s\S]*-\s+FINISH_WORKFLOW(?:\n|$)/,
        ),
    ),
  );
  const decisions = turns(store).filter((turn) => turn.role === "DECISION");
  assert.equal(decisions.length, 3);
  assert.ok(decisions.every((turn) => turn.turnState === "INVALID_OUTPUT"));
  assert.equal(validateRun(workDir).valid, true);
});

test("JSON and missing core-control errors retry the same Task with E01", async () => {
  const workDir = newRun("output-correction");
  let workerCalls = 0;
  let decisionCount = 0;
  const workerPrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      workerCalls += 1;
      workerPrompts.push(dispatch.prompt);
      if (workerCalls === 1) return '{"workOutcome":';
      if (workerCalls === 2) return '{"content":{}}';
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") {
      return JSON.stringify(passReview());
    }
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.match(workerPrompts[1]!, /\[OUTPUT_CORRECTION\]/);
  assert.match(workerPrompts[2]!, /\[OUTPUT_CORRECTION\]/);
  assert.match(workerPrompts[2]!, /output_error_report\.json/);
  assert.match(workerPrompts[2]!, /work-result-anchor-v2/);
  const workerTurns = turns(store).filter((turn) => turn.role === "WORKER");
  assert.equal(workerTurns[0]!.turnState, "INVALID_OUTPUT");
  assert.equal(workerTurns[1]!.turnState, "INVALID_OUTPUT");
  assert.equal(workerTurns[2]!.attempt, 3);
  assert.equal(
    workerTurns[0]!.taskBindingRef,
    workerTurns[1]!.taskBindingRef,
  );
  assert.equal(
    workerTurns[1]!.taskBindingRef,
    workerTurns[2]!.taskBindingRef,
  );
  const parseError = store.readJson<{
    errors: Array<{ check: string }>;
  }>(
    workerTurns[0]!.errorReportRef!,
  );
  const coreError = store.readJson<{
    errors: Array<{ check: string; path: string }>;
  }>(
    workerTurns[1]!.errorReportRef!,
  );
  assert.equal(parseError.errors[0]!.check, "JSON_PARSE");
  assert.deepEqual(coreError.errors[0], {
    check: "CORE_CONTROL",
    path: "/workOutcome",
    message:
      "expected one of READY_FOR_REVIEW, PARTIAL_RESULT, BLOCKED_NO_RESULT",
  });
  assert.equal(validateRun(workDir).valid, true);
});

test("PARTIAL plus false PASS reaches Decision and uses semantic retries", async () => {
  const workDir = newRun("partial-semantic-retry");
  let workerCalls = 0;
  let reviewerCalls = 0;
  let decisionCalls = 0;
  const reviewerPrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      workerCalls += 1;
      const task = taskFromPrompt(dispatch.prompt);
      if (workerCalls === 1) {
        const partial = workerResult(task) as {
          workOutcome: string;
          unresolved: string[];
        };
        return JSON.stringify({
          ...partial,
          workOutcome: "PARTIAL_RESULT",
          unresolved: ["尚未覆盖一项 Task requirement"],
        });
      }
      return JSON.stringify(workerResult(task));
    }
    if (dispatch.role === "REVIEWER") {
      reviewerCalls += 1;
      reviewerPrompts.push(dispatch.prompt);
      if (reviewerCalls === 1) {
        return JSON.stringify({
          reviewVerdict: "PASS",
          summary: "错误接受未完成结果",
          findings: [],
          queryGaps: [],
        });
      }
      if (reviewerCalls === 2) {
        return JSON.stringify({
          reviewVerdict: "REVISE",
          summary: "当前结果仍缺少 Task requirement。",
          findings: [
            {
              severity: "BLOCKING",
              issue: "Work Result 明确标记为 PARTIAL_RESULT",
              basis: "reviewTarget.unresolved 仍有未完成 requirement",
              expected: "按同一 TaskBinding 深化当前 Anchor",
            },
          ],
          queryGaps: [],
        });
      }
      return JSON.stringify(passReview());
    }
    decisionCalls += 1;
    if (decisionCalls === 1) {
      return [
        "decision = RETRY_REVIEWER",
        "guidance = Reviewer 对明确的 PARTIAL_RESULT 给出 PASS，漏审未完成 requirement；重新审阅同一结果。",
      ].join("\n");
    }
    if (decisionCalls === 2) {
      return [
        "decision = RETRY_WORKER",
        "guidance = 当前 Worker 诚实报告未覆盖完整任务，重做同一任务以补齐 Anchor。",
      ].join("\n");
    }
    if (decisionCalls === 3) return "decision = RUN_WORKER";
    return "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.ok(
    reviewerPrompts.every((prompt) => !prompt.includes("[OUTPUT_CORRECTION]")),
    "semantic errors do not enter Script output correction",
  );
  const reviewerTurns = turns(store).filter((turn) => turn.role === "REVIEWER");
  assert.equal(reviewerTurns[0]!.turnState, "SUPERSEDED_BY_RETRY");
  assert.equal(
    reviewerTurns[0]!.taskBindingRef,
    reviewerTurns[1]!.taskBindingRef,
  );
  assert.equal(reviewerTurns[0]!.errorReportRef, null);
  assert.equal(reviewerTurns[1]!.turnState, "SUPERSEDED_BY_RETRY");
  const firstReview = store.readJson<{ reviewVerdict: string }>(
    reviewerTurns[0]!.resultRef!,
  );
  assert.equal(firstReview.reviewVerdict, "PASS");
  assert.equal(validateRun(workDir).valid, true);
});

test("non-standard core-valid results reach Decision and render losslessly", async () => {
  const workDir = newRun("non-standard-results");
  let workerCalls = 0;
  let reviewerCalls = 0;
  let decisionCalls = 0;
  const dispatches: TurnDispatch[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    dispatches.push(dispatch);
    if (dispatch.role === "WORKER") {
      workerCalls += 1;
      return workerCalls === 1
        ? JSON.stringify({
          workOutcome: "READY_FOR_REVIEW",
          content: {
            name: "非标准 Anchor",
            scenario: "可理解的场景",
            customObservation: "模板外但需要保真的 Anchor 正文",
          },
          evidence: "来源已在正文中说明",
          extraTopLevel: { retained: true },
        })
        : JSON.stringify({
          workOutcome: "READY_FOR_REVIEW",
          content: {
            name: "非标准 Direction",
            mechanism: "可理解的因果机制",
            novelMeasurement: ["p99", "goodput"],
          },
          notes: "没有模板要求的可选正文数组",
        });
    }
    if (dispatch.role === "REVIEWER") {
      reviewerCalls += 1;
      return reviewerCalls === 1
        ? JSON.stringify({
          reviewVerdict: "PASS",
          semanticNote: "已理解非标准正文并判断可进入结果",
        })
        : JSON.stringify({
          reviewVerdict: "PASS",
          findings: "非标准但可理解",
        });
    }
    decisionCalls += 1;
    return decisionCalls === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.ok(dispatches.every((dispatch) => dispatch.outputSchema === null));
  const report = store.readText("final/report.md");
  assert.match(report, /非标准 Anchor/);
  assert.match(report, /非标准 Direction/);
  assert.match(report, /Raw non-standard Agent results/);
  assert.match(report, /customObservation/);
  assert.match(report, /novelMeasurement/);
  assert.match(report, /semanticNote/);
  const validation = validateRun(workDir);
  assert.equal(validation.valid, true);
  assert.ok(validation.advisories.length > 0);
  const acceptedTurns = turns(store).filter((turn) => turn.resultRef);
  assert.ok(
    acceptedTurns.every(
      (turn) =>
        turn.controlRef &&
        store.exists(turn.controlRef) &&
        store.readJson<{ role: string }>(turn.controlRef).role === turn.role,
    ),
  );
});

test("Decision RETRY_WORKER supersedes the pair and reuses the Worker TaskBinding", async () => {
  const workDir = newRun("semantic-worker-retry");
  let decisionCount = 0;
  let anchorWorkerCalls = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      const task = taskFromPrompt(dispatch.prompt);
      if (task.action === "CREATE_ANCHOR") anchorWorkerCalls += 1;
      return JSON.stringify(workerResult(task));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    if (decisionCount === 1) {
      return [
        "decision = RETRY_WORKER",
        "guidance = 当前 Anchor 虽结构合法但静默改变了任务范围，无法闭合需求；按 work-result-anchor-v2 重做同一任务。",
      ].join("\n");
    }
    if (decisionCount === 2) return "decision = RUN_WORKER";
    return "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(anchorWorkerCalls, 2);
  const anchorWorkers = turns(store)
    .filter((turn) => turn.role === "WORKER")
    .filter((turn) => {
      const binding = store.readJson<{ action: string }>(
        turn.taskBindingRef!,
      );
      return binding.action === "CREATE_ANCHOR";
    });
  assert.equal(anchorWorkers.length, 2);
  assert.equal(anchorWorkers[0]!.turnState, "SUPERSEDED_BY_RETRY");
  assert.equal(anchorWorkers[0]!.taskBindingRef, anchorWorkers[1]!.taskBindingRef);
  assert.equal(validateRun(workDir).valid, true);
});

test("Decision protocol correction reuses the frozen D01", async () => {
  const workDir = newRun("decision-correction");
  let decisionCalls = 0;
  const decisionPrompts: string[] = [];
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCalls += 1;
    decisionPrompts.push(dispatch.prompt);
    if (decisionCalls === 1) return "I think RUN_WORKER is best.";
    if (decisionCalls === 2) return "decision = RUN_WORKER";
    return "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.match(decisionPrompts[1]!, /\[OUTPUT_CORRECTION\]/);
  assert.match(decisionPrompts[1]!, /decision-line-protocol-v1/);
  const decisions = turns(store).filter((turn) => turn.role === "DECISION");
  assert.equal(decisions[0]!.turnState, "INVALID_OUTPUT");
  assert.equal(
    decisions[0]!.decisionContextRef,
    decisions[1]!.decisionContextRef,
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("Decision RETRY_REVIEWER retains Worker result and reuses Reviewer TaskBinding", async () => {
  const workDir = newRun("semantic-reviewer-retry");
  let decisionCount = 0;
  let anchorWorkerCalls = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      const task = taskFromPrompt(dispatch.prompt);
      if (task.action === "CREATE_ANCHOR") anchorWorkerCalls += 1;
      return JSON.stringify(workerResult(task));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    if (decisionCount === 1) {
      return [
        "decision = RETRY_REVIEWER",
        "guidance = 最近 Reviewer 的 PASS 与其语义结论冲突，会错误关闭审阅要求；按 review-result-v2 重审同一 Work Result。",
      ].join("\n");
    }
    if (decisionCount === 2) return "decision = RUN_WORKER";
    return "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(anchorWorkerCalls, 1);
  const firstRound = turns(store).filter((turn) => turn.round === 1);
  const workers = firstRound.filter((turn) => turn.role === "WORKER");
  const reviewers = firstRound.filter((turn) => turn.role === "REVIEWER");
  assert.equal(workers.length, 1);
  assert.equal(workers[0]!.turnState, "COMMITTED");
  assert.equal(reviewers.length, 2);
  assert.equal(reviewers[0]!.turnState, "SUPERSEDED_BY_RETRY");
  assert.equal(
    reviewers[0]!.taskBindingRef,
    reviewers[1]!.taskBindingRef,
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("semantic retry budget exhaustion fails closed without committing pending results", async () => {
  const workDir = newRun("semantic-budget");
  const store = new FileLoopStore(workDir);
  const run = store.readRun();
  store.writeJson("run.json", {
    ...run,
    budgets: {
      ...run.budgets,
      maxSemanticRetries: 0,
    },
  });
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") {
      return JSON.stringify(passReview());
    }
    return [
      "decision = RETRY_WORKER",
      "guidance = 当前 W01 静默改变 Goal，提交会错误闭合需求；按 work-result-anchor-v2 重做同一任务。",
    ].join("\n");
  });
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FAILED");
  assert.match(outcome.reason!, /semantic retry budget exhausted/);
  assert.equal(Object.keys(store.readObjects().anchors).length, 0);
  assert.ok(store.readState().pending);
  assert.equal(
    turns(store).find((turn) => turn.role === "WORKER")!.turnState,
    "PENDING_DECISION",
  );
  assert.equal(validateRun(workDir).valid, true);
});

test("Provider invalid-request errors fail immediately without runtime retries", async () => {
  const workDir = newRun("provider-invalid-request");
  const store = new FileLoopStore(workDir);
  let calls = 0;
  const runtime = new ScriptedTurnRuntime(() => {
    calls += 1;
    return {
      status: "failed",
      text: "",
      error: JSON.stringify({
        type: "error",
        error: {
          type: "invalid_request_error",
          code: "invalid_request",
          message: "unsupported provider request parameter",
        },
        status: 400,
      }),
    };
  });
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FAILED");
  assert.equal(calls, 1);
  assert.match(outcome.reason!, /non-retriable runtime failure/);
  assert.match(outcome.reason!, /invalid_request/);
  assert.match(outcome.reason!, /unsupported provider request parameter/);
  assert.equal(turns(store).length, 1);
  assert.equal(turns(store)[0]!.turnState, "RUNTIME_FAILED");
  assert.equal(validateRun(workDir).valid, true);
});

test("captured Worker output is replayed after interruption without another provider call", async () => {
  const workDir = newRun("captured-output");
  const store = new FileLoopStore(workDir);
  const prepared = prepareInterruptedWorker(store, true);
  let workerCalls = 0;
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      workerCalls += 1;
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(workerCalls, 1, "only the later Direction calls the provider");
  assert.equal(store.readTurn(prepared.turnRef).turnState, "COMMITTED");
  assert.match(store.readText("events.jsonl"), /CAPTURED_TURN_REPLAYED/);
  assert.equal(validateRun(workDir).valid, true);
});

test("validated Worker output is replayed when interruption precedes state consumption", async () => {
  const workDir = newRun("validated-output");
  const store = new FileLoopStore(workDir);
  const prepared = prepareInterruptedWorker(store, true);
  store.mutateTurn(prepared.turnRef, (turn) => ({
    ...turn,
    turnState: "PENDING_DECISION",
    completedAt: new Date().toISOString(),
  }));
  let workerCalls = 0;
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      workerCalls += 1;
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(workerCalls, 1, "only the later Direction calls the provider");
  assert.equal(store.readTurn(prepared.turnRef).turnState, "COMMITTED");
  assert.match(store.readText("events.jsonl"), /CAPTURED_TURN_REPLAYED/);
  assert.equal(validateRun(workDir).valid, true);
});

test("interrupted Worker without captured output starts a linked fresh Attempt", async () => {
  const workDir = newRun("interrupted-no-output");
  const store = new FileLoopStore(workDir);
  const prepared = prepareInterruptedWorker(store, false);
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();

  assert.equal(outcome.workflowOutcome, "FINISHED");
  assert.equal(store.readTurn(prepared.turnRef).turnState, "RUNTIME_FAILED");
  const replacement = turns(store).find(
    (turn) =>
      turn.role === "WORKER" &&
      turn.taskBindingRef === prepared.bindingRef &&
      turn.turnId !== prepared.turnId,
  );
  assert.ok(replacement);
  assert.equal(replacement.attempt, 2);
  assert.equal(replacement.retryOf, prepared.turnRef);
  assert.match(store.readText("events.jsonl"), /TURN_INTERRUPTED_WITHOUT_OUTPUT/);
  assert.equal(validateRun(workDir).valid, true);
});

test("frozen Decision snapshots reject overwrite and revision drift", async () => {
  const workDir = newRun("snapshot-validation");
  let decisionCount = 0;
  const runtime = new ScriptedTurnRuntime((dispatch) => {
    if (dispatch.role === "WORKER") {
      return JSON.stringify(workerResult(taskFromPrompt(dispatch.prompt)));
    }
    if (dispatch.role === "REVIEWER") return JSON.stringify(passReview());
    decisionCount += 1;
    return decisionCount === 1
      ? "decision = RUN_WORKER"
      : "decision = FINISH_WORKFLOW";
  });
  const store = new FileLoopStore(workDir);
  const outcome = await new RefactoredSemanticLoopController(
    store,
    runtime,
  ).run();
  assert.equal(outcome.workflowOutcome, "FINISHED");

  const decisionTurn = turns(store).find((turn) => turn.role === "DECISION")!;
  const context = store.readJson<{ observationRef: string }>(
    decisionTurn.decisionContextRef!,
  );
  const observation = store.readJson<{
    stateRevision: number;
    researchMemoryRef: string;
  }>(context.observationRef);
  const original = store.readText(observation.researchMemoryRef);
  store.writeImmutableText(observation.researchMemoryRef, original);
  assert.throws(
    () => store.writeImmutableText(observation.researchMemoryRef, "{}\n"),
    /immutable file already exists with different content/,
  );
  assert.throws(
    () => store.writeText(observation.researchMemoryRef, original),
    /immutable DecisionContext file cannot be overwritten/,
  );

  const memory = store.readJson<Record<string, unknown>>(
    observation.researchMemoryRef,
  );
  writeFileSync(
    store.absolute(observation.researchMemoryRef),
    `${JSON.stringify({
      ...memory,
      sourceStateRevision: observation.stateRevision + 1,
    }, null, 2)}\n`,
    "utf8",
  );
  const report = validateRun(workDir);
  assert.equal(report.valid, false);
  assert.match(
    report.checks
      .flatMap((check) => check.details)
      .join("\n"),
    /memory sourceStateRevision .* differs from observation/,
  );
});

test("v5 runs remain valid for read-only audit but cannot be resumed", async () => {
  const workDir = newRun("legacy-format");
  const store = new FileLoopStore(workDir);
  const run = store.readRun();
  const state = store.readState();
  store.writeJson("run.json", { ...run, formatVersion: 5 });
  store.writeJson("state.json", { ...state, formatVersion: 5 });
  store.appendJsonLine("turns/legacy/runtime.jsonl", {
    type: "raw_event",
    event: {
      method: "item/agentMessage/delta",
      params: { itemId: "legacy-message", delta: "x" },
    },
  });
  store.appendJsonLine("turns/legacy/runtime.jsonl", {
    type: "output_delta",
    itemId: "legacy-message",
    delta: "x",
  });
  assert.equal(validateRun(workDir).valid, true);
  let calls = 0;
  const runtime = new ScriptedTurnRuntime(() => {
    calls += 1;
    return "";
  });

  await assert.rejects(
    new RefactoredSemanticLoopController(store, runtime).run(),
    /formatVersion 5 is read-only/,
  );
  assert.equal(calls, 0);
  assert.equal(store.exists(".run.lock"), false);
});

test("v6 runs remain valid for frozen-snapshot audit but are read-only", async () => {
  const workDir = newRun("v6-read-only");
  const store = new FileLoopStore(workDir);
  const run = store.readRun();
  const state = store.readState();
  store.writeJson("run.json", { ...run, formatVersion: 6 });
  store.writeJson("state.json", { ...state, formatVersion: 6 });
  assert.equal(validateRun(workDir).valid, true);

  await assert.rejects(
    new RefactoredSemanticLoopController(
      store,
      new ScriptedTurnRuntime(() => ""),
    ).run(),
    /formatVersion 6 is read-only/,
  );
  assert.equal(store.exists(".run.lock"), false);
});

function newRun(prefix: string, maxRounds = 5): string {
  const workDir = mkdtempSync(resolve(tmpdir(), `simple-loop-${prefix}-`));
  initializeRun({
    projectRoot,
    workDir,
    topic: "任意性能研究 Topic",
    objective: "从本地多维知识库形成可验证的性能优化潜力",
    maxRounds,
    model: "scripted-model",
  });
  return workDir;
}

function taskFromPrompt(prompt: string): TurnTask {
  const match = prompt.match(/^本次任务：(.+)$/m);
  assert.ok(match, "content Prompt contains the absolute T01 path");
  return JSON.parse(readFileSync(match[1]!, "utf8")) as TurnTask;
}

function decisionContextFromPrompt(prompt: string): Record<string, unknown> {
  const match = prompt.match(/^本次决策上下文：(.+)$/m);
  assert.ok(match, "Decision Prompt contains the absolute D01 path");
  return JSON.parse(readFileSync(match[1]!, "utf8")) as Record<string, unknown>;
}

function workerResult(task: TurnTask): unknown {
  if (task.action === "CREATE_ANCHOR" || task.action === "DEEPEN_ANCHOR") {
    return {
      workOutcome: "READY_FOR_REVIEW",
      content: {
        name: "条件化冗余执行 Anchor",
        scenario: "目标 Topic 的在线工作负载、关键执行阶段和受控运行区间",
        baseline: "所有输入都执行完整计算路径",
        performanceTension: "低信息增量仍消耗完整路径资源并恶化主性能指标",
        scope6L: {
          L1: "输入选择与条件执行",
          L2: "运行时门控",
          L3: null,
          L4: "条件执行 kernel",
          L5: null,
          L6: null,
        },
        constraints: ["固定影响主指标和 guardrail 的比较变量"],
      },
      evidence: [
        {
          sourceRef: "idea_notes/example.md#baseline",
          supports: "baseline 对低信息输入仍执行完整路径",
        },
      ],
      unresolved: [],
    };
  }
  return {
    workOutcome: "READY_FOR_REVIEW",
    content: {
      name: "轻量判别后按需执行完整路径",
      mechanism: "以低成本判别输入变化，仅在必要时激活完整路径",
      baselineChange: "从无条件完整执行改为轻量判别加条件执行",
      expectedEffects: [
        {
          metric: "目标主性能指标",
          effect: "减少不必要工作并改善该指标",
          conditions: "判别成本显著低于被跳过路径且误判受控",
        },
      ],
      tradeoffs: ["误判可能损害目标 guardrail"],
      failureConditions: ["几乎所有输入都需要完整执行"],
      measurementPlan: ["固定公平变量，对比主指标、guardrail 和资源开销"],
    },
    evidence: [
      {
        sourceRef: "idea_notes/example.md#method",
        supports: "条件执行可减少完整路径调用次数",
      },
    ],
    unresolved: [],
  };
}

function passReview(): unknown {
  return {
    reviewVerdict: "PASS",
    summary: "目标范围、baseline、证据和可证伪条件足以进入当前结果。",
    findings: [
      {
        severity: "NON_BLOCKING",
        issue: "具体阈值仍需后续测量",
        basis: "当前工作流不执行新实验",
        expected: "按 measurementPlan 在后续实验中核验",
      },
    ],
    queryGaps: [],
  };
}

function turns(store: FileLoopStore): TurnFile[] {
  return readdirSync(store.absolute("turns"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => store.readTurn(`turns/${entry.name}/turn.json`))
    .sort((left, right) => left.startedAt.localeCompare(right.startedAt));
}

function latestDecisionContext(store: FileLoopStore): Record<string, unknown> {
  const refs = readdirSync(store.absolute("contexts"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => `contexts/${entry.name}/decision_context.json`);
  assert.ok(refs.length > 0);
  return store.readJson(refs.at(-1)!);
}

function prepareInterruptedWorker(
  store: FileLoopStore,
  withCapturedOutput: boolean,
): { turnId: string; turnRef: string; bindingRef: string } {
  let state: StateFile = store.readState();
  const created = createWorkerBinding(store, state);
  const turnId = store.newId("turn");
  const turnRef = `turns/${turnId}/turn.json`;
  const promptRef = `turns/${turnId}/prompt.txt`;
  const validationAuditRef = `turns/${turnId}/validation_audit.json`;
  const runtimeRef = `turns/${turnId}/runtime.jsonl`;
  store.writeText(promptRef, "interrupted Worker prompt");
  store.writeJson(validationAuditRef, { checks: [], advisories: [] });
  store.writeText(runtimeRef, "");
  const rawOutputRef = `turns/${turnId}/output.txt`;
  if (withCapturedOutput) {
    const task = store.readJson<TurnTask>(created.binding.taskRef);
    store.writeText(rawOutputRef, JSON.stringify(workerResult(task)));
  }
  const turn: TurnFile = {
    turnId,
    role: "WORKER",
    round: 1,
    attempt: 1,
    taskBindingRef: created.bindingRef,
    decisionContextRef: null,
    retryOf: null,
    skill: "learning-loop-worker",
    turnState: "RUNNING",
    startedAt: new Date().toISOString(),
    completedAt: null,
    promptRef,
    outputCapture: withCapturedOutput ? "COMPLETE" : "NONE",
    partialOutputRef: null,
    rawOutputRef: withCapturedOutput ? rawOutputRef : null,
    resultRef: null,
    controlRef: null,
    validationAuditRef,
    errorReportRef: null,
    runtimeErrorRef: null,
    runtimeRef,
    providerThreadId: "interrupted-thread",
    providerTurnId: "interrupted-turn",
    providerStatus: null,
    runtimeFailureKind: null,
    timeoutProfile: store.readRun().budgets.timeoutProfiles.WORKER,
    recoveryRef: null,
  };
  store.writeTurn(turn);
  store.appendTurnToRound(1, turnRef);
  state = {
    ...state,
    revision: state.revision + 1,
    sequence: [
      {
        ...state.sequence[0]!,
        bindingRef: created.bindingRef,
      },
      ...state.sequence.slice(1),
    ],
    activeTaskBindingRef: created.bindingRef,
    activeTurnRef: turnRef,
  };
  store.writeState(state, "TEST_INTERRUPTED_TURN");
  return { turnId, turnRef, bindingRef: created.bindingRef };
}
