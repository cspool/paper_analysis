import assert from "node:assert/strict";
import {
  mkdirSync,
  mkdtempSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import {
  canonicalJson,
  canonicalSha256,
  sha256Bytes,
  type GateDefinition,
  type ObjectRef,
  type StageContract,
  type WorkflowDecisionProposal,
  type WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  WorkflowStore,
} from "../db/workflow_store.ts";
import {
  SemanticLoopController,
  type ControllerConfig,
} from "../controller.ts";
import {
  ScriptedFreshTurnRuntime,
  type FrozenTurnDispatch,
} from "../turns/runtime.ts";
import {
  consumeResult,
  commitTurnResult,
  reconcileCommittedReplacementFailures,
} from "../workflow/result_consumer.ts";
import {
  EVIDENCE_BUDGET,
  HASH_A,
  HASH_B,
  HASH_C,
  ZERO_BUDGET,
  makeClosureResult,
  makeTopic,
} from "./fixtures/factory.ts";

test("run lock rejects a live owner and atomically reclaims a dead process owner", () => {
  const { store } = makeStore();
  try {
    const live = `controller-pid-${process.pid}-live`;
    store.acquireLock("run-1", live);
    assert.throws(
      () =>
        store.acquireLock(
          "run-1",
          `controller-pid-${process.pid}-other`,
        ),
      /already locked/,
    );
    store.releaseLock("run-1", live);

    store.acquireLock("run-1", "controller-pid-999999999-dead");
    const replacement = `controller-pid-${process.pid}-replacement`;
    store.acquireLock("run-1", replacement);
    const row = store.query(
      "SELECT owner_id FROM run_locks WHERE run_id = 'run-1'",
    );
    assert.equal(row[0]!.owner_id, replacement);
    store.releaseLock("run-1", replacement);
  } finally {
    store.close();
  }
});

test("startup reconciliation never resumes provider history and retries only below the cap", async () => {
  const below = makeStore("blocked_semantic");
  let belowRuntimeCalls = 0;
  createDummyTask(below.store, "task-1", "workflow_decision");
  below.store.createAttempt("run-1", {
    attemptId: "attempt-1",
    taskId: "task-1",
    attemptNo: 1,
    role: "workflow_decision",
    logicalEffort: "max",
    providerWireEffort: "max",
  });
  below.store.markAttemptRunning("attempt-1");
  const belowRuntime = new ScriptedFreshTurnRuntime(async () => {
    belowRuntimeCalls += 1;
    return "{}";
  });
  try {
    const controller = new SemanticLoopController(
      below.store,
      belowRuntime,
      controllerConfig(below.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    assert.equal(belowRuntimeCalls, 0);
    assert.equal(
      below.store.query(
        "SELECT status FROM attempts WHERE attempt_id = 'attempt-1'",
      )[0]!.status,
      "interrupted_reconciled",
    );
    assert.equal(
      below.store.query(
        "SELECT status FROM tasks WHERE task_id = 'task-1'",
      )[0]!.status,
      "pending_output_retry",
    );
  } finally {
    below.store.close();
  }

  const exhausted = makeStore("running");
  let exhaustedRuntimeCalls = 0;
  createDummyTask(exhausted.store, "task-2", "workflow_decision");
  exhausted.store.createAttempt("run-1", {
    attemptId: "attempt-2a",
    taskId: "task-2",
    attemptNo: 1,
    role: "workflow_decision",
    logicalEffort: "max",
    providerWireEffort: "max",
  });
  exhausted.store.markAttemptRunning("attempt-2a");
  exhausted.store.finishAttempt("attempt-2a", "provider_failed");
  exhausted.store.createAttempt("run-1", {
    attemptId: "attempt-2b",
    taskId: "task-2",
    attemptNo: 2,
    role: "workflow_decision",
    logicalEffort: "max",
    providerWireEffort: "max",
  });
  exhausted.store.markAttemptRunning("attempt-2b");
  const exhaustedRuntime = new ScriptedFreshTurnRuntime(async () => {
    exhaustedRuntimeCalls += 1;
    return "{}";
  });
  try {
    const controller = new SemanticLoopController(
      exhausted.store,
      exhaustedRuntime,
      controllerConfig(exhausted.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "failed_retriable");
    assert.equal(exhaustedRuntimeCalls, 0);
    assert.equal(
      exhausted.store.query(
        "SELECT status FROM tasks WHERE task_id = 'task-2'",
      )[0]!.status,
      "failed",
    );
  } finally {
    exhausted.store.close();
  }
});

test("legacy captured completed raw Turn is replayed locally without a provider call", async () => {
  const { store, workDir } = makeStore();
  let runtimeCalls = 0;
  const runtime = new ScriptedFreshTurnRuntime(async () => {
    runtimeCalls += 1;
    throw new Error("captured recovery must not dispatch");
  });
  const config = {
    ...controllerConfig(workDir),
    maxTransitionsPerRun: 1,
  };
  const controller = new SemanticLoopController(store, runtime, config);
  (
    controller as unknown as {
      scheduleWorkflowDecision(
        runId: string,
        trigger: {
          trigger: "INITIALIZE_TOPIC";
          issueCodes: string[];
          facts: string[];
        },
      ): void;
    }
  ).scheduleWorkflowDecision("run-1", {
    trigger: "INITIALIZE_TOPIC",
    issueCodes: ["topic_frame_missing"],
    facts: ["No TopicFrame exists."],
  });
  const taskRow = store.query(
    "SELECT task_json FROM tasks WHERE run_id = 'run-1' LIMIT 1",
  )[0]!;
  const task = JSON.parse(String(taskRow.task_json)) as WorkflowTurnTask;
  delete (
    task as unknown as Record<string, unknown>
  ).correctionFeedback;
  const legacyPreimage = structuredClone(task);
  legacyPreimage.decisionInputHash = "";
  task.decisionInputHash = canonicalSha256({
    task: legacyPreimage,
    skillSha256: task.skill.sha256,
    expectedSchemaSha256:
      task.schema.expectedOutputSchemaSha256,
    inlineArtifactHashes: [],
  });
  store.db
    .prepare(
      `UPDATE tasks SET task_json = ?, input_hash = ?
       WHERE task_id = ?`,
    )
    .run(
      canonicalJson(task),
      task.decisionInputHash,
      task.taskId,
    );
  store.createAttempt("run-1", {
    attemptId: task.attemptId,
    taskId: task.taskId,
    attemptNo: 1,
    role: "workflow_decision",
    logicalEffort: "max",
    providerWireEffort: "max",
  });
  store.markAttemptRunning(task.attemptId);
  store.updateAttemptProviderIds(
    task.attemptId,
    "captured-thread",
    "captured-turn",
  );
  persistTestArtifact(
    store,
    workDir,
    "run-1",
    `prompt-${task.attemptId}`,
    `prompts/${task.attemptId}.txt`,
    "captured frozen prompt",
    "turn_prompt",
    "controller_generated",
  );
  const raw = {
    attemptId: task.attemptId,
    providerThreadId: "captured-thread",
    providerTurnId: "captured-turn",
    status: "completed" as const,
    text: JSON.stringify(initialTopicProposal(task)),
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
    error: null,
    elapsedMs: 0,
  };
  persistTestArtifact(
    store,
    workDir,
    "run-1",
    `raw-${task.attemptId}`,
    `raw_turns/${task.attemptId}.json`,
    `${JSON.stringify(raw, null, 2)}\n`,
    "raw_turn",
    "untrusted_log",
  );
  store.recordUsage("run-1", task.attemptId, {
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningOutputTokens: 0,
    toolCalls: 0,
    elapsedMs: 0,
  });
  try {
    const outcome = await controller.run("run-1");
    assert.equal(runtimeCalls, 0);
    assert.equal(
      outcome.lifecycle,
      "paused_budget",
      canonicalJson(
        store.query(
          `SELECT event_type, payload_json FROM events
           ORDER BY event_cursor DESC LIMIT 3`,
        ),
      ),
    );
    const attempt = store.query(
      `SELECT status, raw_response_artifact_id
       FROM attempts WHERE attempt_id = ?`,
      task.attemptId,
    )[0]!;
    assert.equal(attempt.status, "committed");
    assert.equal(
      attempt.raw_response_artifact_id,
      `raw-${task.attemptId}`,
    );
    assert.equal(
      store.query(
        `SELECT COUNT(*) AS count FROM decision_proposals
         WHERE attempt_id = ? AND status = 'accepted'`,
        task.attemptId,
      )[0]!.count,
      1,
    );
    assert.equal(
      store.query(
        `SELECT COUNT(*) AS count FROM gate_results
         WHERE stage_id = ? AND passed = 1`,
        task.stageId,
      )[0]!.count,
      1,
    );
  } finally {
    store.close();
  }
});

test("explicit resume converts an exhausted orphan into a failed Stage and a new Workflow decision", async () => {
  const resumed = makeStore("failed_retriable");
  createDummyTask(resumed.store, "task-exhausted", "workflow_decision");
  for (const attemptNo of [1, 2, 3]) {
    const attemptId = `attempt-exhausted-${attemptNo}`;
    resumed.store.createAttempt("run-1", {
      attemptId,
      taskId: "task-exhausted",
      attemptNo,
      role: "workflow_decision",
      logicalEffort: "max",
      providerWireEffort: "max",
    });
    resumed.store.markAttemptRunning(attemptId);
    resumed.store.finishAttempt(
      attemptId,
      "output_contract_invalid",
    );
  }
  resumed.store.casTransition(
    "run-1",
    resumed.store.stateBinding("run-1"),
    {
      lifecycle: "running",
      pauseOrBlockReason: null,
      eventType: "operator_resumed",
      eventPayload: {
        from: "failed_retriable",
        to: "running",
      },
    },
  );
  let runtimeCalls = 0;
  const runtime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    runtimeCalls += 1;
    const task = parseTask(dispatch.prompt);
    return JSON.stringify(
      task.trigger === "INITIALIZE_TOPIC"
        ? initialTopicProposal(task)
        : blockedProposal(task),
    );
  });
  try {
    const controller = new SemanticLoopController(
      resumed.store,
      runtime,
      controllerConfig(resumed.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    assert.equal(runtimeCalls, 2);
    assert.equal(
      resumed.store.query(
        "SELECT status FROM tasks WHERE task_id = 'task-exhausted'",
      )[0]!.status,
      "failed",
    );
    assert.deepEqual(
      resumed.store.readWorkflowState("run-1").inFlightTaskIds,
      [],
    );
  } finally {
    resumed.store.close();
  }
});

test("invalid output gets two fresh same-role retries, with success and exhaustion bounded at three", async () => {
  const success = makeStore();
  const successDispatches: FrozenTurnDispatch[] = [];
  const successRuntime = new ScriptedFreshTurnRuntime(
    async (dispatch, invocation) => {
      successDispatches.push(dispatch);
      if (invocation <= 2) return "{not-json";
      const task = parseTask(dispatch.prompt);
      return JSON.stringify(
        task.trigger === "INITIALIZE_TOPIC"
          ? initialTopicProposal(task)
          : blockedProposal(task),
      );
    },
  );
  try {
    const controller = new SemanticLoopController(
      success.store,
      successRuntime,
      controllerConfig(success.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    const firstTaskAttempts = success.store.query(
      `SELECT role, provider_thread_id, status
       FROM attempts
       WHERE task_id = (
         SELECT task_id FROM attempts ORDER BY started_at LIMIT 1
       ) ORDER BY attempt_no`,
    );
    assert.equal(firstTaskAttempts.length, 3);
    assert.ok(
      firstTaskAttempts.every(
        (attempt) => attempt.role === "workflow_decision",
      ),
    );
    assert.equal(
      new Set(
        firstTaskAttempts.map((attempt) => attempt.provider_thread_id),
      ).size,
      3,
    );
    assert.ok(
      successDispatches[1]!.prompt.includes(
        "[CONTROLLER_VALIDATION_FAILURE]",
      ),
    );
    assert.ok(
      successDispatches[1]!.prompt.includes("STRUCTURE_INVALID"),
    );
    assert.equal(
      successDispatches[1]!.prompt.includes("{not-json"),
      false,
    );
    const retryTask = parseTask(successDispatches[1]!.prompt);
    assert.equal(
      retryTask.correctionFeedback?.previousAttemptId,
      successDispatches[0]!.attemptId,
    );
    assert.equal(
      retryTask.correctionFeedback?.previousOutputSha256,
      sha256Bytes("{not-json"),
    );
    assert.ok(
      retryTask.correctionFeedback?.errors.some(
        (error: { code: string }) =>
          error.code.startsWith("normalization."),
      ),
    );
    assert.ok(
      retryTask.correctionFeedback?.errors.every(
        (error) =>
          error.requiredRule.length > 0 &&
          Array.isArray(error.validExamples),
      ),
    );
    assert.equal(
      parseTask(successDispatches[2]!.prompt).correctionFeedback
        ?.previousAttemptId,
      successDispatches[1]!.attemptId,
    );
  } finally {
    success.store.close();
  }

  const exhausted = makeStore();
  const exhaustedRuntime = new ScriptedFreshTurnRuntime(async () => "{");
  try {
    const controller = new SemanticLoopController(
      exhausted.store,
      exhaustedRuntime,
      controllerConfig(exhausted.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "failed_retriable");
    const attempts = exhausted.store.query(
      "SELECT role, status FROM attempts ORDER BY attempt_no",
    );
    assert.equal(attempts.length, 3);
    assert.ok(
      attempts.every(
        (attempt) =>
          attempt.role === "workflow_decision" &&
          attempt.status === "output_contract_invalid",
      ),
    );
    assert.deepEqual(
      exhausted.store.readWorkflowState("run-1").inFlightTaskIds,
      [],
    );
  } finally {
    exhausted.store.close();
  }
});

test("schema-valid semantic error is retried with a bound Controller error packet", async () => {
  const fixture = makeStore();
  const dispatches: FrozenTurnDispatch[] = [];
  const runtime = new ScriptedFreshTurnRuntime(
    async (dispatch, invocation) => {
      dispatches.push(dispatch);
      const task = parseTask(dispatch.prompt);
      if (invocation === 1) {
        const invalid = initialTopicProposal(task);
        invalid.proposedGateDefinition!.mechanicalChecks.push({
          checkId: "volatile-attempt-binding",
          predicate: "equals",
          actual: {
            source: "result",
            pointer: "/attemptId",
            valueType: "string",
          },
          expected: task.attemptId,
        });
        return JSON.stringify(invalid);
      }
      return JSON.stringify(
        task.trigger === "INITIALIZE_TOPIC"
          ? initialTopicProposal(task)
          : blockedProposal(task),
      );
    },
  );
  try {
    const controller = new SemanticLoopController(
      fixture.store,
      runtime,
      controllerConfig(fixture.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    assert.ok(dispatches.length >= 3);
    assert.equal(dispatches[0]!.taskId, dispatches[1]!.taskId);
    assert.notEqual(
      dispatches[0]!.attemptId,
      dispatches[1]!.attemptId,
    );
    const correction = parseTask(dispatches[1]!.prompt)
      .correctionFeedback;
    assert.equal(correction?.failureClass, "SEMANTIC_INVALID");
    assert.equal(
      correction?.previousAttemptId,
      dispatches[0]!.attemptId,
    );
    assert.match(
      correction?.validationReportSha256 ?? "",
      /^[a-f0-9]{64}$/,
    );
    assert.ok(
      correction?.errors.some(
        (error: { code: string }) =>
          error.code === "gate.result_pointer_not_stable",
      ),
    );
    const pointerError = correction?.errors.find(
      (error) => error.code === "gate.result_pointer_not_stable",
    );
    assert.match(pointerError?.requiredRule ?? "", /\/payload/);
    assert.ok(
      pointerError?.validExamples.some((example) =>
        example.includes('"pointer":"/payload"'),
      ),
    );
    assert.equal(
      fixture.store.query(
        `SELECT COUNT(*) AS count FROM canonical_objects
         WHERE object_type = 'topic' AND active = 1`,
      )[0]!.count,
      1,
    );
  } finally {
    fixture.store.close();
  }
});

test("binding errors are corrected in a fresh Turn with no rejected state commit", async () => {
  const fixture = makeStore();
  const dispatches: FrozenTurnDispatch[] = [];
  const runtime = new ScriptedFreshTurnRuntime(
    async (dispatch, invocation) => {
      dispatches.push(dispatch);
      const task = parseTask(dispatch.prompt);
      if (invocation === 1) {
        const invalid = initialTopicProposal(task);
        invalid.attemptId = "attempt-stale-binding";
        return JSON.stringify(invalid);
      }
      return JSON.stringify(
        task.trigger === "INITIALIZE_TOPIC"
          ? initialTopicProposal(task)
          : blockedProposal(task),
      );
    },
  );
  try {
    const controller = new SemanticLoopController(
      fixture.store,
      runtime,
      controllerConfig(fixture.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    assert.equal(dispatches[0]!.taskId, dispatches[1]!.taskId);
    assert.notEqual(
      dispatches[0]!.attemptId,
      dispatches[1]!.attemptId,
    );
    const correction = parseTask(dispatches[1]!.prompt)
      .correctionFeedback;
    assert.equal(correction?.failureClass, "BINDING_INVALID");
    assert.equal(
      correction?.previousAttemptId,
      dispatches[0]!.attemptId,
    );
    assert.equal(
      fixture.store.query(
        `SELECT COUNT(*) AS count FROM canonical_objects
         WHERE object_type = 'topic' AND active = 1`,
      )[0]!.count,
      1,
    );
    assert.equal(
      fixture.store.query(
        `SELECT COUNT(*) AS count FROM decision_proposals
         WHERE attempt_id = ?`,
        dispatches[0]!.attemptId,
      )[0]!.count,
      0,
    );
  } finally {
    fixture.store.close();
  }
});

test("provider retry and output correction use separate bounded budgets", async () => {
  const fixture = makeStore();
  const dispatches: FrozenTurnDispatch[] = [];
  const runtime = new ScriptedFreshTurnRuntime(
    async (dispatch, invocation) => {
      dispatches.push(dispatch);
      if (invocation === 1) {
        return {
          status: "failed",
          error: "transient provider failure",
        };
      }
      const task = parseTask(dispatch.prompt);
      if (invocation === 2) {
        const invalid = initialTopicProposal(task);
        invalid.proposedGateDefinition!.mechanicalChecks.push({
          checkId: "volatile-attempt-binding",
          predicate: "equals",
          actual: {
            source: "result",
            pointer: "/attemptId",
            valueType: "string",
          },
          expected: task.attemptId,
        });
        return JSON.stringify(invalid);
      }
      return JSON.stringify(
        task.trigger === "INITIALIZE_TOPIC"
          ? initialTopicProposal(task)
          : blockedProposal(task),
      );
    },
  );
  try {
    const controller = new SemanticLoopController(
      fixture.store,
      runtime,
      controllerConfig(fixture.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "blocked_semantic");
    assert.equal(dispatches[0]!.taskId, dispatches[1]!.taskId);
    assert.equal(dispatches[1]!.taskId, dispatches[2]!.taskId);
    assert.equal(
      parseTask(dispatches[1]!.prompt).correctionFeedback,
      null,
    );
    assert.equal(
      parseTask(dispatches[2]!.prompt).correctionFeedback
        ?.failureClass,
      "SEMANTIC_INVALID",
    );
    const firstTaskAttempts = fixture.store.query(
      `SELECT status FROM attempts
       WHERE task_id = ? ORDER BY attempt_no`,
      dispatches[0]!.taskId,
    );
    assert.deepEqual(
      firstTaskAttempts.map((attempt) => attempt.status),
      [
        "provider_failed",
        "output_contract_invalid",
        "committed",
      ],
    );
  } finally {
    fixture.store.close();
  }
});

test("runtime tool violations fail terminal and frozen Turn-budget overruns pause", async () => {
  const security = makeStore();
  const securityRuntime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    const task = parseTask(dispatch.prompt);
    return {
      text: JSON.stringify(initialTopicProposal(task)),
      toolEvents: [
        {
          toolName: "shell.exec",
          arguments: { command: "true" },
          status: "completed",
          resultText: "",
          error: null,
        },
      ],
    };
  });
  try {
    const controller = new SemanticLoopController(
      security.store,
      securityRuntime,
      controllerConfig(security.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "failed_terminal");
    assert.equal(
      security.store.query("SELECT status FROM attempts")[0]!.status,
      "security_invalid",
    );
    assert.deepEqual(
      security.store.readWorkflowState("run-1").inFlightTaskIds,
      [],
    );
  } finally {
    security.store.close();
  }

  const budget = makeStore();
  const budgetRuntime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    const task = parseTask(dispatch.prompt);
    return {
      text: JSON.stringify(initialTopicProposal(task)),
      usage: {
        inputTokens: ZERO_BUDGET.maxInputTokens + 1,
        cachedInputTokens: 0,
        outputTokens: 1,
        reasoningOutputTokens: 0,
        totalTokens: ZERO_BUDGET.maxInputTokens + 2,
      },
    };
  });
  try {
    const controller = new SemanticLoopController(
      budget.store,
      budgetRuntime,
      controllerConfig(budget.workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "paused_budget");
    assert.deepEqual(
      budget.store.readWorkflowState("run-1").inFlightTaskIds,
      [],
    );
    assert.equal(
      budget.store.query("SELECT status FROM attempts")[0]!.status,
      "budget_invalid",
    );
  } finally {
    budget.store.close();
  }
});

test("result commit and consumption are idempotent only for the same control binding", () => {
  const { store } = makeStore("running");
  try {
    createDummyTask(store, "task-result", "evidence_reader");
    store.createAttempt("run-1", {
      attemptId: "attempt-result",
      taskId: "task-result",
      attemptNo: 1,
      role: "evidence_reader",
      logicalEffort: "high",
      providerWireEffort: "high",
    });
    store.markAttemptRunning("attempt-result");
    insertGate(store, "gate-result", "stage-dummy");
    const validReport = {
      validatorVersion: "test/1",
      valid: true,
      errors: [],
      checkedArtifactHashes: [],
      checkedObjectRefs: [],
    };
    const gateEvaluation = {
      gateId: "gate-result",
      passed: true,
      checks: [],
    };
    const payload = { packetId: "packet-idempotent", value: 1 };
    const committed = commitTurnResult(
      store,
      "run-1",
      store.stateBinding("run-1"),
      {
        resultId: "result-1",
        taskId: "task-result",
        attemptId: "attempt-result",
        stageId: "stage-dummy",
        gateId: "gate-result",
        role: "evidence_reader",
        messageType: "EVIDENCE_PACKET",
        result: payload,
        validationReport: validReport,
        gateEvaluation,
        rawResponseArtifactId: "raw-attempt-result",
      },
    );
    assert.equal(committed.duplicate, false);
    const duplicate = commitTurnResult(
      store,
      "run-1",
      committed.nextState,
      {
        resultId: "result-replayed",
        taskId: "task-result",
        attemptId: "attempt-result",
        stageId: "stage-dummy",
        gateId: "gate-result",
        role: "evidence_reader",
        messageType: "EVIDENCE_PACKET",
        result: payload,
        validationReport: validReport,
        gateEvaluation,
        rawResponseArtifactId: "raw-attempt-result",
      },
    );
    assert.equal(duplicate.duplicate, true);
    assert.equal(duplicate.resultId, "result-1");
    assert.throws(() =>
      commitTurnResult(store, "run-1", committed.nextState, {
        resultId: "result-conflict",
        taskId: "task-result",
        attemptId: "attempt-result",
        stageId: "stage-other",
        gateId: "gate-result",
        role: "evidence_reader",
        messageType: "EVIDENCE_PACKET",
        result: payload,
        validationReport: validReport,
        gateEvaluation,
        rawResponseArtifactId: "raw-attempt-result",
      }),
    );

    const topic = makeTopic();
    const consumed = consumeResult(
      store,
      "run-1",
      store.stateBinding("run-1"),
      {
        resultId: "result-1",
        deltaId: "delta-1",
        canonicalObject: {
          objectType: "topic",
          objectId: topic.topicId,
          revision: topic.revision,
          value: topic,
        },
      },
    );
    assert.equal(consumed.duplicate, false);
    const replay = consumeResult(
      store,
      "run-1",
      consumed.nextState,
      {
        resultId: "result-1",
        deltaId: "delta-1",
        canonicalObject: {
          objectType: "topic",
          objectId: topic.topicId,
          revision: topic.revision,
          value: topic,
        },
      },
    );
    assert.equal(replay.duplicate, true);
    assert.throws(() =>
      consumeResult(store, "run-1", consumed.nextState, {
        resultId: "result-1",
        deltaId: "delta-conflict",
        canonicalObject: {
          objectType: "topic",
          objectId: topic.topicId,
          revision: topic.revision,
          value: { ...topic, objective: "different" },
        },
      }),
    );
  } finally {
    store.close();
  }
});

test("an exact-scope replacement resolves prior failed tasks and validation reports", () => {
  const { store } = makeStore("running");
  const scope: ObjectRef[] = [
    {
      objectType: "search_need",
      objectId: "need-recovery",
      revision: 1,
    },
  ];
  try {
    insertStageContract(
      store,
      "stage-failed",
      scope,
      store.getRun("run-1").snapshotVersion,
    );
    insertStageContract(
      store,
      "stage-replacement",
      scope,
      store.getRun("run-1").snapshotVersion + 1,
    );
    createDummyTask(
      store,
      "task-failed",
      "evidence_reader",
      "stage-failed",
    );
    store.createAttempt("run-1", {
      attemptId: "attempt-failed",
      taskId: "task-failed",
      attemptNo: 1,
      role: "evidence_reader",
      logicalEffort: "high",
      providerWireEffort: "high",
    });
    store.markAttemptRunning("attempt-failed");
    store.finishAttempt(
      "attempt-failed",
      "output_contract_invalid",
    );
    const failedReportId = store.insertValidationReport(
      "run-1",
      {
        validatorVersion: "test/1",
        valid: false,
        errors: [
          {
            code: "test.invalid",
            jsonPointer: "/",
            message: "first bounded task failed",
          },
        ],
        checkedArtifactHashes: [],
        checkedObjectRefs: [],
      },
      "task-failed",
      "attempt-failed",
    );
    store.db
      .prepare(
        "UPDATE tasks SET status = 'failed' WHERE task_id = 'task-failed'",
      )
      .run();

    createDummyTask(
      store,
      "task-replacement",
      "evidence_reader",
      "stage-replacement",
    );
    store.createAttempt("run-1", {
      attemptId: "attempt-replacement",
      taskId: "task-replacement",
      attemptNo: 1,
      role: "evidence_reader",
      logicalEffort: "high",
      providerWireEffort: "high",
    });
    store.markAttemptRunning("attempt-replacement");
    insertGate(
      store,
      "gate-replacement",
      "stage-replacement",
    );
    const committed = commitTurnResult(
      store,
      "run-1",
      store.stateBinding("run-1"),
      {
        resultId: "result-replacement",
        taskId: "task-replacement",
        attemptId: "attempt-replacement",
        stageId: "stage-replacement",
        gateId: "gate-replacement",
        role: "evidence_reader",
        messageType: "EVIDENCE_PACKET",
        result: { packetId: "packet-replacement" },
        validationReport: {
          validatorVersion: "test/1",
          valid: true,
          errors: [],
          checkedArtifactHashes: [],
          checkedObjectRefs: [],
        },
        gateEvaluation: {
          gateId: "gate-replacement",
          passed: true,
          checks: [],
        },
        rawResponseArtifactId: "raw-attempt-replacement",
      },
    );
    assert.equal(committed.duplicate, false);
    assert.equal(
      store.query(
        "SELECT status FROM tasks WHERE task_id = 'task-failed'",
      )[0]!.status,
      "superseded",
    );
    assert.equal(
      store.query(
        `SELECT resolved_by_id FROM validation_reports
         WHERE validation_report_id = ?`,
        failedReportId,
      )[0]!.resolved_by_id,
      "result-replacement",
    );

    // Startup reconciliation repairs the same recoverable metadata after a
    // crash between result commit and recovery bookkeeping.
    store.db
      .prepare(
        "UPDATE tasks SET status = 'failed' WHERE task_id = 'task-failed'",
      )
      .run();
    store.db
      .prepare(
        `UPDATE validation_reports SET resolved_by_id = NULL
         WHERE validation_report_id = ?`,
      )
      .run(failedReportId);
    assert.equal(
      reconcileCommittedReplacementFailures(store, "run-1"),
      1,
    );
    assert.equal(
      store.query(
        "SELECT status FROM tasks WHERE task_id = 'task-failed'",
      )[0]!.status,
      "superseded",
    );
    assert.equal(
      store.query(
        `SELECT resolved_by_id FROM validation_reports
         WHERE validation_report_id = ?`,
        failedReportId,
      )[0]!.resolved_by_id,
      "result-replacement",
    );
  } finally {
    store.close();
  }
});

test("a committed ClosureReview is reconciled after a crash without provider resume", async () => {
  const { store, workDir } = makeStore("closure_preflight");
  createDummyTask(store, "task-closure", "closure_reviewer");
  store.createAttempt("run-1", {
    attemptId: "attempt-closure",
    taskId: "task-closure",
    attemptNo: 1,
    role: "closure_reviewer",
    logicalEffort: "high",
    providerWireEffort: "high",
  });
  store.markAttemptRunning("attempt-closure");
  insertGate(store, "gate-closure", "stage-dummy");
  const closure = makeClosureResult();
  closure.payload.decision = "reject";
  closure.payload.allowsFinalization = false;
  closure.payload.finalizationRequirements = [];
  const committed = commitTurnResult(
    store,
    "run-1",
    store.stateBinding("run-1"),
    {
      resultId: "closure-crash-result",
      taskId: "task-closure",
      attemptId: "attempt-closure",
      stageId: "stage-dummy",
      gateId: "gate-closure",
      role: "closure_reviewer",
      messageType: "CLOSURE_REVIEW",
      result: closure,
      validationReport: {
        validatorVersion: "test/1",
        valid: true,
        errors: [],
        checkedArtifactHashes: [],
        checkedObjectRefs: [],
      },
      gateEvaluation: {
        gateId: "gate-closure",
        passed: true,
        checks: [],
      },
      rawResponseArtifactId: "raw-attempt-closure",
      committedLifecycle: "waiting_closure_review",
    },
  );
  assert.equal(
    store.getRun("run-1").lifecycle,
    "waiting_closure_review",
  );

  let runtimeCalls = 0;
  const runtime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    runtimeCalls += 1;
    return JSON.stringify(askUserProposal(parseTask(dispatch.prompt)));
  });
  try {
    const controller = new SemanticLoopController(
      store,
      runtime,
      controllerConfig(workDir),
    );
    const outcome = await controller.run("run-1");
    assert.equal(outcome.lifecycle, "waiting_user");
    assert.equal(runtimeCalls, 1);
    assert.equal(
      store.query(
        `SELECT COUNT(*) AS count FROM result_consumptions
         WHERE result_id = 'closure-crash-result'`,
      )[0]!.count,
      1,
    );
    assert.equal(
      store.query(
        `SELECT COUNT(*) AS count FROM events
         WHERE event_type = 'closure_review_rejected'`,
      )[0]!.count,
      1,
    );
    assert.ok(committed.nextState.snapshotVersion > 0);
  } finally {
    store.close();
  }
});

test("completed runs carry completed_at and reject every later write", () => {
  const { store } = makeStore("finalizing");
  try {
    store.registerArtifact("run-1", {
      artifactId: "final-artifact",
      kind: "final_markdown",
      relativePath: "final.md",
      sha256: "f".repeat(64),
      sizeBytes: 1,
      trustClass: "canonical",
    });
    store.markCompleted(
      "run-1",
      store.stateBinding("run-1"),
      "final-artifact",
    );
    const run = store.getRun("run-1");
    assert.equal(run.lifecycle, "completed");
    assert.ok(run.completedAt);
    assert.throws(() =>
      store.casTransition("run-1", store.stateBinding("run-1"), {
        lifecycle: "running",
        eventType: "illegal",
        eventPayload: {},
      }),
    );
    assert.throws(() =>
      store.registerArtifact("run-1", {
        artifactId: "late",
        kind: "late",
        relativePath: "late",
        sha256: "e".repeat(64),
        sizeBytes: 0,
        trustClass: "canonical",
      }),
    );
  } finally {
    store.close();
  }
});

function persistTestArtifact(
  store: WorkflowStore,
  workDir: string,
  runId: string,
  artifactId: string,
  relativePath: string,
  content: string,
  kind: string,
  trustClass: string,
): void {
  const absolutePath = resolve(workDir, relativePath);
  mkdirSync(dirname(absolutePath), { recursive: true });
  writeFileSync(absolutePath, content, "utf8");
  const bytes = Buffer.from(content, "utf8");
  store.registerArtifact(runId, {
    artifactId,
    kind,
    relativePath,
    sha256: sha256Bytes(bytes),
    sizeBytes: bytes.byteLength,
    trustClass,
  });
}

function makeStore(
  lifecycle:
    | "initialized"
    | "running"
    | "blocked_semantic"
    | "closure_preflight"
    | "finalizing" = "initialized",
): { store: WorkflowStore; workDir: string } {
  const workDir = mkdtempSync(resolve(tmpdir(), "simple-loop-recovery-"));
  const store = new WorkflowStore(resolve(workDir, "workflow.db"));
  store.createRun({
    runId: "run-1",
    workflowId: "workflow-1",
    objective: "Build a bounded semantic exploration.",
    acceptanceCriteria: ["Preserve scope and produce traceable decisions."],
    config: {
      maxTurns: 40,
      maxInputTokens: 1_000_000,
      maxOutputTokens: 500_000,
      maxToolCalls: 100,
      maxElapsedMs: 3_600_000,
    },
  });
  transitionTo(store, lifecycle);
  return { store, workDir };
}

function transitionTo(
  store: WorkflowStore,
  lifecycle:
    | "initialized"
    | "running"
    | "blocked_semantic"
    | "closure_preflight"
    | "finalizing",
): void {
  if (lifecycle === "initialized") return;
  store.casTransition("run-1", store.stateBinding("run-1"), {
    lifecycle: "running",
    eventType: "test_running",
    eventPayload: {},
  });
  if (lifecycle === "running") return;
  if (lifecycle === "blocked_semantic") {
    store.casTransition("run-1", store.stateBinding("run-1"), {
      lifecycle,
      eventType: "test_blocked",
      eventPayload: {},
    });
    return;
  }
  store.casTransition("run-1", store.stateBinding("run-1"), {
    lifecycle: "closure_preflight",
    eventType: "test_preflight",
    eventPayload: {},
  });
  if (lifecycle === "closure_preflight") return;
  store.casTransition("run-1", store.stateBinding("run-1"), {
    lifecycle: "waiting_closure_review",
    eventType: "test_waiting_closure",
    eventPayload: {},
  });
  store.casTransition("run-1", store.stateBinding("run-1"), {
    lifecycle: "finalizing",
    eventType: "test_finalizing",
    eventPayload: {},
  });
}

function createDummyTask(
  store: WorkflowStore,
  taskId: string,
  role:
    | "workflow_decision"
    | "evidence_reader"
    | "closure_reviewer",
  stageId = "stage-dummy",
): void {
  store.createTask("run-1", {
    taskId,
    stageId,
    role,
    inputMessageType:
      role === "workflow_decision"
        ? "WORKFLOW_TURN_TASK"
        : role === "evidence_reader"
          ? "EVIDENCE_READER_TASK"
          : "CLOSURE_REVIEW_TASK",
    expectedOutputMessageType:
      role === "workflow_decision"
        ? "WORKFLOW_DECISION_PROPOSAL"
        : role === "evidence_reader"
          ? "EVIDENCE_PACKET"
          : "CLOSURE_REVIEW",
    stateBinding: store.stateBinding("run-1"),
    inputHash: HASH_A,
    stageContractHash: HASH_B,
    skillHash: HASH_C,
    schemaManifestHash: HASH_A,
    task: { taskId, role },
  });
}

function insertStageContract(
  store: WorkflowStore,
  stageId: string,
  scope: ObjectRef[],
  definedAtSnapshotVersion: number,
): StageContract {
  const draft = {
    proposalLocalStageKey: `local-${stageId}`,
    stageType: "EVIDENCE_READ" as const,
    objective: "Read one bounded recovery scope.",
    scope,
    executionKind: "WORKER_TURN" as const,
    role: "evidence_reader" as const,
    requiredInputs: [],
    expectedOutputMessageType: "EVIDENCE_PACKET" as const,
    requestedTools: [
      "mcp__obsidian__obsidian_search_notes",
      "mcp__obsidian__obsidian_get_note",
    ],
    requestedPaths: ["knowledge_notes/"],
    prohibitedActions: ["experiment execution"],
    budget: structuredClone(EVIDENCE_BUDGET),
  };
  const contract: StageContract = {
    ...draft,
    contractId: `contract-${stageId}`,
    stageId,
    revision: 1,
    definedAtSnapshotVersion,
    sha256: canonicalSha256(draft),
  };
  store.db
    .prepare(
      `INSERT INTO stage_contracts(
         contract_id, run_id, stage_id, revision, stage_type, role,
         defined_at_snapshot_version, sha256, contract_json, created_at
       ) VALUES (?, 'run-1', ?, 1, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      contract.contractId,
      stageId,
      contract.stageType,
      contract.role,
      contract.definedAtSnapshotVersion,
      contract.sha256,
      canonicalJson(contract),
      new Date().toISOString(),
    );
  return contract;
}

function insertGate(
  store: WorkflowStore,
  gateId: string,
  stageId: string,
): void {
  const gate: GateDefinition = {
    gateId,
    stageId,
    revision: 1,
    definedAtSnapshotVersion: store.getRun("run-1").snapshotVersion,
    sha256: canonicalSha256({ gateId, stageId }),
    proposalLocalStageKey: `local-${gateId}`,
    mechanicalChecks: [],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  };
  store.db
    .prepare(
      `INSERT INTO gate_definitions(
         gate_id, run_id, stage_id, revision, defined_at_snapshot_version,
         sha256, gate_json, created_at
       ) VALUES (?, 'run-1', ?, 1, ?, ?, ?, ?)`,
    )
    .run(
      gateId,
      stageId,
      gate.definedAtSnapshotVersion,
      gate.sha256,
      canonicalJson(gate),
      new Date().toISOString(),
    );
}

function controllerConfig(workDir: string): ControllerConfig {
  const projectRoot = resolve(import.meta.dirname, "../../..");
  return {
    projectRoot,
    workDir,
    model: "scripted-model",
    skillRoot: resolve(projectRoot, ".codex/skills"),
    schemaManifestPath: resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/schema_manifest.json",
    ),
    capabilityManifest: {
      provider: "scripted",
      model: "scripted-model",
      wireEffortByLogicalEffort: { high: "high", max: "max" },
      highestWireEffort: "max",
    },
    budgets: {
      workflow: structuredClone(ZERO_BUDGET),
      evidence: structuredClone(EVIDENCE_BUDGET),
      direction: structuredClone(ZERO_BUDGET),
      closure: structuredClone(ZERO_BUDGET),
    },
    maxTransitionsPerRun: 30,
    noProgressThreshold: 3,
  };
}

function parseTask(prompt: string): WorkflowTurnTask {
  const match = prompt.match(
    /\[TASK_PACKET\]\n([\s\S]*?)\n\n\[EXPECTED_OUTPUT_SCHEMA\]/,
  );
  assert.ok(match);
  return JSON.parse(match[1]!);
}

function baseProposal(task: WorkflowTurnTask): WorkflowDecisionProposal {
  return {
    protocolVersion: 1,
    messageType: "WORKFLOW_DECISION_PROPOSAL",
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    expectedState: task.stateSnapshot,
    decisionInputHash: task.decisionInputHash,
    proposalId: `proposal-${task.taskId}`,
    decision: "REPORT_BLOCKED",
    reason: "A bounded test fixture stops here.",
    assumptions: [],
    proposedStageContract: null,
    proposedGateDefinition: null,
    proposedPlanPatch: null,
    targetStageId: null,
    domainProposal: null,
    basisArtifactRefs: [],
    basisResultRefs: [],
    requestedUserInput: null,
    blockedReport: null,
    pauseProposal: null,
    confidence: null,
  };
}

function initialTopicProposal(
  task: WorkflowTurnTask,
): WorkflowDecisionProposal {
  const base = baseProposal(task);
  const local = "topic-stage-local";
  return {
    ...base,
    decision: "RUN_STAGE",
    proposedStageContract: {
      proposalLocalStageKey: local,
      stageType: "SCRIPT_APPLY_TOPIC_FRAME",
      objective: "Commit the initial TopicFrame.",
      scope: [],
      executionKind: "SCRIPT_TRANSITION",
      role: null,
      requiredInputs: [],
      expectedOutputMessageType: null,
      requestedTools: [],
      requestedPaths: [],
      prohibitedActions: [
        "state write outside Controller",
        "experiment execution",
      ],
      budget: structuredClone(ZERO_BUDGET),
    },
    proposedGateDefinition: {
      proposalLocalStageKey: local,
      mechanicalChecks: [],
      semanticEvaluation: {
        required: false,
        evaluatorRole: null,
        rubricId: null,
        inputProjection: [],
        expectedOutputMessageType: null,
      },
    },
    domainProposal: { kind: "topic_frame", value: makeTopic() },
  };
}

function blockedProposal(
  task: WorkflowTurnTask,
): WorkflowDecisionProposal {
  return {
    ...baseProposal(task),
    blockedReport: {
      blockedReportId: `blocked-${task.taskId}`,
      kind: "semantic",
      summary: "The bounded test fixture intentionally stops.",
      blockingFacts: ["No further fixture decision is supplied."],
      exhaustedAlternatives: [],
      relatedRefs: [],
      userActionNeeded: null,
    },
  };
}

function askUserProposal(
  task: WorkflowTurnTask,
): WorkflowDecisionProposal {
  return {
    ...baseProposal(task),
    decision: "ASK_USER",
    requestedUserInput: {
      questionId: `question-${task.taskId}`,
      prompt: "Which authorized closure-recovery scope should be reopened?",
      rationale:
        "The Closure Reviewer rejected the candidate and the fixture requires an operator choice.",
      choices: [
        {
          choiceId: "reopen",
          label: "Reopen frontier",
          consequence: "Resume semantic exploration.",
        },
      ],
      requiredAuthority: null,
      relatedRefs: [],
    },
  };
}
