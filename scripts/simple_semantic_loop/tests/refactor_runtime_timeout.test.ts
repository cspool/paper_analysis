import assert from "node:assert/strict";
import { resolve } from "node:path";
import { test } from "node:test";
import { CodexAppServerRuntime } from "../refactor/runtime.ts";
import type { RuntimeLiveEvent } from "../refactor/runtime.ts";
import { LiveConsoleRenderer } from "../refactor/live_console.ts";
import type {
  RuntimePersistenceEvent,
  TurnDispatch,
} from "../refactor/types.ts";

const fixture = resolve(
  import.meta.dirname,
  "fixtures/fake_codex_app_server.mjs",
);

test("idle timeout snapshots before interrupt and preserves provider identity", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const runtime = fakeRuntime();
  try {
    const result = await runtime.run(dispatch("IDLE_TIMEOUT", events, {
      idleTimeoutMs: 60,
      hardTimeoutMs: 300,
      interruptGraceMs: 10,
    }));
    assert.equal(result.status, "timeout");
    assert.equal(result.failureKind, "IDLE_TIMEOUT");
    assert.equal(result.outputCapture, "NONE");
    assert.equal(result.providerThreadId, "fake-thread");
    assert.equal(result.providerTurnId, "fake-turn");
    const timeoutIndex = events.findIndex((event) => event.type === "timeout");
    const interruptIndex = events.findIndex((event) => event.type === "interrupt");
    assert.ok(timeoutIndex >= 0);
    assert.ok(interruptIndex > timeoutIndex, "snapshot precedes interrupt");
  } finally {
    await runtime.close();
  }
});

test("meaningful activity resets idle timeout but cannot extend the hard cap", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const runtime = fakeRuntime();
  try {
    const result = await runtime.run(dispatch("HARD_ACTIVITY", events, {
      idleTimeoutMs: 50,
      hardTimeoutMs: 140,
      interruptGraceMs: 10,
    }));
    assert.equal(result.status, "timeout");
    assert.equal(result.failureKind, "HARD_TIMEOUT");
    assert.ok(result.usage.totalTokens > 0);
    assert.ok(events.filter((event) => event.type === "usage").length >= 3);
    assert.ok(result.elapsedMs >= 120);
  } finally {
    await runtime.close();
  }
});

test("MCP progress resets idle timeout but cannot extend the hard cap", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const runtime = fakeRuntime();
  try {
    const result = await runtime.run(dispatch("MCP_PROGRESS_ACTIVITY", events, {
      idleTimeoutMs: 50,
      hardTimeoutMs: 140,
      interruptGraceMs: 10,
    }));
    assert.equal(result.status, "timeout");
    assert.equal(result.failureKind, "HARD_TIMEOUT");
    assert.ok(
      result.rawEvents.filter((event) =>
        (event as { method?: string }).method === "item/mcpToolCall/progress"
      ).length >= 3,
    );
  } finally {
    await runtime.close();
  }
});

test("current collabAgentToolCall items are captured as tool events", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const runtime = fakeRuntime();
  try {
    const result = await runtime.run(dispatch("COLLAB_AGENT_TOOL", events, {
      idleTimeoutMs: 100,
      hardTimeoutMs: 300,
      interruptGraceMs: 10,
    }));
    assert.equal(result.status, "completed");
    assert.equal(result.toolEvents.length, 1);
    assert.equal(result.toolEvents[0]?.toolName, "spawnAgent");
  } finally {
    await runtime.close();
  }
});

test("completed final-answer boundary remains COMPLETE when provider never terminates", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const runtime = fakeRuntime();
  try {
    const result = await runtime.run(
      dispatch("COMPLETE_BEFORE_TIMEOUT", events, {
        idleTimeoutMs: 60,
        hardTimeoutMs: 300,
        interruptGraceMs: 10,
      }),
    );
    assert.equal(result.status, "timeout");
    assert.equal(result.failureKind, "IDLE_TIMEOUT");
    assert.equal(result.outputCapture, "COMPLETE");
    assert.equal(result.text, '{"workOutcome":"READY_FOR_REVIEW"}');
    const outputDeltas = events.filter((event) => event.type === "output_delta");
    assert.equal(outputDeltas.length, 1);
    assert.equal(
      events.some(
        (event) =>
          event.type === "raw_event" &&
          (event.event as { method?: string }).method ===
            "item/agentMessage/delta",
      ),
      false,
      "persistence stores the Agent delta only as output_delta",
    );
    assert.ok(events.some((event) => event.type === "message_completed"));
  } finally {
    await runtime.close();
  }
});

test("live output stays per-delta while persisted output deltas are coalesced", async () => {
  const events: RuntimePersistenceEvent[] = [];
  const live: RuntimeLiveEvent[] = [];
  const runtime = fakeRuntime((event) => live.push(event));
  try {
    const result = await runtime.run(dispatch("BATCHED_DELTAS", events, {
      idleTimeoutMs: 200,
      hardTimeoutMs: 500,
      interruptGraceMs: 10,
    }));
    assert.equal(result.status, "completed");
    assert.equal(result.text, "x".repeat(4_096));
    const persisted = events.filter(
      (event): event is Extract<RuntimePersistenceEvent, { type: "output_delta" }> =>
        event.type === "output_delta",
    );
    assert.equal(
      live.filter((event) => event.type === "agent_output_delta").length,
      64,
    );
    assert.ok(persisted.length < 64);
    assert.equal(persisted.map((event) => event.delta).join(""), result.text);
  } finally {
    await runtime.close();
  }
});

test("console labels distinguish stream, message, provider terminal, and timeout", () => {
  let output = "";
  const renderer = new LiveConsoleRenderer({
    write(chunk: string) {
      output += chunk;
    },
  });
  renderer.handle({
    type: "agent_output_delta",
    turnId: "t",
    role: "WORKER",
    itemId: "i",
    delta: "{",
  });
  renderer.handle({
    type: "agent_message_complete",
    turnId: "t",
    role: "WORKER",
    itemId: "i",
    phase: "final_answer",
  });
  renderer.handle({
    type: "turn_timeout",
    turnId: "t",
    role: "WORKER",
    timeoutMs: 100,
    kind: "HARD_TIMEOUT",
    capture: "COMPLETE",
  });
  renderer.handle({
    type: "turn_completed",
    turnId: "t",
    role: "WORKER",
    status: "timeout",
    elapsedMs: 100,
    usage: {
      inputTokens: 1,
      cachedInputTokens: 0,
      outputTokens: 1,
      reasoningOutputTokens: 0,
      totalTokens: 2,
    },
    toolCalls: 0,
  });
  renderer.finish();
  assert.match(output, /agent:stream-start/);
  assert.match(output, /agent:message-complete/);
  assert.match(output, /turn:timeout.*capture=COMPLETE/);
  assert.match(output, /turn:complete.*status=timeout/);
});

function fakeRuntime(
  onLiveEvent?: (event: RuntimeLiveEvent) => void,
): CodexAppServerRuntime {
  return new CodexAppServerRuntime({
    command: process.execPath,
    argsPrefix: [fixture],
    startupTimeoutMs: 2_000,
    requestTimeoutMs: 2_000,
    onLiveEvent,
  });
}

function dispatch(
  prompt: string,
  events: RuntimePersistenceEvent[],
  timeoutProfile: TurnDispatch["timeoutProfile"],
): TurnDispatch {
  return {
    turnId: `controller-${prompt}`,
    role: "WORKER",
    prompt,
    outputSchema: null,
    cwd: resolve(import.meta.dirname, "../../.."),
    model: "fake-model",
    effort: "high",
    timeoutProfile,
    maxInputTokens: 1_000,
    maxOutputTokens: 1_000,
    onRuntimeEvent: (event) => events.push(event),
  };
}
