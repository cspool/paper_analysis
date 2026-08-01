import assert from "node:assert/strict";
import { resolve } from "node:path";
import { test } from "node:test";
import { LiveConsoleRenderer } from "../live_console.ts";
import {
  AppServerFreshTurnRuntime,
  type AppServerLiveEvent,
} from "../turns/app_server_runtime.ts";
import type { FrozenTurnDispatch } from "../turns/runtime.ts";

const fakeServer = resolve(
  import.meta.dirname,
  "fixtures/fake_app_server.mjs",
);

test("App Server runtime applies explicit YOLO and streams live console events", async () => {
  const events: AppServerLiveEvent[] = [];
  const consoleChunks: string[] = [];
  const renderer = new LiveConsoleRenderer({
    write(chunk: string) {
      consoleChunks.push(chunk);
    },
  });
  const runtime = new AppServerFreshTurnRuntime({
    command: process.execPath,
    argsPrefix: [fakeServer],
    sandbox: "danger-full-access",
    onLiveEvent(event) {
      events.push(event);
      renderer.handle(event);
    },
  });
  const dispatch: FrozenTurnDispatch = {
    attemptId: "attempt-live-1",
    taskId: "task-live-1",
    role: "evidence_reader",
    model: "gpt-5.6-sol",
    logicalEffort: "high",
    providerWireEffort: "high",
    prompt: "Return one test result.",
    outputSchema: { type: "object" },
    cwd: resolve(import.meta.dirname, "../../.."),
    timeoutMs: 10_000,
  };

  try {
    const result = await runtime.run(dispatch);
    renderer.finish();
    assert.deepEqual(JSON.parse(result.text), {
      approvalPolicy: "never",
      sandbox: "danger-full-access",
    });
    assert.equal(result.status, "completed");
    assert.equal(result.usage.inputTokens, 11);
    assert.equal(result.usage.outputTokens, 7);
    assert.equal(
      result.toolEvents[0]?.toolName,
      "mcp__obsidian__obsidian_search_notes",
    );
    assert.deepEqual(result.rawEvents[0], {
      method: "controller/runtimePolicy",
      params: {
        approvalPolicy: "never",
        sandbox: "danger-full-access",
        ephemeralThread: true,
        providerHistoryResumed: false,
      },
    });
    assert.ok(events.some((event) => event.type === "turn_starting"));
    assert.ok(events.some((event) => event.type === "turn_started"));
    assert.equal(
      events
        .filter(
          (event): event is Extract<
            AppServerLiveEvent,
            { type: "agent_output_delta" }
          > => event.type === "agent_output_delta",
        )
        .map((event) => event.delta)
        .join(""),
      result.text,
    );
    assert.ok(
      events.some(
        (event) =>
          event.type === "tool_status" && event.phase === "started",
      ),
    );
    assert.ok(
      events.some(
        (event) =>
          event.type === "tool_status" && event.phase === "completed",
      ),
    );
    assert.ok(events.some((event) => event.type === "turn_completed"));

    const consoleText = consoleChunks.join("");
    assert.match(consoleText, /sandbox=danger-full-access/);
    assert.match(consoleText, /tool:started/);
    assert.match(consoleText, /tool:completed/);
    assert.match(consoleText, /"approvalPolicy":"never"/);
    assert.match(consoleText, /turn:complete/);
  } finally {
    await runtime.close();
  }
});
