import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { AppServerClient } from "../app_server_client.ts";
import { testConfig } from "./test_helpers.ts";

const fakeServer = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "fake_app_server.mjs",
);

test("AppServerClient multiplexes two threads and records usage", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-app-client-"));
  const config = testConfig(root);
  const workDir = path.join(root, "run");
  fs.mkdirSync(workDir, { recursive: true });
  const client = new AppServerClient(config, workDir, {
    command: process.execPath,
    argsPrefix: [fakeServer],
  });
  await client.start();
  try {
    const first = await client.startThread("anchor_stage_controller", root, true);
    const second = await client.startThread("anchor_curator_worker", root, false);
    const [firstResult, secondResult] = await Promise.all([
      client.runTurn("anchor_stage_controller", first.threadId, "high", [{ type: "text", text: "one" }]),
      client.runTurn("anchor_curator_worker", second.threadId, "high", [{ type: "text", text: "two" }]),
    ]);
    assert.equal(firstResult.text, "response:one");
    assert.equal(secondResult.text, "response:two");
    assert.equal(firstResult.usage?.inputTokens, 10);
    assert.equal(firstResult.securityViolations.length, 0);
    await client.resumeThread("anchor_stage_controller", first.threadId, root);
  } finally {
    await client.stop();
  }
});

test("AppServerClient reconstructs text when completed turn items are not loaded", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-app-stream-"));
  const config = testConfig(root);
  const workDir = path.join(root, "run");
  fs.mkdirSync(workDir, { recursive: true });
  const client = new AppServerClient(config, workDir, {
    command: process.execPath,
    argsPrefix: [fakeServer],
  });
  await client.start();
  try {
    const thread = await client.startThread("anchor_stage_controller", root, true);
    const result = await client.runTurn(
      "anchor_stage_controller",
      thread.threadId,
      "high",
      [{ type: "text", text: "STREAM_ONLY" }],
    );
    assert.equal(result.text, "response:STREAM_ONLY");
  } finally {
    await client.stop();
  }
});

test("AppServerClient admission gate rejects shell and permits Obsidian read MCP", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-app-admission-"));
  const config = testConfig(root);
  const workDir = path.join(root, "run");
  fs.mkdirSync(workDir, { recursive: true });
  const client = new AppServerClient(config, workDir, {
    command: process.execPath,
    argsPrefix: [fakeServer],
  });
  await client.start();
  try {
    const blind = await client.startThread("anchor_stage_controller", root, true);
    const blindResult = await client.runTurn(
      "anchor_stage_controller",
      blind.threadId,
      "high",
      [{ type: "text", text: "TOOL_VIOLATION" }],
    );
    assert.match(blindResult.securityViolations.join("\n"), /commandExecution/);

    const evidence = await client.startThread("anchor_evidence_worker", root, false);
    const evidenceResult = await client.runTurn(
      "anchor_evidence_worker",
      evidence.threadId,
      "medium",
      [{ type: "text", text: "MCP_OK" }],
    );
    assert.deepEqual(evidenceResult.securityViolations, []);
  } finally {
    await client.stop();
  }
});

test("AppServerClient surfaces server crash during an active turn", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-app-crash-"));
  const config = testConfig(root);
  const workDir = path.join(root, "run");
  fs.mkdirSync(workDir, { recursive: true });
  const client = new AppServerClient(config, workDir, {
    command: process.execPath,
    argsPrefix: [fakeServer],
  });
  await client.start();
  const thread = await client.startThread("anchor_stage_controller", root, true);
  await assert.rejects(
    client.runTurn(
      "anchor_stage_controller",
      thread.threadId,
      "high",
      [{ type: "text", text: "CRASH_SERVER" }],
    ),
    /exited unexpectedly/,
  );
});
