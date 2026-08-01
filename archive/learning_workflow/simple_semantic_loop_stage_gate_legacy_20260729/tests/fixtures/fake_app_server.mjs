import readline from "node:readline";

process.stdin.resume();
const keepAlive = setInterval(() => {}, 60_000);
const reader = readline.createInterface({ input: process.stdin });
reader.on("close", () => clearInterval(keepAlive));

let threadCounter = 0;
let turnCounter = 0;
const threadPolicies = new Map();

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

reader.on("line", (line) => {
  const message = JSON.parse(line);
  const method = message.method;
  if (method === "initialize") {
    send({
      id: message.id,
      result: {
        userAgent: "simple-loop-fake/1.0",
        codexHome: "/tmp/fake-codex",
        platformFamily: "unix",
        platformOs: "linux",
      },
    });
    return;
  }
  if (method === "initialized") return;
  if (method === "thread/start") {
    threadCounter += 1;
    const threadId = `thread-${threadCounter}`;
    threadPolicies.set(threadId, {
      approvalPolicy: message.params.approvalPolicy,
      sandbox: message.params.sandbox,
    });
    send({
      id: message.id,
      result: {
        thread: { id: threadId, turns: [], status: { type: "idle" } },
      },
    });
    return;
  }
  if (method === "turn/start") {
    turnCounter += 1;
    const turnId = `turn-${turnCounter}`;
    const threadId = message.params.threadId;
    const policy = threadPolicies.get(threadId);
    const outputText = JSON.stringify(policy);
    send({
      id: message.id,
      result: {
        turn: { id: turnId, status: "inProgress", items: [] },
      },
    });
    setTimeout(() => {
      const toolItem = {
        id: `${turnId}-tool`,
        type: "mcpToolCall",
        server: "obsidian",
        tool: "obsidian_search_notes",
        status: "inProgress",
        arguments: {
          mode: "omnisearch",
          query: "path:idea_notes/ multimodal latency",
        },
        result: null,
        error: null,
      };
      send({
        method: "item/started",
        params: { threadId, turnId, item: toolItem },
      });
      send({
        method: "item/completed",
        params: {
          threadId,
          turnId,
          item: { ...toolItem, status: "completed" },
        },
      });
      const itemId = `${turnId}-agent`;
      const midpoint = Math.floor(outputText.length / 2);
      for (const delta of [
        outputText.slice(0, midpoint),
        outputText.slice(midpoint),
      ]) {
        send({
          method: "item/agentMessage/delta",
          params: { threadId, turnId, itemId, delta },
        });
      }
      send({
        method: "item/completed",
        params: {
          threadId,
          turnId,
          item: {
            id: itemId,
            type: "agentMessage",
            text: outputText,
            phase: "final_answer",
          },
        },
      });
      send({
        method: "thread/tokenUsage/updated",
        params: {
          threadId,
          turnId,
          tokenUsage: {
            last: {
              inputTokens: 11,
              cachedInputTokens: 2,
              outputTokens: 7,
              reasoningOutputTokens: 3,
              totalTokens: 18,
            },
          },
        },
      });
      send({
        method: "turn/completed",
        params: {
          threadId,
          turn: {
            id: turnId,
            status: "completed",
            items: [],
            itemsView: "notLoaded",
          },
        },
      });
    }, 10);
    return;
  }
  if (method === "turn/interrupt") {
    send({ id: message.id, result: {} });
    return;
  }
  if (typeof message.id === "number") {
    send({ id: message.id, result: {} });
  }
});
