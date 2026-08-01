import { createInterface } from "node:readline";

const reader = createInterface({ input: process.stdin });
let activity = null;

function send(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

reader.on("line", (line) => {
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    send({ id: message.id, result: { codexHome: "/tmp/fake-codex" } });
    return;
  }
  if (message.method === "initialized") return;
  if (message.method === "thread/start") {
    send({ id: message.id, result: { thread: { id: "fake-thread" } } });
    return;
  }
  if (message.method === "turn/start") {
    const prompt = message.params?.input?.[0]?.text ?? "";
    send({ id: message.id, result: { turn: { id: "fake-turn" } } });
    if (prompt.includes("HARD_ACTIVITY")) {
      let n = 0;
      activity = setInterval(() => {
        n += 1;
        send({
          method: "thread/tokenUsage/updated",
          params: {
            turnId: "fake-turn",
            tokenUsage: {
              last: {
                inputTokens: n,
                cachedInputTokens: 0,
                outputTokens: n,
                reasoningOutputTokens: 0,
                totalTokens: n * 2,
              },
            },
          },
        });
      }, 20);
    } else if (prompt.includes("MCP_PROGRESS_ACTIVITY")) {
      let n = 0;
      activity = setInterval(() => {
        n += 1;
        send({
          method: "item/mcpToolCall/progress",
          params: {
            turnId: "fake-turn",
            itemId: "fake-mcp",
            message: `progress-${n}`,
          },
        });
      }, 20);
    } else if (prompt.includes("COLLAB_AGENT_TOOL")) {
      const item = {
        id: "fake-collab",
        type: "collabAgentToolCall",
        tool: "spawnAgent",
        status: "completed",
      };
      setTimeout(() => {
        send({
          method: "item/started",
          params: {
            turnId: "fake-turn",
            item: { ...item, status: "inProgress" },
          },
        });
        send({
          method: "item/completed",
          params: { turnId: "fake-turn", item },
        });
        send({
          method: "turn/completed",
          params: {
            turn: {
              id: "fake-turn",
              status: "completed",
              items: [
                item,
                {
                  id: "fake-message",
                  type: "agentMessage",
                  phase: "final_answer",
                  text: '{"workOutcome":"READY_FOR_REVIEW"}',
                },
              ],
            },
          },
        });
      }, 10);
    } else if (prompt.includes("COMPLETE_BEFORE_TIMEOUT")) {
      const text = '{"workOutcome":"READY_FOR_REVIEW"}';
      setTimeout(() => {
        send({
          method: "item/agentMessage/delta",
          params: {
            turnId: "fake-turn",
            itemId: "fake-message",
            delta: text,
          },
        });
        send({
          method: "item/completed",
          params: {
            turnId: "fake-turn",
            item: {
              id: "fake-message",
              type: "agentMessage",
              phase: "final_answer",
              text,
            },
          },
        });
      }, 10);
    } else if (prompt.includes("BATCHED_DELTAS")) {
      const chunks = Array.from({ length: 64 }, () => "x".repeat(64));
      const text = chunks.join("");
      setTimeout(() => {
        for (const delta of chunks) {
          send({
            method: "item/agentMessage/delta",
            params: {
              turnId: "fake-turn",
              itemId: "fake-message",
              delta,
            },
          });
        }
        send({
          method: "item/completed",
          params: {
            turnId: "fake-turn",
            item: {
              id: "fake-message",
              type: "agentMessage",
              phase: "final_answer",
              text,
            },
          },
        });
        send({
          method: "turn/completed",
          params: {
            turn: {
              id: "fake-turn",
              status: "completed",
              items: [{
                id: "fake-message",
                type: "agentMessage",
                phase: "final_answer",
                text,
              }],
            },
          },
        });
      }, 10);
    }
    return;
  }
  if (message.method === "turn/interrupt") {
    if (activity) clearInterval(activity);
    activity = null;
    send({ id: message.id, result: {} });
    return;
  }
});

process.on("SIGTERM", () => process.exit(0));
