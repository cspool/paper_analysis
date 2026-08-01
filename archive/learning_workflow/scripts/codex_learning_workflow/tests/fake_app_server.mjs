import readline from "node:readline";

process.stdin.resume();
const keepAlive = setInterval(() => {}, 60_000);
const reader = readline.createInterface({ input: process.stdin });
reader.on("close", () => clearInterval(keepAlive));
let threadCounter = 0;
let turnCounter = 0;

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function textInput(params) {
  const inputs = Array.isArray(params?.input) ? params.input : [];
  return inputs
    .filter((item) => item?.type === "text")
    .map((item) => item.text)
    .join("\n");
}

reader.on("line", (line) => {
  const message = JSON.parse(line);
  const method = message.method;
  if (method === "initialize") {
    const respond = () => send({
      id: message.id,
      result: {
        userAgent: "fake/1.0",
        codexHome: process.env.CODEX_HOME,
        platformFamily: "unix",
        platformOs: "linux",
      },
    });
    setTimeout(respond, Number(process.env.FAKE_STARTUP_DELAY_MS ?? 0));
    return;
  }
  if (method === "initialized") return;
  if (method === "thread/start") {
    threadCounter += 1;
    const threadId = `thread-${threadCounter}`;
    send({
      id: message.id,
      result: {
        thread: { id: threadId, turns: [], status: { type: "idle" } },
        model: "gpt-5.6-sol",
        modelProvider: "openai",
        cwd: message.params.cwd,
        approvalPolicy: "never",
        approvalsReviewer: "user",
        sandbox: { type: "readOnly", access: { type: "fullAccess" }, networkAccess: false },
      },
    });
    return;
  }
  if (method === "thread/resume") {
    send({
      id: message.id,
      result: {
        thread: { id: message.params.threadId, turns: [], status: { type: "idle" } },
        model: "gpt-5.6-sol",
        modelProvider: "openai",
        cwd: message.params.cwd,
        approvalPolicy: "never",
        approvalsReviewer: "user",
        sandbox: { type: "readOnly", access: { type: "fullAccess" }, networkAccess: false },
      },
    });
    return;
  }
  if (method === "model/list") {
    send({
      id: message.id,
      result: {
        data: [{
          id: "gpt-5.6-sol",
          supportedReasoningEfforts: ["low", "medium", "high", "xhigh"].map((reasoningEffort) => ({ reasoningEffort })),
        }],
      },
    });
    return;
  }
  if (method === "skills/list") {
    send({ id: message.id, result: { data: [] } });
    return;
  }
  if (method === "permissionProfile/list") {
    send({ id: message.id, result: { data: [] } });
    return;
  }
  if (method === "turn/interrupt") {
    send({ id: message.id, result: {} });
    return;
  }
  if (method === "turn/start") {
    turnCounter += 1;
    const turnId = `turn-${turnCounter}`;
    const threadId = message.params.threadId;
    const text = textInput(message.params);
    send({
      id: message.id,
      result: { turn: { id: turnId, status: "inProgress", items: [] } },
    });
    if (text.includes("CRASH_SERVER")) {
      setTimeout(() => process.exit(7), 10);
      return;
    }
    setTimeout(() => {
      const items = [
        { id: `${turnId}-u`, type: "userMessage", content: message.params.input },
      ];
      if (text.includes("TOOL_VIOLATION")) {
        items.push({
          id: `${turnId}-tool`,
          type: "commandExecution",
          command: "pwd",
          cwd: "/tmp",
          status: "completed",
          commandActions: [],
        });
      }
      if (text.includes("MCP_OK")) {
        items.push({
          id: `${turnId}-mcp`,
          type: "mcpToolCall",
          server: "obsidian",
          tool: "obsidian_search_notes",
          status: "completed",
          arguments: {},
          result: null,
          error: null,
        });
      }
      const outputText = `response:${text}`;
      items.push({ id: `${turnId}-a`, type: "agentMessage", text: outputText, phase: "final_answer" });
      for (const item of items) {
        if (item.type === "agentMessage" && text.includes("STREAM_ONLY")) {
          const midpoint = Math.floor(item.text.length / 2);
          for (const delta of [item.text.slice(0, midpoint), item.text.slice(midpoint)]) {
            send({
              method: "item/agentMessage/delta",
              params: {
                threadId,
                turnId,
                itemId: item.id,
                delta,
              },
            });
          }
        }
        send({
          method: "item/completed",
          params: {
            threadId,
            turnId,
            completedAtMs: Date.now(),
            item,
          },
        });
      }
      send({
        method: "thread/tokenUsage/updated",
        params: {
          threadId,
          turnId,
          tokenUsage: {
            total: {
              inputTokens: 10,
              cachedInputTokens: 2,
              outputTokens: 4,
              reasoningOutputTokens: 1,
              totalTokens: 14,
            },
            last: {
              inputTokens: 10,
              cachedInputTokens: 2,
              outputTokens: 4,
              reasoningOutputTokens: 1,
              totalTokens: 14,
            },
            modelContextWindow: 1000,
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
            items: text.includes("STREAM_ONLY") ? [] : items,
            ...(text.includes("STREAM_ONLY") ? { itemsView: "notLoaded" } : {}),
          },
        },
      });
    }, threadId.endsWith("1") ? 20 : 5);
    return;
  }
  if (typeof message.id === "number") {
    send({ id: message.id, result: {} });
  }
});
