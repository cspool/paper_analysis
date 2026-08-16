import { createInterface } from "node:readline";

const reader = createInterface({ input: process.stdin });
let activity = null;
let goal = null;

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
  if (message.method === "thread/resume") {
    goal = {
      threadId: message.params.threadId,
      objective: "paused fixture goal",
      status: "paused",
      tokenBudget: null,
      tokensUsed: 10,
      timeUsedSeconds: 2,
    };
    send({
      id: message.id,
      result: { thread: { id: message.params.threadId, turns: [] } },
    });
    return;
  }
  if (message.method === "thread/goal/get") {
    send({ id: message.id, result: { goal } });
    return;
  }
  if (message.method === "thread/goal/set") {
    goal = {
      threadId: message.params.threadId,
      objective: message.params.objective ?? goal?.objective ?? "",
      status: message.params.status ?? goal?.status ?? "active",
      tokenBudget: Object.hasOwn(message.params, "tokenBudget")
        ? message.params.tokenBudget
        : goal?.tokenBudget ?? null,
      tokensUsed: goal?.tokensUsed ?? 0,
      timeUsedSeconds: goal?.timeUsedSeconds ?? 0,
    };
    send({ id: message.id, result: { goal } });
    send({
      method: "thread/goal/updated",
      params: { threadId: goal.threadId, turnId: null, goal },
    });
    return;
  }
  if (message.method === "turn/start") {
    const prompt = message.params?.input?.[0]?.text ?? "";
    send({ id: message.id, result: { turn: { id: "fake-turn" } } });
    if (prompt.includes("EXP_GOAL_TRANSPORT")) {
      setTimeout(() => {
        const firstText = "第一次测量完成，Goal 保持 active 并自动续转。";
        send({
          method: "turn/started",
          params: {
            threadId: goal?.threadId ?? "fake-thread",
            turn: { id: "fake-turn", status: "inProgress" },
          },
        });
        send({
          method: "item/completed",
          params: {
            threadId: goal?.threadId ?? "fake-thread",
            turnId: "fake-turn",
            item: {
              id: "fake-goal-message",
              type: "agentMessage",
              phase: "final_answer",
              text: firstText,
            },
          },
        });
        send({
          method: "turn/completed",
          params: {
            threadId: goal?.threadId ?? "fake-thread",
            turn: {
              id: "fake-turn",
              status: "completed",
              items: [{
                id: "fake-goal-message",
                type: "agentMessage",
                phase: "final_answer",
                text: firstText,
              }],
            },
          },
        });
        setTimeout(() => {
          const text = "实验结论：受控微基准支持存在有界 headroom。";
          send({
            method: "turn/started",
            params: {
              threadId: goal?.threadId ?? "fake-thread",
              turn: { id: "fake-turn-auto", status: "inProgress" },
            },
          });
          send({
            method: "item/completed",
            params: {
              threadId: goal?.threadId ?? "fake-thread",
              turnId: "fake-turn-auto",
              item: {
                id: "fake-goal-message-auto",
                type: "agentMessage",
                phase: "final_answer",
                text,
              },
            },
          });
          send({
            method: "turn/completed",
            params: {
              threadId: goal?.threadId ?? "fake-thread",
              turn: {
                id: "fake-turn-auto",
                status: "completed",
                items: [{
                  id: "fake-goal-message-auto",
                  type: "agentMessage",
                  phase: "final_answer",
                  text,
                }],
              },
            },
          });
          goal = {
            ...goal,
            status: "complete",
            tokensUsed: 321,
            timeUsedSeconds: 7,
          };
          send({
            method: "thread/goal/updated",
            params: {
              threadId: goal?.threadId ?? "fake-thread",
              turnId: "fake-turn-auto",
              goal,
            },
          });
        }, 10);
      }, 10);
    } else if (prompt.includes("HARD_ACTIVITY")) {
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
