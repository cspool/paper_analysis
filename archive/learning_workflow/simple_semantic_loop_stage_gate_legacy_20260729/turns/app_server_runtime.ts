import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type {
  RegisteredRole,
} from "../contracts/index.ts";
import type {
  AttemptRecord,
  FreshTurnRuntime,
  FrozenTurnDispatch,
  RawTurnResult,
  ReconcileResult,
  TokenUsage,
} from "./runtime.ts";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

interface PendingTurn {
  attemptId: string;
  role: RegisteredRole;
  threadId: string;
  turnId: string;
  startedAt: number;
  resolve: (value: RawTurnResult) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
  rawEvents: unknown[];
  toolEvents: RawTurnResult["toolEvents"];
  messageOrder: string[];
  messageText: Map<string, string>;
  usage: TokenUsage;
  compacted: boolean;
  liveAgentOutputSeen: boolean;
  liveMessageItemIds: Set<string>;
}

export type AppServerSandboxMode =
  | "read-only"
  | "danger-full-access";

export type AppServerLiveEvent =
  | {
      type: "runtime_started";
      command: string;
    }
  | {
      type: "app_server_stderr";
      text: string;
    }
  | {
      type: "turn_starting";
      attemptId: string;
      taskId: string;
      role: RegisteredRole;
      model: string;
      effort: string;
      approvalPolicy: "never";
      sandbox: AppServerSandboxMode;
    }
  | {
      type: "turn_started";
      attemptId: string;
      role: RegisteredRole;
      threadId: string;
      turnId: string;
    }
  | {
      type: "agent_output_delta";
      attemptId: string;
      role: RegisteredRole;
      itemId: string;
      delta: string;
      firstChunk: boolean;
    }
  | {
      type: "tool_status";
      attemptId: string;
      role: RegisteredRole;
      phase: "started" | "completed";
      toolName: string;
      status: string;
    }
  | {
      type: "turn_completed";
      attemptId: string;
      role: RegisteredRole;
      status: RawTurnResult["status"];
      elapsedMs: number;
      usage: TokenUsage;
      toolCalls: number;
    }
  | {
      type: "turn_timeout";
      attemptId: string;
      role: RegisteredRole;
      timeoutMs: number;
    }
  | {
      type: "runtime_error";
      message: string;
    };

export interface AppServerRuntimeConfig {
  command?: string;
  argsPrefix?: string[];
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  sandbox?: AppServerSandboxMode;
  onLiveEvent?: (event: AppServerLiveEvent) => void;
}

export const APP_SERVER_RUNTIME_INVARIANTS = Object.freeze({
  ephemeralThreadPerAttempt: true,
  resumesProviderThread: false,
  createsGoal: false,
  structuredOutputSchema: true,
  approvalPolicy: "never",
  defaultSandbox: "read-only",
  yoloSandbox: "danger-full-access",
} as const);

export class AppServerFreshTurnRuntime implements FreshTurnRuntime {
  private readonly config: AppServerRuntimeConfig;
  private process: ChildProcessWithoutNullStreams | null = null;
  private reader: Interface | null = null;
  private nextRequestId = 1;
  private requests = new Map<number, PendingRequest>();
  private turns = new Map<string, PendingTurn>();
  private attemptToTurn = new Map<string, string>();
  private earlyTurnMessages = new Map<string, Record<string, unknown>[]>();
  private starting: Promise<void> | null = null;
  private closing = false;
  private readonly sandbox: AppServerSandboxMode;

  constructor(config: AppServerRuntimeConfig = {}) {
    this.config = config;
    this.sandbox =
      config.sandbox ?? APP_SERVER_RUNTIME_INVARIANTS.defaultSandbox;
  }

  async run(dispatch: FrozenTurnDispatch): Promise<RawTurnResult> {
    await this.start();
    const evidence = dispatch.role === "evidence_reader";
    this.emit({
      type: "turn_starting",
      attemptId: dispatch.attemptId,
      taskId: dispatch.taskId,
      role: dispatch.role,
      model: dispatch.model,
      effort: dispatch.providerWireEffort,
      approvalPolicy: APP_SERVER_RUNTIME_INVARIANTS.approvalPolicy,
      sandbox: this.sandbox,
    });
    const threadResponse = asObject(
      await this.request(
        "thread/start",
        {
          model: dispatch.model,
          allowProviderModelFallback: false,
          ephemeral: APP_SERVER_RUNTIME_INVARIANTS.ephemeralThreadPerAttempt,
          approvalPolicy: APP_SERVER_RUNTIME_INVARIANTS.approvalPolicy,
          sandbox: this.sandbox,
          cwd: dispatch.cwd,
          config: {
            features: { apps: false },
            mcp_servers: {
              obsidian: { enabled: evidence },
            },
          },
          developerInstructions: [
            "You are one fresh Turn controlled by a deterministic external Controller.",
            "Do not create, invoke, delegate to, or manage another agent.",
            "Do not use Goals, checkpoints, provider history, or Session continuation.",
            evidence
              ? "Only the two task-authorized read-only Obsidian tools are allowed."
              : "Do not call any tool.",
            "Return exactly the one JSON protocol requested in the prompt, then terminate.",
          ].join("\n"),
        },
        this.config.requestTimeoutMs,
      ),
    );
    const threadId = getString(asObject(threadResponse?.thread), "id");
    if (!threadId) throw new Error("thread/start returned no fresh thread ID");
    const turnResponse = asObject(
      await this.request(
        "turn/start",
        {
          threadId,
          model: dispatch.model,
          effort: dispatch.providerWireEffort,
          input: [{ type: "text", text: dispatch.prompt }],
          outputSchema: dispatch.outputSchema,
        },
        this.config.requestTimeoutMs,
      ),
    );
    const turnId = getString(asObject(turnResponse?.turn), "id");
    if (!turnId) throw new Error("turn/start returned no Turn ID");
    this.emit({
      type: "turn_started",
      attemptId: dispatch.attemptId,
      role: dispatch.role,
      threadId,
      turnId,
    });
    return new Promise<RawTurnResult>((resolve, reject) => {
      const timer = setTimeout(async () => {
        this.turns.delete(turnId);
        this.attemptToTurn.delete(dispatch.attemptId);
        this.emit({
          type: "turn_timeout",
          attemptId: dispatch.attemptId,
          role: dispatch.role,
          timeoutMs: dispatch.timeoutMs,
        });
        try {
          await this.request(
            "turn/interrupt",
            { threadId, turnId },
            Math.min(10_000, dispatch.timeoutMs),
          );
        } catch {
          // Preserve timeout as primary failure.
        }
        reject(new Error(`Turn timed out after ${dispatch.timeoutMs} ms`));
      }, dispatch.timeoutMs);
      this.turns.set(turnId, {
        attemptId: dispatch.attemptId,
        role: dispatch.role,
        threadId,
        turnId,
        startedAt: Date.now(),
        resolve,
        reject,
        timer,
        rawEvents: [],
        toolEvents: [],
        messageOrder: [],
        messageText: new Map(),
        usage: zeroUsage(),
        compacted: false,
        liveAgentOutputSeen: false,
        liveMessageItemIds: new Set(),
      });
      const pending = this.turns.get(turnId)!;
      pending.rawEvents.push({
        method: "controller/runtimePolicy",
        params: {
          approvalPolicy: APP_SERVER_RUNTIME_INVARIANTS.approvalPolicy,
          sandbox: this.sandbox,
          ephemeralThread: true,
          providerHistoryResumed: false,
        },
      });
      this.attemptToTurn.set(dispatch.attemptId, turnId);
      this.replayEarlyTurnMessages(turnId);
    });
  }

  async cancel(attemptId: string): Promise<void> {
    const turnId = this.attemptToTurn.get(attemptId);
    const turn = turnId ? this.turns.get(turnId) : undefined;
    if (!turn) return;
    await this.request(
      "turn/interrupt",
      { threadId: turn.threadId, turnId: turn.turnId },
      10_000,
    );
  }

  async reconcile(_attempt: AttemptRecord): Promise<ReconcileResult> {
    // Provider history is deliberately non-authoritative and ephemeral.
    // A missing Controller commit is reconciled from captured DB/raw artifacts,
    // never by resuming the old thread.
    return { status: "unknown", recoverableResult: null };
  }

  async close(): Promise<void> {
    this.closing = true;
    for (const turn of this.turns.values()) {
      clearTimeout(turn.timer);
      turn.reject(new Error("runtime closed"));
    }
    this.turns.clear();
    this.attemptToTurn.clear();
    this.earlyTurnMessages.clear();
    this.reader?.close();
    this.process?.kill("SIGTERM");
    this.process = null;
  }

  async probeModel(modelId: string): Promise<{
    model: Record<string, unknown>;
    supportedReasoningEfforts: string[];
  }> {
    await this.start();
    const response = asObject(
      await this.request("model/list", { includeHidden: true, limit: 100 }),
    );
    const models = Array.isArray(response?.data)
      ? response.data.map(asObject).filter(Boolean) as Record<string, unknown>[]
      : [];
    const model = models.find((candidate) => candidate.id === modelId);
    if (!model) throw new Error(`required model is unavailable: ${modelId}`);
    const supportedReasoningEfforts = Array.isArray(
      model.supportedReasoningEfforts,
    )
      ? model.supportedReasoningEfforts
          .map(asObject)
          .map((entry) => getString(entry, "reasoningEffort"))
          .filter((entry): entry is string => Boolean(entry))
      : [];
    return { model, supportedReasoningEfforts };
  }

  async probeSkills(cwd: string): Promise<Set<string>> {
    await this.start();
    const response = asObject(
      await this.request("skills/list", { cwds: [cwd], forceReload: true }),
    );
    const names = new Set<string>();
    const groups = Array.isArray(response?.data) ? response.data : [];
    for (const rawGroup of groups) {
      const group = asObject(rawGroup);
      const skills = Array.isArray(group?.skills) ? group.skills : [];
      for (const rawSkill of skills) {
        const name = getString(asObject(rawSkill), "name");
        if (name) names.add(name);
      }
    }
    return names;
  }

  private async start(): Promise<void> {
    if (this.process) return;
    if (this.starting) return this.starting;
    this.starting = this.startImpl();
    try {
      await this.starting;
    } finally {
      this.starting = null;
    }
  }

  private async startImpl(): Promise<void> {
    const command = this.config.command ?? "codex";
    const args = [
      ...(this.config.argsPrefix ?? []),
      "app-server",
      "--listen",
      "stdio://",
    ];
    this.process = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });
    this.emit({ type: "runtime_started", command: `${command} ${args.join(" ")}` });
    this.process.stderr.setEncoding("utf8");
    this.process.stderr.on("data", (chunk: string) => {
      this.emit({ type: "app_server_stderr", text: chunk });
    });
    this.process.on("error", (error) => this.failAll(error));
    this.process.on("exit", (code, signal) => {
      if (!this.closing) {
        this.failAll(
          new Error(
            `codex app-server exited unexpectedly (code=${code}, signal=${signal})`,
          ),
        );
      }
    });
    this.reader = createInterface({ input: this.process.stdout });
    this.reader.on("line", (line) => this.onLine(line));
    const initialized = asObject(
      await this.request(
        "initialize",
        {
          clientInfo: {
            name: "simple_semantic_loop",
            title: "Simple Semantic Loop Controller",
            version: "1.0.0",
          },
          capabilities: { experimentalApi: false },
        },
        this.config.startupTimeoutMs ?? 30_000,
      ),
    );
    if (!initialized?.codexHome) {
      throw new Error("app-server initialize response is missing codexHome");
    }
    this.notify("initialized", {});
  }

  private request(
    method: string,
    params: unknown,
    timeoutMs = 30_000,
  ): Promise<unknown> {
    if (!this.process?.stdin.writable) {
      return Promise.reject(new Error("app-server is not running"));
    }
    const id = this.nextRequestId++;
    const response = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.requests.delete(id);
        reject(new Error(`${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.requests.set(id, { resolve, reject, timer });
    });
    this.process.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
    return response;
  }

  private notify(method: string, params: unknown): void {
    if (!this.process?.stdin.writable) {
      throw new Error("app-server is not running");
    }
    this.process.stdin.write(`${JSON.stringify({ method, params })}\n`);
  }

  private onLine(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line);
    } catch {
      this.failAll(new Error("app-server emitted malformed JSONL"));
      return;
    }
    if (typeof message.id === "number" && !message.method) {
      const pending = this.requests.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.requests.delete(message.id);
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
      else pending.resolve(message.result);
      return;
    }
    if (typeof message.id === "number" && typeof message.method === "string") {
      this.process?.stdin.write(
        `${JSON.stringify({
          id: message.id,
          error: {
            code: -32001,
            message: "interactive requests are disabled",
          },
        })}\n`,
      );
      return;
    }
    const method = typeof message.method === "string" ? message.method : "";
    const params = asObject(message.params);
    const turn = asObject(params?.turn);
    const item = asObject(params?.item);
    const turnId =
      getString(params, "turnId") ?? getString(turn, "id");
    const pending = turnId ? this.turns.get(turnId) : undefined;
    if (!pending && turnId && isTurnNotification(method)) {
      this.bufferEarlyTurnMessage(turnId, message);
      return;
    }
    if (pending) {
      pending.rawEvents.push(message);
      if (method === "item/agentMessage/delta") {
        const itemId = getString(params, "itemId");
        const delta = getString(params, "delta");
        if (itemId && delta !== null) {
          rememberMessage(pending, itemId, delta, false);
          const firstChunk = !pending.liveAgentOutputSeen;
          pending.liveAgentOutputSeen = true;
          pending.liveMessageItemIds.add(itemId);
          this.emit({
            type: "agent_output_delta",
            attemptId: pending.attemptId,
            role: pending.role,
            itemId,
            delta,
            firstChunk,
          });
        }
      }
      if (method === "item/completed" || method === "item/started") {
        const itemType = getString(item, "type");
        if (itemType === "agentMessage" && method === "item/completed") {
          const itemId = getString(item, "id");
          const text = getString(item, "text");
          if (itemId && text !== null) {
            rememberMessage(pending, itemId, text, true);
            if (!pending.liveMessageItemIds.has(itemId)) {
              const firstChunk = !pending.liveAgentOutputSeen;
              pending.liveAgentOutputSeen = true;
              pending.liveMessageItemIds.add(itemId);
              this.emit({
                type: "agent_output_delta",
                attemptId: pending.attemptId,
                role: pending.role,
                itemId,
                delta: text,
                firstChunk,
              });
            }
          }
        }
        if (
          [
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
            "collabToolCall",
            "webSearch",
            "imageView",
          ].includes(itemType ?? "")
        ) {
          this.emit({
            type: "tool_status",
            attemptId: pending.attemptId,
            role: pending.role,
            phase: method === "item/started" ? "started" : "completed",
            toolName: normalizeToolName(itemType, item),
            status: getString(item, "status") ?? "unknown",
          });
        }
        if (
          method === "item/completed" &&
          [
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
            "collabToolCall",
            "webSearch",
            "imageView",
          ].includes(itemType ?? "")
        ) {
          pending.toolEvents.push({
            toolName: normalizeToolName(itemType, item),
            arguments:
              asObject(item?.arguments) ??
              asObject(item?.input) ??
              {},
            status: getString(item, "status") ?? "unknown",
            resultText: normalizeToolResultText(item),
            error: item?.error ?? null,
          });
        }
      }
      if (method === "thread/tokenUsage/updated") {
        pending.usage = normalizeUsage(params?.tokenUsage);
      }
      if (
        method === "thread/compacted" ||
        getString(item, "type") === "contextCompaction"
      ) {
        pending.compacted = true;
      }
    }
    if (method === "turn/completed" && turnId) this.completeTurn(turnId, turn);
  }

  private bufferEarlyTurnMessage(
    turnId: string,
    message: Record<string, unknown>,
  ): void {
    if (
      !this.earlyTurnMessages.has(turnId) &&
      this.earlyTurnMessages.size >= 32
    ) {
      const oldest = this.earlyTurnMessages.keys().next().value as
        | string
        | undefined;
      if (oldest) this.earlyTurnMessages.delete(oldest);
    }
    const messages = this.earlyTurnMessages.get(turnId) ?? [];
    if (messages.length < 256) messages.push(message);
    this.earlyTurnMessages.set(turnId, messages);
  }

  private replayEarlyTurnMessages(turnId: string): void {
    const messages = this.earlyTurnMessages.get(turnId);
    if (!messages) return;
    this.earlyTurnMessages.delete(turnId);
    for (const message of messages) {
      this.onLine(JSON.stringify(message));
    }
  }

  private completeTurn(
    turnId: string,
    turn: Record<string, unknown> | null,
  ): void {
    const pending = this.turns.get(turnId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.turns.delete(turnId);
    this.attemptToTurn.delete(pending.attemptId);
    const rawStatus = getString(turn, "status");
    const status: RawTurnResult["status"] =
      rawStatus === "completed"
        ? "completed"
        : rawStatus === "interrupted"
          ? "interrupted"
          : "failed";
    const completedText = extractCompletedText(turn);
    const streamedText = pending.messageOrder
      .map((id) => pending.messageText.get(id) ?? "")
      .filter(Boolean)
      .join("\n")
      .trim();
    if (!pending.liveAgentOutputSeen && completedText) {
      pending.liveAgentOutputSeen = true;
      this.emit({
        type: "agent_output_delta",
        attemptId: pending.attemptId,
        role: pending.role,
        itemId: "turn-completed-text",
        delta: completedText,
        firstChunk: true,
      });
    }
    const elapsedMs = Date.now() - pending.startedAt;
    const result: RawTurnResult = {
      attemptId: pending.attemptId,
      providerThreadId: pending.threadId,
      providerTurnId: pending.turnId,
      status,
      text: completedText || streamedText,
      usage: pending.usage,
      toolEvents: pending.toolEvents,
      rawEvents: pending.rawEvents,
      compacted: pending.compacted,
      error: getString(asObject(turn?.error), "message"),
      elapsedMs,
    };
    this.emit({
      type: "turn_completed",
      attemptId: pending.attemptId,
      role: pending.role,
      status,
      elapsedMs,
      usage: pending.usage,
      toolCalls: pending.toolEvents.length,
    });
    pending.resolve(result);
  }

  private failAll(error: Error): void {
    this.emit({ type: "runtime_error", message: error.message });
    for (const request of this.requests.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.requests.clear();
    for (const turn of this.turns.values()) {
      clearTimeout(turn.timer);
      turn.reject(error);
    }
    this.turns.clear();
    this.attemptToTurn.clear();
    this.earlyTurnMessages.clear();
  }

  private emit(event: AppServerLiveEvent): void {
    try {
      this.config.onLiveEvent?.(event);
    } catch {
      // Console/reporting failures must not change workflow semantics.
    }
  }
}

function isTurnNotification(method: string): boolean {
  return (
    method.startsWith("item/") ||
    method === "turn/completed" ||
    method === "thread/tokenUsage/updated" ||
    method === "thread/compacted"
  );
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function getString(
  value: Record<string, unknown> | null,
  key: string,
): string | null {
  return typeof value?.[key] === "string" ? (value[key] as string) : null;
}

function rememberMessage(
  pending: PendingTurn,
  id: string,
  text: string,
  replace: boolean,
): void {
  if (!pending.messageText.has(id)) pending.messageOrder.push(id);
  pending.messageText.set(
    id,
    replace ? text : `${pending.messageText.get(id) ?? ""}${text}`,
  );
}

function extractCompletedText(turn: Record<string, unknown> | null): string {
  const items = Array.isArray(turn?.items) ? turn.items : [];
  return items
    .map(asObject)
    .filter((item) => getString(item, "type") === "agentMessage")
    .map((item) => getString(item, "text") ?? "")
    .filter(Boolean)
    .join("\n")
    .trim();
}

function normalizeUsage(value: unknown): TokenUsage {
  const last = asObject(asObject(value)?.last);
  return {
    inputTokens: Number(last?.inputTokens ?? 0),
    cachedInputTokens: Number(last?.cachedInputTokens ?? 0),
    outputTokens: Number(last?.outputTokens ?? 0),
    reasoningOutputTokens: Number(last?.reasoningOutputTokens ?? 0),
    totalTokens: Number(last?.totalTokens ?? 0),
  };
}

function zeroUsage(): TokenUsage {
  return {
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningOutputTokens: 0,
    totalTokens: 0,
  };
}

function normalizeToolName(
  itemType: string | null,
  item: Record<string, unknown> | null,
): string {
  const tool = getString(item, "tool");
  if (itemType === "mcpToolCall") {
    if (tool?.startsWith("mcp__")) return tool;
    const server = getString(item, "server");
    if (server && tool) return `mcp__${server}__${tool}`;
  }
  return tool ?? getString(item, "name") ?? itemType ?? "unknown";
}

function normalizeToolResultText(
  item: Record<string, unknown> | null,
): string {
  const result = asObject(item?.result);
  const content = Array.isArray(result?.content) ? result.content : [];
  const parts = content
    .map(asObject)
    .map((block) => getString(block, "text") ?? "")
    .filter(Boolean);
  if (result?.structuredContent !== undefined) {
    parts.push(JSON.stringify(result.structuredContent));
  }
  const contentItems = Array.isArray(item?.contentItems)
    ? item.contentItems
    : [];
  for (const raw of contentItems) {
    const block = asObject(raw);
    const text = getString(block, "text");
    if (text) parts.push(text);
  }
  return parts.join("\n");
}
