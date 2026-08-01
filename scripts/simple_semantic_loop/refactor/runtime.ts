import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type {
  LoopRole,
  OutputCaptureState,
  RawTurnResult,
  RuntimeFailureKind,
  RuntimePersistenceEvent,
  RuntimeToolEvent,
  TokenUsage,
  TurnDispatch,
  TurnRuntime,
} from "./types.ts";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

interface PendingTurn {
  dispatch: TurnDispatch;
  threadId: string;
  providerTurnId: string;
  startedAt: number;
  resolve: (value: RawTurnResult) => void;
  reject: (error: Error) => void;
  idleTimer: NodeJS.Timeout;
  hardTimer: NodeJS.Timeout;
  rawEvents: unknown[];
  toolEvents: RuntimeToolEvent[];
  messageOrder: string[];
  messageText: Map<string, string>;
  messagePhase: Map<string, MessagePhase>;
  completedMessageIds: Set<string>;
  usage: TokenUsage;
  compacted: boolean;
  liveMessageIds: Set<string>;
  deltaBuffers: Map<string, { at: string; delta: string }>;
  deltaFlushTimers: Map<string, NodeJS.Timeout>;
  lastActivityAt: number;
  timingOut: boolean;
  timeoutKind: Extract<RuntimeFailureKind, "IDLE_TIMEOUT" | "HARD_TIMEOUT"> | null;
  interruptError: string | null;
}

type MessagePhase = "commentary" | "final_answer" | null;

const OUTPUT_DELTA_FLUSH_INTERVAL_MS = 100;
const OUTPUT_DELTA_FLUSH_CHAR_LIMIT = 2_048;

export interface AgentMessageCandidate {
  text: string;
  phase: MessagePhase;
}

export interface ProtocolMessageSelection {
  text: string;
  error: string | null;
}

export type SandboxMode = "read-only" | "danger-full-access";

export type RuntimeLiveEvent =
  | { type: "runtime_started"; command: string }
  | { type: "app_server_stderr"; text: string }
  | {
      type: "turn_starting";
      turnId: string;
      role: LoopRole;
      model: string;
      effort: "high" | "max";
      approvalPolicy: "never";
      sandbox: SandboxMode;
    }
  | {
      type: "turn_started";
      turnId: string;
      role: LoopRole;
      threadId: string;
      providerTurnId: string;
    }
  | {
      type: "agent_output_delta";
      turnId: string;
      role: LoopRole;
      itemId: string;
      delta: string;
    }
  | {
      type: "agent_message_complete";
      turnId: string;
      role: LoopRole;
      itemId: string;
      phase: MessagePhase;
    }
  | {
      type: "tool_status";
      turnId: string;
      role: LoopRole;
      phase: "started" | "completed";
      toolName: string;
      status: string;
    }
  | {
      type: "turn_completed";
      turnId: string;
      role: LoopRole;
      status: RawTurnResult["status"];
      elapsedMs: number;
      usage: TokenUsage;
      toolCalls: number;
    }
  | {
      type: "turn_timeout";
      turnId: string;
      role: LoopRole;
      timeoutMs: number;
      kind: Extract<RuntimeFailureKind, "IDLE_TIMEOUT" | "HARD_TIMEOUT">;
      capture: OutputCaptureState;
    }
  | { type: "runtime_error"; message: string };

export interface CodexRuntimeConfig {
  command?: string;
  argsPrefix?: string[];
  sandbox?: SandboxMode;
  startupTimeoutMs?: number;
  requestTimeoutMs?: number;
  onLiveEvent?: (event: RuntimeLiveEvent) => void;
}

export class CodexAppServerRuntime implements TurnRuntime {
  private readonly config: CodexRuntimeConfig;
  private readonly sandbox: SandboxMode;
  private process: ChildProcessWithoutNullStreams | null = null;
  private reader: Interface | null = null;
  private starting: Promise<void> | null = null;
  private closing = false;
  private nextRequestId = 1;
  private requests = new Map<number, PendingRequest>();
  private turns = new Map<string, PendingTurn>();
  private earlyMessages = new Map<string, Record<string, unknown>[]>();

  constructor(config: CodexRuntimeConfig = {}) {
    this.config = config;
    this.sandbox = config.sandbox ?? "read-only";
  }

  async run(dispatch: TurnDispatch): Promise<RawTurnResult> {
    await this.start();
    this.emit({
      type: "turn_starting",
      turnId: dispatch.turnId,
      role: dispatch.role,
      model: dispatch.model,
      effort: dispatch.effort,
      approvalPolicy: "never",
      sandbox: this.sandbox,
    });

    const threadResponse = asObject(
      await this.request(
        "thread/start",
        {
          model: dispatch.model,
          allowProviderModelFallback: false,
          ephemeral: true,
          approvalPolicy: "never",
          sandbox: this.sandbox,
          cwd: dispatch.cwd,
          developerInstructions: developerInstructions(dispatch.role),
        },
        this.config.requestTimeoutMs,
      ),
    );
    const threadId = getString(asObject(threadResponse?.thread), "id");
    if (!threadId) throw new Error("thread/start returned no fresh thread ID");

    const turnParams: Record<string, unknown> = {
      threadId,
      model: dispatch.model,
      effort: dispatch.effort,
      input: [{ type: "text", text: dispatch.prompt }],
    };
    if (dispatch.outputSchema) turnParams.outputSchema = dispatch.outputSchema;
    const turnResponse = asObject(
      await this.request(
        "turn/start",
        turnParams,
        this.config.requestTimeoutMs,
      ),
    );
    const providerTurnId = getString(asObject(turnResponse?.turn), "id");
    if (!providerTurnId) throw new Error("turn/start returned no Turn ID");
    this.emit({
      type: "turn_started",
      turnId: dispatch.turnId,
      role: dispatch.role,
      threadId,
      providerTurnId,
    });
    this.persist(dispatch, {
      type: "provider_started",
      at: nowIso(),
      threadId,
      providerTurnId,
    });

    return new Promise<RawTurnResult>((resolve, reject) => {
      const startedAt = Date.now();
      const pending: PendingTurn = {
        dispatch,
        threadId,
        providerTurnId,
        startedAt,
        resolve,
        reject,
        idleTimer: setTimeout(() => {}, 0),
        hardTimer: setTimeout(() => {}, 0),
        rawEvents: [
          {
            method: "controller/runtimePolicy",
            params: {
              approvalPolicy: "never",
              sandbox: this.sandbox,
              ephemeralThread: true,
              providerHistoryResumed: false,
            },
          },
        ],
        toolEvents: [],
        messageOrder: [],
        messageText: new Map(),
        messagePhase: new Map(),
        completedMessageIds: new Set(),
        usage: zeroUsage(),
        compacted: false,
        liveMessageIds: new Set(),
        deltaBuffers: new Map(),
        deltaFlushTimers: new Map(),
        lastActivityAt: startedAt,
        timingOut: false,
        timeoutKind: null,
        interruptError: null,
      };
      clearTimeout(pending.idleTimer);
      clearTimeout(pending.hardTimer);
      pending.idleTimer = setTimeout(
        () => void this.beginTimeout(pending, "IDLE_TIMEOUT"),
        dispatch.timeoutProfile.idleTimeoutMs,
      );
      pending.hardTimer = setTimeout(
        () => void this.beginTimeout(pending, "HARD_TIMEOUT"),
        dispatch.timeoutProfile.hardTimeoutMs,
      );
      this.turns.set(providerTurnId, pending);
      this.replayEarlyMessages(providerTurnId);
    });
  }

  async probeModel(modelId: string): Promise<string[]> {
    await this.start();
    const response = asObject(
      await this.request("model/list", { includeHidden: true, limit: 100 }),
    );
    const models = Array.isArray(response?.data)
      ? response.data.map(asObject).filter(Boolean)
      : [];
    const model = models.find((candidate) => candidate?.id === modelId);
    if (!model) throw new Error(`required model is unavailable: ${modelId}`);
    return Array.isArray(model.supportedReasoningEfforts)
      ? model.supportedReasoningEfforts
          .map(asObject)
          .map((item) => getString(item, "reasoningEffort"))
          .filter((item): item is string => Boolean(item))
      : [];
  }

  async probeSkills(cwd: string): Promise<Set<string>> {
    await this.start();
    const response = asObject(
      await this.request("skills/list", { cwds: [cwd], forceReload: true }),
    );
    const names = new Set<string>();
    for (const rawGroup of Array.isArray(response?.data) ? response.data : []) {
      const group = asObject(rawGroup);
      for (const rawSkill of Array.isArray(group?.skills) ? group.skills : []) {
        const name = getString(asObject(rawSkill), "name");
        if (name) names.add(name);
      }
    }
    return names;
  }

  async close(): Promise<void> {
    this.closing = true;
    for (const pending of this.turns.values()) {
      clearTimeout(pending.idleTimer);
      clearTimeout(pending.hardTimer);
      this.flushAllOutputDeltas(pending);
      pending.reject(new Error("runtime closed"));
    }
    for (const request of this.requests.values()) {
      clearTimeout(request.timer);
      request.reject(new Error("runtime closed"));
    }
    this.turns.clear();
    this.requests.clear();
    this.earlyMessages.clear();
    this.reader?.close();
    this.process?.kill("SIGTERM");
    this.process = null;
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
    this.emit({
      type: "runtime_started",
      command: `${command} ${args.join(" ")}`,
    });
    this.process.stderr.setEncoding("utf8");
    this.process.stderr.on("data", (text: string) =>
      this.emit({ type: "app_server_stderr", text }),
    );
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
            name: "learning_simple_semantic_loop",
            title: "Learning Simple Semantic Loop",
            version: "2.0.0",
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
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      this.failAll(new Error("app-server emitted malformed JSONL"));
      return;
    }
    if (typeof message.id === "number" && !message.method) {
      const pending = this.requests.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.requests.delete(message.id);
      if (message.error) {
        pending.reject(new Error(JSON.stringify(message.error)));
      } else {
        pending.resolve(message.result);
      }
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
    const providerTurnId =
      getString(params, "turnId") ?? getString(turn, "id");
    const pending = providerTurnId
      ? this.turns.get(providerTurnId)
      : undefined;
    if (!pending && providerTurnId && isTurnNotification(method)) {
      this.bufferEarlyMessage(providerTurnId, message);
      return;
    }
    if (pending) {
      pending.rawEvents.push(message);
      const isAgentMessageDelta = method === "item/agentMessage/delta";
      if (!isAgentMessageDelta) {
        this.persist(pending.dispatch, {
          type: "raw_event",
          at: nowIso(),
          event: message,
        });
      }
      if (isMeaningfulActivity(method, item)) this.touch(pending);
      if (isAgentMessageDelta) {
        const itemId = getString(params, "itemId");
        const delta = getString(params, "delta");
        if (itemId && delta !== null) {
          rememberMessage(pending, itemId, delta, false);
          pending.liveMessageIds.add(itemId);
          this.queueOutputDelta(pending, itemId, delta);
          this.emit({
            type: "agent_output_delta",
            turnId: pending.dispatch.turnId,
            role: pending.dispatch.role,
            itemId,
            delta,
          });
        }
      }
      if (method === "item/started" || method === "item/completed") {
        const itemType = getString(item, "type");
        if (itemType === "agentMessage" && method === "item/completed") {
          const itemId = getString(item, "id");
          const text = getString(item, "text");
          if (itemId && text !== null) {
            rememberMessage(pending, itemId, text, true);
            this.flushOutputDelta(pending, itemId);
            const phase = getMessagePhase(item);
            pending.messagePhase.set(itemId, phase);
            pending.completedMessageIds.add(itemId);
            this.persist(pending.dispatch, {
              type: "message_completed",
              at: nowIso(),
              itemId,
              phase,
              text,
            });
            this.emit({
              type: "agent_message_complete",
              turnId: pending.dispatch.turnId,
              role: pending.dispatch.role,
              itemId,
              phase,
            });
            if (!pending.liveMessageIds.has(itemId)) {
              pending.liveMessageIds.add(itemId);
              this.emit({
                type: "agent_output_delta",
                turnId: pending.dispatch.turnId,
                role: pending.dispatch.role,
                itemId,
                delta: text,
              });
            }
          }
        }
        if (isToolItem(itemType)) {
          const phase = method === "item/started" ? "started" : "completed";
          const toolName = normalizeToolName(itemType, item);
          const toolEvent: RuntimeToolEvent = {
            toolName,
            status: getString(item, "status") ?? "unknown",
            arguments: normalizeToolArguments(itemType, item),
            resultSummary:
              method === "item/completed" ? normalizeToolResult(item) : null,
            error: method === "item/completed" ? item?.error ?? null : null,
          };
          this.persist(pending.dispatch, {
            type: "tool",
            at: nowIso(),
            phase,
            event: toolEvent,
          });
          this.emit({
            type: "tool_status",
            turnId: pending.dispatch.turnId,
            role: pending.dispatch.role,
            phase,
            toolName,
            status: getString(item, "status") ?? "unknown",
          });
          if (method === "item/completed") {
            pending.toolEvents.push(toolEvent);
          }
        }
      }
      if (method === "thread/tokenUsage/updated") {
        pending.usage = normalizeUsage(params?.tokenUsage);
        this.persist(pending.dispatch, {
          type: "usage",
          at: nowIso(),
          usage: pending.usage,
        });
      }
      if (
        method === "thread/compacted" ||
        getString(item, "type") === "contextCompaction"
      ) {
        pending.compacted = true;
        this.persist(pending.dispatch, {
          type: "compacted",
          at: nowIso(),
        });
      }
    }
    if (method === "turn/completed" && providerTurnId) {
      this.completeTurn(providerTurnId, turn);
    }
  }

  private completeTurn(
    providerTurnId: string,
    turn: Record<string, unknown> | null,
  ): void {
    const pending = this.turns.get(providerTurnId);
    if (!pending) return;
    clearTimeout(pending.idleTimer);
    clearTimeout(pending.hardTimer);
    this.flushAllOutputDeltas(pending);
    this.turns.delete(providerTurnId);
    const rawStatus = getString(turn, "status");
    const status: RawTurnResult["status"] =
      rawStatus === "completed"
        ? "completed"
        : rawStatus === "interrupted"
          ? "interrupted"
          : "failed";
    const protocolMessage = selectProtocolAgentMessage(
      turn,
      pending.messageOrder
        .filter((id) => pending.completedMessageIds.has(id))
        .map((id) => ({
          text: pending.messageText.get(id) ?? "",
          phase: pending.messagePhase.get(id) ?? null,
        })),
    );
    if (pending.liveMessageIds.size === 0 && protocolMessage.text) {
      this.emit({
        type: "agent_output_delta",
        turnId: pending.dispatch.turnId,
        role: pending.dispatch.role,
        itemId: "turn-completed-text",
        delta: protocolMessage.text,
      });
    }
    const elapsedMs = Date.now() - pending.startedAt;
    const effectiveStatus: RawTurnResult["status"] =
      status === "completed" && protocolMessage.error ? "failed" : status;
    const capture = captureState(pending);
    const partialText = partialMessageText(pending);
    pending.rawEvents.push({
      method: "controller/protocolMessageSelection",
      params: {
        selected: protocolMessage.error === null,
        error: protocolMessage.error,
      },
    });
    const result: RawTurnResult = {
      status: effectiveStatus,
      text: protocolMessage.text,
      providerThreadId: pending.threadId,
      providerTurnId,
      usage: pending.usage,
      toolEvents: pending.toolEvents,
      rawEvents: pending.rawEvents,
      compacted: pending.compacted,
      outputCapture: protocolMessage.text ? "COMPLETE" : capture,
      partialText,
      failureKind:
        pending.timeoutKind ??
        (effectiveStatus === "completed" ? null : "PROVIDER_ERROR"),
      interruptError: pending.interruptError,
      lastActivityAt: new Date(pending.lastActivityAt).toISOString(),
      incrementalEventsPersisted: Boolean(
        pending.dispatch.onRuntimeEvent,
      ),
      error:
        getString(asObject(turn?.error), "message") ?? protocolMessage.error,
      elapsedMs,
    };
    this.emit({
      type: "turn_completed",
      turnId: pending.dispatch.turnId,
      role: pending.dispatch.role,
      status: effectiveStatus,
      elapsedMs,
      usage: pending.usage,
      toolCalls: pending.toolEvents.length,
    });
    pending.resolve(result);
  }

  private touch(pending: PendingTurn): void {
    if (pending.timingOut) return;
    pending.lastActivityAt = Date.now();
    clearTimeout(pending.idleTimer);
    pending.idleTimer = setTimeout(
      () => void this.beginTimeout(pending, "IDLE_TIMEOUT"),
      pending.dispatch.timeoutProfile.idleTimeoutMs,
    );
  }

  private queueOutputDelta(
    pending: PendingTurn,
    itemId: string,
    delta: string,
  ): void {
    const existing = pending.deltaBuffers.get(itemId);
    pending.deltaBuffers.set(itemId, {
      at: existing?.at ?? nowIso(),
      delta: `${existing?.delta ?? ""}${delta}`,
    });
    if ((existing?.delta.length ?? 0) + delta.length >= OUTPUT_DELTA_FLUSH_CHAR_LIMIT) {
      this.flushOutputDelta(pending, itemId);
      return;
    }
    if (!pending.deltaFlushTimers.has(itemId)) {
      pending.deltaFlushTimers.set(
        itemId,
        setTimeout(
          () => this.flushOutputDelta(pending, itemId),
          OUTPUT_DELTA_FLUSH_INTERVAL_MS,
        ),
      );
    }
  }

  private flushOutputDelta(pending: PendingTurn, itemId: string): void {
    const timer = pending.deltaFlushTimers.get(itemId);
    if (timer) clearTimeout(timer);
    pending.deltaFlushTimers.delete(itemId);
    const buffered = pending.deltaBuffers.get(itemId);
    if (!buffered) return;
    pending.deltaBuffers.delete(itemId);
    this.persist(pending.dispatch, {
      type: "output_delta",
      at: buffered.at,
      itemId,
      delta: buffered.delta,
    });
  }

  private flushAllOutputDeltas(pending: PendingTurn): void {
    for (const itemId of [...pending.deltaBuffers.keys()]) {
      this.flushOutputDelta(pending, itemId);
    }
    for (const timer of pending.deltaFlushTimers.values()) clearTimeout(timer);
    pending.deltaFlushTimers.clear();
  }

  private async beginTimeout(
    pending: PendingTurn,
    kind: Extract<RuntimeFailureKind, "IDLE_TIMEOUT" | "HARD_TIMEOUT">,
  ): Promise<void> {
    if (pending.timingOut || !this.turns.has(pending.providerTurnId)) return;
    pending.timingOut = true;
    pending.timeoutKind = kind;
    clearTimeout(pending.idleTimer);
    clearTimeout(pending.hardTimer);
    this.flushAllOutputDeltas(pending);
    const capture = captureState(pending);
    this.persist(pending.dispatch, {
      type: "timeout",
      at: nowIso(),
      kind,
      capture,
      lastActivityAt: new Date(pending.lastActivityAt).toISOString(),
    });
    this.emit({
      type: "turn_timeout",
      turnId: pending.dispatch.turnId,
      role: pending.dispatch.role,
      timeoutMs:
        kind === "IDLE_TIMEOUT"
          ? pending.dispatch.timeoutProfile.idleTimeoutMs
          : pending.dispatch.timeoutProfile.hardTimeoutMs,
      kind,
      capture,
    });

    try {
      await this.request(
        "turn/interrupt",
        { threadId: pending.threadId, turnId: pending.providerTurnId },
        Math.min(
          10_000,
          pending.dispatch.timeoutProfile.interruptGraceMs,
        ),
      );
      this.persist(pending.dispatch, {
        type: "interrupt",
        at: nowIso(),
        completed: true,
        error: null,
      });
    } catch (error) {
      pending.interruptError = errorMessage(error);
      this.persist(pending.dispatch, {
        type: "interrupt",
        at: nowIso(),
        completed: false,
        error: pending.interruptError,
      });
    }

    await delay(pending.dispatch.timeoutProfile.interruptGraceMs);
    const current = this.turns.get(pending.providerTurnId);
    if (!current) return;
    this.finalizeTimeout(current, kind);
  }

  private finalizeTimeout(
    pending: PendingTurn,
    kind: Extract<RuntimeFailureKind, "IDLE_TIMEOUT" | "HARD_TIMEOUT">,
  ): void {
    clearTimeout(pending.idleTimer);
    clearTimeout(pending.hardTimer);
    this.flushAllOutputDeltas(pending);
    this.turns.delete(pending.providerTurnId);
    const completedCandidates = pending.messageOrder
      .filter((id) => pending.completedMessageIds.has(id))
      .map((id) => ({
        text: pending.messageText.get(id) ?? "",
        phase: pending.messagePhase.get(id) ?? null,
      }));
    const selected = selectProtocolAgentMessage(null, completedCandidates);
    const capture: OutputCaptureState = selected.text
      ? "COMPLETE"
      : captureState(pending);
    const elapsedMs = Date.now() - pending.startedAt;
    const timeoutMs = kind === "IDLE_TIMEOUT"
      ? pending.dispatch.timeoutProfile.idleTimeoutMs
      : pending.dispatch.timeoutProfile.hardTimeoutMs;
    const result: RawTurnResult = {
      status: "timeout",
      text: selected.text,
      providerThreadId: pending.threadId,
      providerTurnId: pending.providerTurnId,
      usage: pending.usage,
      toolEvents: pending.toolEvents,
      rawEvents: pending.rawEvents,
      compacted: pending.compacted,
      outputCapture: capture,
      partialText: partialMessageText(pending),
      failureKind: kind,
      interruptError: pending.interruptError,
      lastActivityAt: new Date(pending.lastActivityAt).toISOString(),
      incrementalEventsPersisted: Boolean(
        pending.dispatch.onRuntimeEvent,
      ),
      error:
        `${kind} after ${timeoutMs} ms` +
        (selected.error && pending.completedMessageIds.size > 0
          ? `; ${selected.error}`
          : ""),
      elapsedMs,
    };
    this.emit({
      type: "turn_completed",
      turnId: pending.dispatch.turnId,
      role: pending.dispatch.role,
      status: "timeout",
      elapsedMs,
      usage: pending.usage,
      toolCalls: pending.toolEvents.length,
    });
    pending.resolve(result);
  }

  private persist(
    dispatch: TurnDispatch,
    event: RuntimePersistenceEvent,
  ): void {
    try {
      dispatch.onRuntimeEvent?.(event);
    } catch (error) {
      this.failAll(
        new Error(`runtime persistence failed: ${errorMessage(error)}`),
      );
    }
  }

  private bufferEarlyMessage(
    providerTurnId: string,
    message: Record<string, unknown>,
  ): void {
    const items = this.earlyMessages.get(providerTurnId) ?? [];
    if (items.length < 512) items.push(message);
    this.earlyMessages.set(providerTurnId, items);
  }

  private replayEarlyMessages(providerTurnId: string): void {
    const messages = this.earlyMessages.get(providerTurnId);
    if (!messages) return;
    this.earlyMessages.delete(providerTurnId);
    for (const message of messages) this.onLine(JSON.stringify(message));
  }

  private failAll(error: Error): void {
    this.emit({ type: "runtime_error", message: error.message });
    for (const request of this.requests.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    for (const turn of this.turns.values()) {
      clearTimeout(turn.idleTimer);
      clearTimeout(turn.hardTimer);
      this.flushAllOutputDeltas(turn);
      const completedCandidates = turn.messageOrder
        .filter((id) => turn.completedMessageIds.has(id))
        .map((id) => ({
          text: turn.messageText.get(id) ?? "",
          phase: turn.messagePhase.get(id) ?? null,
        }));
      const selected = selectProtocolAgentMessage(null, completedCandidates);
      const elapsedMs = Date.now() - turn.startedAt;
      turn.resolve({
        status: "failed",
        text: selected.text,
        providerThreadId: turn.threadId,
        providerTurnId: turn.providerTurnId,
        usage: turn.usage,
        toolEvents: turn.toolEvents,
        rawEvents: turn.rawEvents,
        compacted: turn.compacted,
        outputCapture: selected.text ? "COMPLETE" : captureState(turn),
        partialText: partialMessageText(turn),
        failureKind: turn.timeoutKind ?? "PROVIDER_ERROR",
        interruptError: turn.interruptError,
        lastActivityAt: new Date(turn.lastActivityAt).toISOString(),
        incrementalEventsPersisted: Boolean(
          turn.dispatch.onRuntimeEvent,
        ),
        error: error.message,
        elapsedMs,
      });
    }
    this.requests.clear();
    this.turns.clear();
    this.earlyMessages.clear();
  }

  private emit(event: RuntimeLiveEvent): void {
    try {
      this.config.onLiveEvent?.(event);
    } catch {
      // Console reporting is never workflow authority.
    }
  }
}

export class ScriptedTurnRuntime implements TurnRuntime {
  private invocation = 0;
  private readonly handler: (
    dispatch: TurnDispatch,
    invocation: number,
  ) => Promise<string | Partial<RawTurnResult>> | string | Partial<RawTurnResult>;

  constructor(
    handler: (
      dispatch: TurnDispatch,
      invocation: number,
    ) => Promise<string | Partial<RawTurnResult>> | string | Partial<RawTurnResult>,
  ) {
    this.handler = handler;
  }

  async run(dispatch: TurnDispatch): Promise<RawTurnResult> {
    this.invocation += 1;
    const startedAt = Date.now();
    const value = await this.handler(dispatch, this.invocation);
    if (typeof value === "string") {
      return baseScriptedResult(value, this.invocation, Date.now() - startedAt);
    }
    return {
      ...baseScriptedResult(
        value.text ?? "",
        this.invocation,
        Date.now() - startedAt,
      ),
      ...value,
      providerThreadId:
        value.providerThreadId ?? `scripted-thread-${this.invocation}`,
      providerTurnId:
        value.providerTurnId ?? `scripted-turn-${this.invocation}`,
    };
  }
}

function developerInstructions(role: LoopRole): string {
  const output =
    role === "DECISION"
      ? "In the final_answer phase, return only the requested decision line protocol."
      : "In the final_answer phase, return exactly one JSON object. Include the role's required core control field and follow the Result Ref as the recommended semantic contract.";
  const auxiliary =
    role === "DECISION"
      ? "Do not delegate scheduling or use auxiliary agents."
      : "Auxiliary agents may be used inside this Turn, but no auxiliary result is a top-level workflow message.";
  return [
    "You are one fresh Turn controlled by a deterministic external Script.",
    "Use the Skill explicitly named in the user Prompt.",
    "Read only the frozen inputs referenced by the Prompt as workflow authority.",
    "Do not create or use a Goal, provider-history continuation, or persistent agent state.",
    "Do not modify Controller run-state files or schedule another top-level Turn.",
    "Do not execute a new experiment.",
    auxiliary,
    "Only the final_answer phase is the workflow protocol payload; commentary is progress only.",
    output,
    "Exit after the requested result.",
  ].join("\n");
}

function baseScriptedResult(
  text: string,
  invocation: number,
  elapsedMs: number,
): RawTurnResult {
  return {
    status: "completed",
    text,
    providerThreadId: `scripted-thread-${invocation}`,
    providerTurnId: `scripted-turn-${invocation}`,
    usage: zeroUsage(),
    toolEvents: [],
    rawEvents: [],
    compacted: false,
    outputCapture: text.trim() ? "COMPLETE" : "NONE",
    partialText: text,
    failureKind: null,
    interruptError: null,
    lastActivityAt: nowIso(),
    incrementalEventsPersisted: false,
    error: null,
    elapsedMs,
  };
}

function isTurnNotification(method: string): boolean {
  return (
    method.startsWith("item/") ||
    method === "turn/completed" ||
    method === "thread/tokenUsage/updated" ||
    method === "thread/compacted"
  );
}

function isToolItem(itemType: string | null): boolean {
  return [
    "mcpToolCall",
    "dynamicToolCall",
    "commandExecution",
    "fileChange",
    "collabAgentToolCall",
    "webSearch",
    "imageView",
  ].includes(itemType ?? "");
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

function captureState(pending: PendingTurn): OutputCaptureState {
  if (
    [...pending.completedMessageIds].some(
      (id) => pending.messagePhase.get(id) !== "commentary",
    )
  ) return "COMPLETE";
  return pending.messageText.size > 0 ? "PARTIAL" : "NONE";
}

function partialMessageText(pending: PendingTurn): string {
  return pending.messageOrder
    .map((id) => pending.messageText.get(id) ?? "")
    .filter(Boolean)
    .join("\n");
}

function isMeaningfulActivity(
  method: string,
  item: Record<string, unknown> | null,
): boolean {
  return (
    method === "item/agentMessage/delta" ||
    method === "item/started" ||
    method === "item/completed" ||
    method === "item/mcpToolCall/progress" ||
    method === "command/exec/outputDelta" ||
    method === "process/outputDelta" ||
    method === "item/commandExecution/outputDelta" ||
    method === "item/reasoning/summaryTextDelta" ||
    method === "item/reasoning/summaryPartAdded" ||
    method === "item/reasoning/textDelta" ||
    method === "turn/plan/updated" ||
    method === "thread/tokenUsage/updated" ||
    method === "thread/compacted" ||
    getString(item, "type") === "contextCompaction"
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function nowIso(): string {
  return new Date().toISOString();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function selectProtocolAgentMessage(
  turn: Record<string, unknown> | null,
  streamedMessages: readonly AgentMessageCandidate[],
): ProtocolMessageSelection {
  const items = Array.isArray(turn?.items) ? turn.items : [];
  const completed = items
    .map(asObject)
    .filter((item) => getString(item, "type") === "agentMessage")
    .map((item) => ({
      text: (getString(item, "text") ?? "").trim(),
      phase: getMessagePhase(item),
    }))
    .filter((message) => Boolean(message.text));
  const candidates = completed.length > 0
    ? completed
    : streamedMessages
      .map((message) => ({
        text: message.text.trim(),
        phase: message.phase,
      }))
      .filter((message) => Boolean(message.text));

  const finalAnswers = candidates.filter(
    (message) => message.phase === "final_answer",
  );
  if (finalAnswers.length === 1) {
    return { text: finalAnswers[0]!.text, error: null };
  }
  if (finalAnswers.length > 1) {
    return {
      text: "",
      error:
        `Codex Turn emitted ${finalAnswers.length} final_answer agentMessages; protocol payload is ambiguous`,
    };
  }

  const unknown = candidates.filter((message) => message.phase === null);
  if (unknown.length === 1) {
    return { text: unknown[0]!.text, error: null };
  }
  if (unknown.length > 1) {
    return {
      text: "",
      error:
        `Codex Turn emitted ${unknown.length} phase-unknown agentMessages and no final_answer; protocol payload is ambiguous`,
    };
  }
  return {
    text: "",
    error: "Codex Turn completed without a final_answer agentMessage",
  };
}

function getMessagePhase(
  item: Record<string, unknown> | null,
): MessagePhase {
  const phase = getString(item, "phase");
  return phase === "commentary" || phase === "final_answer" ? phase : null;
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

function normalizeToolResult(
  item: Record<string, unknown> | null,
): string | null {
  const result = asObject(item?.result);
  const content = Array.isArray(result?.content) ? result.content : [];
  let text = content
    .map(asObject)
    .map((entry) => getString(entry, "text"))
    .filter((entry): entry is string => Boolean(entry))
    .join("\n");
  if (!text) {
    for (const candidate of [
      item?.aggregatedOutput,
      item?.output,
      item?.text,
      typeof item?.result === "string" ? item.result : null,
    ]) {
      if (typeof candidate === "string" && candidate) {
        text = candidate;
        break;
      }
    }
  }
  if (!text) return null;
  return text.length > 2_000 ? `${text.slice(0, 2_000)}…` : text;
}

function normalizeToolArguments(
  itemType: string | null,
  item: Record<string, unknown> | null,
): unknown {
  if (item?.arguments !== undefined) return item.arguments;
  if (item?.input !== undefined) return item.input;
  if (itemType === "commandExecution") {
    return {
      command: getString(item, "command"),
      cwd: getString(item, "cwd"),
    };
  }
  const compact: Record<string, unknown> = {};
  for (const key of ["path", "query", "url", "changes"]) {
    if (item?.[key] !== undefined) compact[key] = item[key];
  }
  return compact;
}
