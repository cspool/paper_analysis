import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

import { appendJsonLine, atomicWriteJson } from "./canonical_store.ts";
import { OBSIDIAN_READONLY_TOOLS } from "./role_profiles.ts";
import { ZERO_USAGE } from "./types.ts";
import type {
  JsonValue,
  NormalizedEvent,
  Role,
  RunConfig,
  TokenUsage,
  TurnResult,
} from "./types.ts";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

interface PendingTurn {
  threadId: string;
  turnId: string;
  resolve: (value: TurnResult) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
  events: NormalizedEvent[];
  rawMessages: JsonValue[];
  usage: TokenUsage | null;
  compacted: boolean;
  repairMode: boolean;
  role: Role;
  agentMessageOrder: string[];
  agentMessageText: Map<string, string>;
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function getString(object: Record<string, unknown> | null, key: string): string | null {
  const value = object?.[key];
  return typeof value === "string" ? value : null;
}

function normalizeUsage(value: unknown): TokenUsage | null {
  const root = asObject(value);
  const last = asObject(root?.last);
  if (!last) return null;
  return {
    inputTokens: Number(last.inputTokens ?? 0),
    cachedInputTokens: Number(last.cachedInputTokens ?? 0),
    outputTokens: Number(last.outputTokens ?? 0),
    reasoningOutputTokens: Number(last.reasoningOutputTokens ?? 0),
    totalTokens: Number(last.totalTokens ?? 0),
  };
}

function normalizeEvent(message: Record<string, unknown>): NormalizedEvent {
  const method = String(message.method ?? "");
  const params = asObject(message.params);
  const item = asObject(params?.item);
  return {
    method,
    threadId: getString(params, "threadId"),
    turnId: getString(params, "turnId") ?? getString(asObject(params?.turn), "id"),
    itemType: getString(item, "type"),
    server: getString(item, "server") ?? getString(item, "namespace"),
    tool: getString(item, "tool"),
    raw: message as unknown as JsonValue,
  };
}

function itemAdmissionViolation(event: NormalizedEvent, role: Role, repairMode: boolean): string | null {
  if (event.method !== "item/completed" && event.method !== "item/started") {
    return null;
  }
  if (!event.itemType) return null;
  const alwaysAllowed = new Set(["userMessage", "agentMessage", "reasoning", "contextCompaction"]);
  if (alwaysAllowed.has(event.itemType)) return null;
  if (repairMode) {
    return `protocol repair used forbidden item type ${event.itemType}`;
  }
  const evidenceRole = role === "anchor_evidence_worker" || role === "review_evidence_worker";
  if (evidenceRole && (event.itemType === "mcpToolCall" || event.itemType === "dynamicToolCall")) {
    const server = (event.server ?? "").toLowerCase();
    const tool = event.tool ?? "";
    if (server.includes("obsidian") && OBSIDIAN_READONLY_TOOLS.has(tool)) {
      return null;
    }
    return `evidence role used non-allowlisted MCP tool ${event.server ?? "<unknown>"}/${tool || "<unknown>"}`;
  }
  return `${role} used forbidden item type ${event.itemType}`;
}

function extractTurnText(turn: Record<string, unknown>): string {
  const items = Array.isArray(turn.items) ? turn.items : [];
  const messages: string[] = [];
  for (const rawItem of items) {
    const item = asObject(rawItem);
    if (item?.type === "agentMessage" && typeof item.text === "string") {
      messages.push(item.text);
    }
  }
  return messages.join("\n").trim();
}

function rememberAgentMessage(
  pending: PendingTurn,
  itemId: string,
  text: string,
  replace: boolean,
): void {
  if (!pending.agentMessageText.has(itemId)) {
    pending.agentMessageOrder.push(itemId);
  }
  const previous = pending.agentMessageText.get(itemId) ?? "";
  pending.agentMessageText.set(itemId, replace ? text : `${previous}${text}`);
}

function extractPendingTurnText(
  turn: Record<string, unknown>,
  pending: PendingTurn,
): string {
  const completedText = extractTurnText(turn);
  if (completedText) return completedText;
  return pending.agentMessageOrder
    .map((itemId) => pending.agentMessageText.get(itemId) ?? "")
    .filter(Boolean)
    .join("\n")
    .trim();
}

function ensureSymlink(linkPath: string, targetPath: string): void {
  if (!fs.existsSync(targetPath)) {
    throw new Error(`required Codex source file does not exist: ${targetPath}`);
  }
  try {
    const existing = fs.lstatSync(linkPath);
    if (!existing.isSymbolicLink()) {
      throw new Error(`${linkPath} exists and is not a symlink`);
    }
    const currentTarget = fs.readlinkSync(linkPath);
    if (path.resolve(path.dirname(linkPath), currentTarget) !== path.resolve(targetPath)) {
      throw new Error(`${linkPath} points to an unexpected target`);
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      throw error;
    }
    fs.symlinkSync(targetPath, linkPath);
  }
}

export class AppServerClient {
  private readonly config: RunConfig;
  private readonly workDir: string;
  private process: ChildProcessWithoutNullStreams | null = null;
  private lineReader: readline.Interface | null = null;
  private nextRequestId = 1;
  private pendingRequests = new Map<number, PendingRequest>();
  private pendingTurns = new Map<string, PendingTurn>();
  private stderrPath: string;
  private globalRawPath: string;
  private launchCommand: string;
  private launchArgsPrefix: string[];
  private started = false;
  private shuttingDown = false;

  constructor(
    config: RunConfig,
    workDir: string,
    launch: { command?: string; argsPrefix?: string[] } = {},
  ) {
    this.config = config;
    this.workDir = path.resolve(workDir);
    this.stderrPath = path.join(this.workDir, "provider/app_server.stderr.log");
    this.globalRawPath = path.join(this.workDir, "provider/app_server.jsonl");
    this.launchCommand = launch.command ?? "codex";
    this.launchArgsPrefix = launch.argsPrefix ?? [];
  }

  private prepareCodexHome(): { codexHome: string; sqliteHome: string } {
    const codexHome = path.join(this.workDir, "provider/codex_home");
    const sqliteHome = path.join(this.workDir, "provider/sqlite");
    fs.mkdirSync(codexHome, { recursive: true, mode: 0o700 });
    fs.mkdirSync(sqliteHome, { recursive: true, mode: 0o700 });
    fs.chmodSync(codexHome, 0o700);
    fs.chmodSync(sqliteHome, 0o700);
    ensureSymlink(path.join(codexHome, "auth.json"), this.config.sourceAuthPath);
    ensureSymlink(path.join(codexHome, "config.toml"), this.config.sourceConfigPath);
    return { codexHome, sqliteHome };
  }

  async start(): Promise<void> {
    if (this.started) return;
    const { codexHome, sqliteHome } = this.prepareCodexHome();
    const args = [
      "-c",
      `sqlite_home="${sqliteHome}"`,
      "-c",
      "features.apps=false",
      "app-server",
      "--listen",
      "stdio://",
    ];
    this.process = spawn(this.launchCommand, [...this.launchArgsPrefix, ...args], {
      cwd: this.config.projectRoot,
      env: {
        ...process.env,
        CODEX_HOME: codexHome,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.process.stderr.on("data", (chunk: Buffer) => {
      fs.appendFileSync(this.stderrPath, chunk);
    });
    this.process.on("error", (error) => this.failAll(error));
    this.process.on("exit", (code, signal) => {
      if (!this.shuttingDown) {
        this.failAll(new Error(`codex app-server exited unexpectedly (code=${code}, signal=${signal})`));
      }
    });
    this.lineReader = readline.createInterface({ input: this.process.stdout });
    this.lineReader.on("line", (line) => this.onLine(line));

    const initialize = await this.request("initialize", {
      clientInfo: {
        name: "codex_learning_workflow",
        title: "Codex Learning Workflow",
        version: "1.0.0",
      },
      capabilities: {
        experimentalApi: false,
      },
    }, this.config.startupTimeoutMs);
    const initialized = asObject(initialize);
    if (!initialized?.codexHome) {
      throw new Error("app-server initialize response is missing codexHome");
    }
    this.notify("initialized", {});
    this.started = true;
  }

  async stop(): Promise<void> {
    if (!this.process) return;
    this.shuttingDown = true;
    for (const turn of this.pendingTurns.values()) {
      try {
        await this.request("turn/interrupt", {
          threadId: turn.threadId,
          turnId: turn.turnId,
        }, 5_000);
      } catch {
        // The process will be terminated below.
      }
    }
    this.lineReader?.close();
    this.process.kill("SIGTERM");
    this.process = null;
    this.started = false;
  }

  async request(method: string, params: JsonValue, timeoutMs = this.config.requestTimeoutMs): Promise<unknown> {
    if (!this.process) {
      throw new Error("app-server is not running");
    }
    const id = this.nextRequestId;
    this.nextRequestId += 1;
    const response = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`${method} request timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      this.pendingRequests.set(id, { resolve, reject, timer });
    });
    this.send({ id, method, params });
    return response;
  }

  notify(method: string, params: JsonValue): void {
    this.send({ method, params });
  }

  async startThread(
    role: Role,
    cwd: string,
    persistent: boolean,
  ): Promise<{ threadId: string; response: JsonValue }> {
    const evidenceRole = role === "anchor_evidence_worker" || role === "review_evidence_worker";
    const response = await this.request("thread/start", {
      model: this.config.model,
      allowProviderModelFallback: false,
      ephemeral: !persistent,
      approvalPolicy: "never",
      sandbox: "read-only",
      cwd,
      config: {
        features: {
          apps: false,
        },
        mcp_servers: {
          "code-review-graph": {
            enabled: false,
          },
          obsidian: {
            enabled: evidenceRole,
          },
        },
      },
      developerInstructions: [
        "You are one workflow role controlled by an external deterministic script.",
        "Never create, invoke, delegate to, or manage another agent.",
        "Never use collaboration mode or subagents.",
        evidenceRole
          ? "Only Obsidian read-only MCP search/get/list tools are permitted. Do not use shell, filesystem, web, or write tools."
          : "Do not use any tool. Reason only over the exact input packet.",
        "Return only the role protocol requested by the attached skill.",
      ].join("\n"),
    } as unknown as JsonValue);
    const object = asObject(response);
    const thread = asObject(object?.thread);
    const threadId = getString(thread, "id");
    if (!threadId) {
      throw new Error(`thread/start for ${role} returned no thread id`);
    }
    return { threadId, response: response as JsonValue };
  }

  async resumeThread(role: Role, threadId: string, cwd: string): Promise<void> {
    const evidenceRole = role === "anchor_evidence_worker" || role === "review_evidence_worker";
    await this.request("thread/resume", {
      threadId,
      model: this.config.model,
      approvalPolicy: "never",
      sandbox: "read-only",
      cwd,
      config: {
        features: { apps: false },
        mcp_servers: {
          "code-review-graph": { enabled: false },
          obsidian: { enabled: evidenceRole },
        },
      },
      excludeTurns: true,
    } as unknown as JsonValue);
  }

  async runTurn(
    role: Role,
    threadId: string,
    effort: string,
    input: JsonValue[],
    repairMode = false,
  ): Promise<TurnResult> {
    const response = await this.request("turn/start", {
      threadId,
      model: this.config.model,
      effort,
      input,
    } as unknown as JsonValue, this.config.requestTimeoutMs);
    const turn = asObject(asObject(response)?.turn);
    const turnId = getString(turn, "id");
    if (!turnId) {
      throw new Error("turn/start response is missing turn id");
    }

    return new Promise<TurnResult>((resolve, reject) => {
      const timer = setTimeout(async () => {
        this.pendingTurns.delete(turnId);
        try {
          await this.request("turn/interrupt", { threadId, turnId }, 10_000);
        } catch {
          // Preserve the original timeout as the task error.
        }
        reject(new Error(`turn ${turnId} timed out after ${this.config.turnTimeoutMs} ms`));
      }, this.config.turnTimeoutMs);
      this.pendingTurns.set(turnId, {
        threadId,
        turnId,
        resolve,
        reject,
        timer,
        events: [],
        rawMessages: [],
        usage: null,
        compacted: false,
        repairMode,
        role,
        agentMessageOrder: [],
        agentMessageText: new Map<string, string>(),
      });
    });
  }

  private send(message: Record<string, unknown>): void {
    if (!this.process?.stdin.writable) {
      throw new Error("app-server stdin is not writable");
    }
    this.process.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private onLine(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      this.failAll(new Error(`app-server emitted malformed JSONL: ${line.slice(0, 400)}`));
      return;
    }
    appendJsonLine(this.globalRawPath, message as unknown as JsonValue);
    if (typeof message.id === "number" && !message.method) {
      const pending = this.pendingRequests.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pendingRequests.delete(message.id);
      if (message.error) {
        pending.reject(new Error(JSON.stringify(message.error)));
      } else {
        pending.resolve(message.result);
      }
      return;
    }

    if (typeof message.id === "number" && typeof message.method === "string") {
      this.send({
        id: message.id,
        error: {
          code: -32001,
          message: "interactive server requests are disabled for this workflow",
        },
      });
      return;
    }

    if (typeof message.method !== "string") return;
    const event = normalizeEvent(message);
    const turnId = event.turnId;
    const pending = turnId ? this.pendingTurns.get(turnId) : null;
    if (pending) {
      pending.events.push(event);
      pending.rawMessages.push(message as unknown as JsonValue);
      const params = asObject(message.params);
      if (event.method === "item/agentMessage/delta") {
        const itemId = getString(params, "itemId");
        const delta = getString(params, "delta");
        if (itemId && delta !== null) {
          rememberAgentMessage(pending, itemId, delta, false);
        }
      }
      if (event.method === "item/completed") {
        const item = asObject(params?.item);
        const itemId = getString(item, "id");
        const text = getString(item, "text");
        if (item?.type === "agentMessage" && itemId && text !== null) {
          // App Server may stream deltas and then repeat the complete text here.
          // The completed item is authoritative, so replace rather than append.
          rememberAgentMessage(pending, itemId, text, true);
        }
      }
      if (event.method === "thread/tokenUsage/updated") {
        pending.usage = normalizeUsage(params?.tokenUsage);
      }
      if (event.method === "thread/compacted" || event.itemType === "contextCompaction") {
        pending.compacted = true;
      }
    }
    if (event.method === "turn/completed") {
      this.completeTurn(message);
    }
  }

  private completeTurn(message: Record<string, unknown>): void {
    const params = asObject(message.params);
    const turn = asObject(params?.turn);
    const turnId = getString(turn, "id");
    if (!turnId) return;
    const pending = this.pendingTurns.get(turnId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pendingTurns.delete(turnId);
    const statusRaw = getString(turn, "status") ?? "failed";
    const status = statusRaw === "completed"
      ? "completed"
      : statusRaw === "interrupted"
        ? "interrupted"
        : "failed";
    const errorObject = asObject(turn?.error);
    const error = getString(errorObject, "message");
    const securityViolations = pending.events
      .map((event) => itemAdmissionViolation(event, pending.role, pending.repairMode))
      .filter((value): value is string => Boolean(value));
    const rawLogPath = path.join(this.workDir, "provider/raw_turns", `${turnId}.jsonl`);
    fs.mkdirSync(path.dirname(rawLogPath), { recursive: true, mode: 0o700 });
    fs.writeFileSync(
      rawLogPath,
      pending.rawMessages.length
        ? `${pending.rawMessages.map((raw) => JSON.stringify(raw)).join("\n")}\n`
        : "",
      "utf8",
    );
    const result: TurnResult = {
      threadId: pending.threadId,
      turnId,
      status,
      text: extractPendingTurnText(turn ?? {}, pending),
      usage: pending.usage ?? { ...ZERO_USAGE },
      observedEvents: pending.events,
      rawLogPath,
      securityViolations,
      compacted: pending.compacted,
      error,
    };
    pending.resolve(result);
  }

  private failAll(error: Error): void {
    for (const pending of this.pendingRequests.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pendingRequests.clear();
    for (const pending of this.pendingTurns.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pendingTurns.clear();
  }
}
