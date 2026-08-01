import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { AppServerClient } from "./app_server_client.ts";
import { atomicWriteJson } from "./canonical_store.ts";
import {
  findFirstJsonValue,
  parseProtocol,
  semanticPayloadUnchanged,
} from "./protocol_parser.ts";
import { validateProtocolTransition } from "./protocol_state_machine.ts";
import { roleProfile } from "./role_profiles.ts";
import { ZERO_USAGE } from "./types.ts";
import type {
  AgentHandle,
  Effort,
  JsonValue,
  ParsedProtocol,
  Role,
  RunState,
  SessionRecord,
  TokenUsage,
  TurnResult,
} from "./types.ts";

function hashFile(filePath: string): string {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function addUsage(target: TokenUsage, increment: TokenUsage | null): void {
  if (!increment) return;
  target.inputTokens += increment.inputTokens;
  target.cachedInputTokens += increment.cachedInputTokens;
  target.outputTokens += increment.outputTokens;
  target.reasoningOutputTokens += increment.reasoningOutputTokens;
  target.totalTokens += increment.totalTokens;
}

function sessionKey(role: Role, scopeId: string): string {
  return `${role}:${scopeId}`;
}

function semanticComparable(role: Role, parsed: ParsedProtocol): JsonValue | null {
  if (role === "review_evidence_worker" && parsed.payload && !Array.isArray(parsed.payload)) {
    return (parsed.payload as { [key: string]: JsonValue }).sources ?? null;
  }
  return parsed.payload;
}

export class SecurityViolationError extends Error {
  readonly violations: string[];

  constructor(violations: string[]) {
    super(`agent tool admission violation: ${violations.join("; ")}`);
    this.name = "SecurityViolationError";
    this.violations = violations;
  }
}

export class RuntimeManager {
  readonly client: AppServerClient;
  private readonly state: RunState;
  private readonly workDir: string;

  constructor(state: RunState, workDir: string) {
    this.state = state;
    this.workDir = path.resolve(workDir);
    this.client = new AppServerClient(state.config, this.workDir);
  }

  async start(): Promise<void> {
    await this.client.start();
  }

  async stop(): Promise<void> {
    await this.client.stop();
  }

  private roleCwd(role: Role, scopeId: string): string {
    const profile = roleProfile(role, this.state.config.skillRoot);
    if (profile.knowledgeAccess === "obsidian_readonly") {
      return this.state.config.projectRoot;
    }
    const safeScope = scopeId.replace(/[^A-Za-z0-9_.-]/g, "_");
    const directory = path.join(this.workDir, "provider/role_sandboxes", role, safeScope);
    fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
    return directory;
  }

  private createSession(handle: AgentHandle): SessionRecord {
    const profile = roleProfile(handle.role, this.state.config.skillRoot);
    return {
      role: handle.role,
      scopeId: handle.scopeId,
      threadId: handle.threadId,
      lastTurnId: null,
      model: this.state.config.model,
      effort: profile.effort,
      skillPath: handle.skillPath,
      skillHash: handle.skillHash,
      lastLoop: null,
      lastNormalizedOutputPath: null,
      status: "waiting_input",
      turnCount: 0,
      cumulativeUsage: { ...ZERO_USAGE },
    };
  }

  async startAgent(role: Role, scopeId: string, persistent: boolean): Promise<AgentHandle> {
    const profile = roleProfile(role, this.state.config.skillRoot);
    if (profile.persistent !== persistent) {
      throw new Error(`role ${role} persistent=${profile.persistent}, requested ${persistent}`);
    }
    if (!fs.existsSync(profile.skillPath)) {
      throw new Error(`skill file is missing: ${profile.skillPath}`);
    }
    const skillHash = hashFile(profile.skillPath);
    const started = await this.client.startThread(role, this.roleCwd(role, scopeId), persistent);
    const handle: AgentHandle = {
      role,
      scopeId,
      threadId: started.threadId,
      skillPath: profile.skillPath,
      skillHash,
      persistent,
      firstTurn: true,
      turnCount: 0,
    };
    if (persistent) {
      this.state.sessions[sessionKey(role, scopeId)] = this.createSession(handle);
    }
    return handle;
  }

  async persistentAgent(role: Role, scopeId: string): Promise<AgentHandle> {
    const profile = roleProfile(role, this.state.config.skillRoot);
    if (!profile.persistent) {
      throw new Error(`${role} is not a persistent role`);
    }
    const key = sessionKey(role, scopeId);
    const session = this.state.sessions[key];
    const currentHash = hashFile(profile.skillPath);
    if (
      session
      && session.status !== "contaminated"
      && session.skillHash === currentHash
      && session.model === this.state.config.model
      && session.effort === profile.effort
      && session.turnCount < this.state.config.maxPersistentTurns
    ) {
      await this.client.resumeThread(role, session.threadId, this.roleCwd(role, scopeId));
      return {
        role,
        scopeId,
        threadId: session.threadId,
        skillPath: profile.skillPath,
        skillHash: currentHash,
        persistent: true,
        firstTurn: false,
        turnCount: session.turnCount,
      };
    }
    return this.startAgent(role, scopeId, true);
  }

  private buildInput(handle: AgentHandle, text: string): JsonValue[] {
    const input: JsonValue[] = [{ type: "text", text }];
    if (handle.firstTurn) {
      input.push({
        type: "skill",
        name: roleProfile(handle.role, this.state.config.skillRoot).skillName,
        path: handle.skillPath,
      });
    }
    return input;
  }

  private recordTurn(handle: AgentHandle, result: TurnResult, parsed: ParsedProtocol | null): void {
    this.state.usage.turns += 1;
    addUsage(this.state.usage.total, result.usage);
    if (!this.state.usage.byRole[handle.role]) {
      this.state.usage.byRole[handle.role] = { ...ZERO_USAGE };
    }
    addUsage(this.state.usage.byRole[handle.role], result.usage);
    handle.firstTurn = false;
    handle.turnCount += 1;
    if (!handle.persistent) return;
    const session = this.state.sessions[sessionKey(handle.role, handle.scopeId)];
    if (!session) return;
    session.lastTurnId = result.turnId;
    session.turnCount += 1;
    session.lastLoop = parsed?.loop ?? null;
    session.status = parsed?.loop?.includes("§TERMINATED") ? "terminated" : "yielded";
    addUsage(session.cumulativeUsage, result.usage);
    if (result.compacted) {
      session.status = "contaminated";
    }
  }

  async runRawTurn(
    handle: AgentHandle,
    text: string,
    effort?: Effort,
    repairMode = false,
  ): Promise<TurnResult> {
    if (this.state.usage.turns >= this.state.config.maxTotalTurns) {
      throw new Error("run turn budget exhausted");
    }
    const profile = roleProfile(handle.role, this.state.config.skillRoot);
    const result = await this.client.runTurn(
      handle.role,
      handle.threadId,
      effort ?? profile.effort,
      this.buildInput(handle, text),
      repairMode,
    );
    if (result.status !== "completed") {
      this.recordTurn(handle, result, null);
      throw new Error(`turn ${result.turnId} ended with ${result.status}: ${result.error ?? "unknown error"}`);
    }
    if (result.securityViolations.length > 0) {
      this.recordTurn(handle, result, null);
      if (handle.persistent) {
        const session = this.state.sessions[sessionKey(handle.role, handle.scopeId)];
        if (session) session.status = "contaminated";
      }
      throw new SecurityViolationError(result.securityViolations);
    }
    return result;
  }

  async runProtocolTurn(
    handle: AgentHandle,
    text: string,
  ): Promise<{ handle: AgentHandle; result: TurnResult; protocol: ParsedProtocol; repaired: boolean }> {
    let activeHandle = handle;
    let result: TurnResult;
    try {
      result = await this.runRawTurn(activeHandle, text);
    } catch (error) {
      if (!(error instanceof SecurityViolationError)) throw error;
      activeHandle = await this.startAgent(handle.role, handle.scopeId, handle.persistent);
      result = await this.runRawTurn(activeHandle, text);
    }

    try {
      const parsed = parseProtocol(activeHandle.role, result.text, this.state.config.skillRoot);
      validateProtocolTransition(activeHandle.role, parsed);
      this.recordTurn(activeHandle, result, parsed);
      this.writeNormalized(activeHandle, result, parsed, false);
      return { handle: activeHandle, result, protocol: parsed, repaired: false };
    } catch (initialError) {
      const originalPayload = findFirstJsonValue(result.text);
      if (originalPayload === null) {
        this.recordTurn(activeHandle, result, null);
        throw new Error(`protocol invalid and no intact JSON payload is repairable: ${initialError instanceof Error ? initialError.message : String(initialError)}`);
      }
      this.recordTurn(activeHandle, result, null);
      const repairPrompt = [
        "Repair only the control envelope of the previous output.",
        "Do not reason again, add facts, remove facts, reorder arrays, or change the JSON payload.",
        "Do not use any tool.",
        `Role: ${activeHandle.role}`,
        `Parser error: ${initialError instanceof Error ? initialError.message : String(initialError)}`,
        "Re-emit the exact semantic JSON unchanged inside the Marker/LOOP or TASK_TERMINATED required by the active skill.",
        "Previous output follows:",
        result.text,
      ].join("\n\n");
      const repairedResult = await this.runRawTurn(activeHandle, repairPrompt, "low", true);
      const repaired = parseProtocol(activeHandle.role, repairedResult.text, this.state.config.skillRoot);
      validateProtocolTransition(activeHandle.role, repaired);
      if (!semanticPayloadUnchanged(originalPayload, semanticComparable(activeHandle.role, repaired))) {
        this.recordTurn(activeHandle, repairedResult, null);
        throw new Error("protocol repair changed the semantic payload");
      }
      this.recordTurn(activeHandle, repairedResult, repaired);
      this.writeNormalized(activeHandle, repairedResult, repaired, true);
      return { handle: activeHandle, result: repairedResult, protocol: repaired, repaired: true };
    }
  }

  private writeNormalized(
    handle: AgentHandle,
    result: TurnResult,
    protocol: ParsedProtocol,
    repaired: boolean,
  ): void {
    const outputPath = path.join(this.workDir, "provider/raw_turns", `${result.turnId}.normalized.json`);
    atomicWriteJson(outputPath, {
      role: handle.role,
      scopeId: handle.scopeId,
      threadId: handle.threadId,
      turnId: result.turnId,
      repaired,
      protocol,
    } as unknown as JsonValue);
    if (handle.persistent) {
      const session = this.state.sessions[sessionKey(handle.role, handle.scopeId)];
      if (session) session.lastNormalizedOutputPath = outputPath;
    }
  }
}

