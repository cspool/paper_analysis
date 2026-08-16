import { createHash, randomUUID } from "node:crypto";
import {
  appendFileSync,
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type {
  DirectionGoalRecord,
  DirectionRunFile,
  DirectionStateFile,
} from "./types.ts";

export class DirectionExperimentStore {
  readonly workDir: string;
  private lockHeld = false;
  private eventSeq = 0;

  constructor(workDir: string) {
    this.workDir = resolve(workDir);
    this.eventSeq = this.readExistingEventSeq();
  }

  initializeLayout(): void {
    mkdirSync(this.workDir, { recursive: true });
    for (const directory of [
      "inputs",
      "decisions",
      "contracts",
      "judges",
      "judge_requests",
      "lab_goals",
      "control",
      "snapshots",
      "workspace/direction",
      "workspace/knowledge",
      "workspace/carriers",
      "workspace/env",
      "workspace/code",
      "workspace/patches",
      "workspace/configs",
      "workspace/raw",
      "workspace/analysis",
      "workspace/cycles",
      "final",
    ]) mkdirSync(this.absolute(directory), { recursive: true });
    if (!this.exists("workspace/commands.md")) {
      this.writeText("workspace/commands.md", "# Direction Lab command history\n");
    }
  }

  ensureDir(path: string): void {
    mkdirSync(this.absolute(path), { recursive: true });
  }

  acquireLock(): void {
    const path = this.absolute(".run.lock");
    if (existsSync(path)) {
      let live = false;
      try {
        const value = JSON.parse(readFileSync(path, "utf8")) as { pid?: number };
        if (typeof value.pid === "number") {
          try {
            process.kill(value.pid, 0);
            live = true;
          } catch {
            live = false;
          }
        }
      } catch {
        live = false;
      }
      if (live) throw new Error(`run is already locked by a live process: ${path}`);
      unlinkSync(path);
    }
    const descriptor = openSync(path, "wx", 0o600);
    try {
      writeFileSync(descriptor, `${JSON.stringify({
        pid: process.pid,
        acquiredAt: new Date().toISOString(),
      })}\n`, "utf8");
    } finally {
      closeSync(descriptor);
    }
    this.lockHeld = true;
  }

  releaseLock(): void {
    if (!this.lockHeld) return;
    const path = this.absolute(".run.lock");
    if (existsSync(path)) unlinkSync(path);
    this.lockHeld = false;
  }

  absolute(path: string): string {
    const candidate = resolve(isAbsolute(path) ? path : resolve(this.workDir, path));
    if (candidate !== this.workDir && !candidate.startsWith(`${this.workDir}${sep}`)) {
      throw new Error(`path escapes run directory: ${candidate}`);
    }
    return candidate;
  }

  relative(path: string): string {
    return relative(this.workDir, this.absolute(path)).split(sep).join("/");
  }

  exists(path: string): boolean {
    return existsSync(this.absolute(path));
  }

  isNonEmptyFile(path: string): boolean {
    try {
      const stat = statSync(this.absolute(path));
      return stat.isFile() && stat.size > 0;
    } catch {
      return false;
    }
  }

  readJson<T>(path: string): T {
    return JSON.parse(readFileSync(this.absolute(path), "utf8")) as T;
  }

  readText(path: string): string {
    return readFileSync(this.absolute(path), "utf8");
  }

  writeText(path: string, value: string): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    const temporary = `${target}.tmp-${process.pid}-${randomUUID()}`;
    writeFileSync(temporary, value, "utf8");
    renameSync(temporary, target);
  }

  writeJson(path: string, value: unknown): void {
    this.writeText(path, `${JSON.stringify(value, null, 2)}\n`);
  }

  appendJsonLine(path: string, value: unknown): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    appendFileSync(target, `${JSON.stringify(value)}\n`, "utf8");
  }

  appendEvent(type: string, refs: string[] = [], detail?: unknown): void {
    this.eventSeq += 1;
    this.appendJsonLine("events.jsonl", {
      seq: this.eventSeq,
      time: new Date().toISOString(),
      type,
      refs,
      ...(detail === undefined ? {} : { detail }),
    });
  }

  runtimeEvent(ref: string, event: unknown): void {
    this.appendJsonLine(ref, event);
  }

  readRun(): DirectionRunFile {
    return this.readJson<DirectionRunFile>("run.json");
  }

  readState(): DirectionStateFile {
    return this.readJson<DirectionStateFile>("state.json");
  }

  writeState(next: DirectionStateFile, event = "STATE_UPDATED"): void {
    const current = this.exists("state.json") ? this.readState() : null;
    if (current && next.revision !== current.revision + 1) {
      throw new Error(
        `state revision must advance by one: current=${current.revision}, next=${next.revision}`,
      );
    }
    this.writeJson("state.json", next);
    this.appendEvent(event, ["state.json"]);
  }

  readGoal(ref: string): DirectionGoalRecord {
    return this.readJson<DirectionGoalRecord>(ref);
  }

  sha256(path: string): string {
    return createHash("sha256").update(readFileSync(this.absolute(path))).digest("hex");
  }

  sha256External(path: string): string {
    return createHash("sha256").update(readFileSync(resolve(path))).digest("hex");
  }

  newId(prefix: string): string {
    return `${prefix}-${randomUUID()}`;
  }

  writePauseRequest(reason: string, source: "CLI" | "SIGINT" | "SIGTERM"): void {
    this.writeJson("control/pause-request.json", {
      requestId: this.newId("pause-request"),
      requestedAt: new Date().toISOString(),
      reason,
      source,
    });
  }

  readPauseRequest(): {
    requestId: string;
    requestedAt: string;
    reason: string;
    source: "CLI" | "SIGINT" | "SIGTERM";
  } | null {
    if (!this.exists("control/pause-request.json")) return null;
    try {
      const value = this.readJson<Record<string, unknown>>("control/pause-request.json");
      if (
        typeof value.requestId !== "string" ||
        typeof value.requestedAt !== "string" ||
        typeof value.reason !== "string" ||
        !["CLI", "SIGINT", "SIGTERM"].includes(String(value.source))
      ) return null;
      return value as ReturnType<DirectionExperimentStore["readPauseRequest"]>;
    } catch {
      return null;
    }
  }

  clearPauseRequest(): void {
    const path = this.absolute("control/pause-request.json");
    if (existsSync(path)) unlinkSync(path);
  }

  private readExistingEventSeq(): number {
    const path = resolve(this.workDir, "events.jsonl");
    if (!existsSync(path)) return 0;
    const lines = readFileSync(path, "utf8").trim().split("\n").filter(Boolean);
    if (lines.length === 0) return 0;
    try {
      const value = JSON.parse(lines.at(-1)!) as { seq?: number };
      return Number.isInteger(value.seq) ? Number(value.seq) : lines.length;
    } catch {
      return lines.length;
    }
  }
}
