import { createHash, randomUUID } from "node:crypto";
import {
  appendFileSync,
  closeSync,
  existsSync,
  mkdirSync,
  linkSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type {
  ObjectsIndex,
  RoundFile,
  RunFile,
  StateFile,
  TurnFile,
} from "./types.ts";

export class FileLoopStore {
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
      "tasks",
      "bindings",
      "turns",
      "contexts",
      "results",
      "audits",
      "rounds",
      "objects",
      "observations",
      "observations/checkpoints",
      "recoveries",
      "authorizations/rounds",
      "final",
    ]) {
      mkdirSync(resolve(this.workDir, directory), { recursive: true });
    }
  }

  acquireLock(): void {
    const lockPath = resolve(this.workDir, ".run.lock");
    if (existsSync(lockPath)) {
      let stale = true;
      try {
        const lock = JSON.parse(readFileSync(lockPath, "utf8")) as {
          pid?: number;
        };
        if (typeof lock.pid === "number") {
          try {
            process.kill(lock.pid, 0);
            stale = false;
          } catch {
            stale = true;
          }
        }
      } catch {
        stale = true;
      }
      if (!stale) {
        throw new Error(`run is already locked by a live process: ${lockPath}`);
      }
      unlinkSync(lockPath);
    }
    const descriptor = openSync(lockPath, "wx", 0o600);
    try {
      writeFileSync(
        descriptor,
        `${JSON.stringify({
          pid: process.pid,
          acquiredAt: new Date().toISOString(),
        })}\n`,
        "utf8",
      );
    } finally {
      closeSync(descriptor);
    }
    this.lockHeld = true;
  }

  releaseLock(): void {
    if (!this.lockHeld) return;
    const lockPath = resolve(this.workDir, ".run.lock");
    try {
      if (existsSync(lockPath)) unlinkSync(lockPath);
    } finally {
      this.lockHeld = false;
    }
  }

  exists(path: string): boolean {
    return existsSync(this.absolute(path));
  }

  absolute(path: string): string {
    if (isAbsolute(path)) {
      const candidate = resolve(path);
      this.assertContained(candidate);
      return candidate;
    }
    const candidate = resolve(this.workDir, path);
    this.assertContained(candidate);
    return candidate;
  }

  relative(path: string): string {
    const candidate = resolve(path);
    this.assertContained(candidate);
    return relative(this.workDir, candidate).split(sep).join("/");
  }

  readJson<T>(path: string): T {
    return JSON.parse(readFileSync(this.absolute(path), "utf8")) as T;
  }

  readText(path: string): string {
    return readFileSync(this.absolute(path), "utf8");
  }

  writeJson(path: string, value: unknown): void {
    this.writeText(path, `${JSON.stringify(value, null, 2)}\n`);
  }

  writeImmutableJson(path: string, value: unknown): void {
    this.writeImmutableText(path, `${JSON.stringify(value, null, 2)}\n`);
  }

  writeText(path: string, value: string): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    const relativePath = this.relative(target);
    if (existsSync(target) && isImmutableDecisionContextPath(relativePath)) {
      throw new Error(
        `immutable DecisionContext file cannot be overwritten: ${relativePath}`,
      );
    }
    const temporary = `${target}.tmp-${process.pid}-${randomUUID()}`;
    writeFileSync(temporary, value, "utf8");
    renameSync(temporary, target);
  }

  writeImmutableText(path: string, value: string): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    if (existsSync(target)) {
      if (readFileSync(target, "utf8") === value) return;
      throw new Error(`immutable file already exists with different content: ${path}`);
    }

    const temporary = `${target}.tmp-${process.pid}-${randomUUID()}`;
    writeFileSync(temporary, value, "utf8");
    try {
      try {
        // A hard link publishes the fully written temporary file without an
        // overwrite window. The run lock prevents ordinary competing writers;
        // EEXIST is still handled defensively for crash/retry races.
        linkSync(temporary, target);
      } catch (error) {
        if (
          (error as NodeJS.ErrnoException).code === "EEXIST" &&
          readFileSync(target, "utf8") === value
        ) {
          return;
        }
        if ((error as NodeJS.ErrnoException).code === "EEXIST") {
          throw new Error(
            `immutable file already exists with different content: ${path}`,
          );
        }
        throw error;
      }
    } finally {
      if (existsSync(temporary)) unlinkSync(temporary);
    }
  }

  move(from: string, to: string): void {
    const source = this.absolute(from);
    const target = this.absolute(to);
    mkdirSync(dirname(target), { recursive: true });
    renameSync(source, target);
  }

  appendJsonLine(path: string, value: unknown): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    appendFileSync(target, `${JSON.stringify(value)}\n`, "utf8");
  }

  appendText(path: string, value: string): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    appendFileSync(target, value, "utf8");
  }

  readJsonLines<T>(path: string): T[] {
    if (!this.exists(path)) return [];
    return this.readText(path)
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as T);
  }

  appendEvent(type: string, refs: string[] = []): void {
    this.eventSeq += 1;
    this.appendJsonLine("events.jsonl", {
      seq: this.eventSeq,
      time: new Date().toISOString(),
      type,
      refs,
    });
  }

  runtimeEvent(turnId: string, event: unknown): void {
    this.appendJsonLine(`turns/${turnId}/runtime.jsonl`, event);
  }

  readRun(): RunFile {
    return this.readJson<RunFile>("run.json");
  }

  readState(): StateFile {
    return this.readJson<StateFile>("state.json");
  }

  writeState(next: StateFile, eventType = "STATE_UPDATED"): void {
    const current = this.exists("state.json") ? this.readState() : null;
    if (current && next.revision !== current.revision + 1) {
      throw new Error(
        `state revision must advance by one: current=${current.revision}, next=${next.revision}`,
      );
    }
    this.writeJson("state.json", next);
    this.appendEvent(eventType, ["state.json"]);
  }

  readObjects(): ObjectsIndex {
    return this.readJson<ObjectsIndex>("objects/index.json");
  }

  writeObjects(index: ObjectsIndex): void {
    this.writeJson("objects/index.json", index);
    this.appendEvent("OBJECT_INDEX_UPDATED", ["objects/index.json"]);
  }

  readTurn(turnRef: string): TurnFile {
    return this.readJson<TurnFile>(turnRef);
  }

  turnRefs(): string[] {
    const root = this.absolute("turns");
    if (!existsSync(root)) return [];
    return readdirSync(root, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => `turns/${entry.name}/turn.json`)
      .filter((ref) => this.exists(ref));
  }

  writeTurn(turn: TurnFile): string {
    const ref = `turns/${turn.turnId}/turn.json`;
    this.writeJson(ref, turn);
    return ref;
  }

  mutateTurn(
    turnRef: string,
    mutate: (turn: TurnFile) => TurnFile,
  ): TurnFile {
    const next = mutate(this.readTurn(turnRef));
    this.writeJson(turnRef, next);
    return next;
  }

  readRound(round: number): RoundFile {
    return this.readJson<RoundFile>(`rounds/${round}.json`);
  }

  writeRound(value: RoundFile): void {
    this.writeJson(`rounds/${value.round}.json`, value);
  }

  appendTurnToRound(round: number, turnRef: string): void {
    const value = this.readRound(round);
    if (!value.turnRefs.includes(turnRef)) value.turnRefs.push(turnRef);
    this.writeRound(value);
  }

  sha256File(path: string): string {
    return createHash("sha256")
      .update(readFileSync(this.absolute(path)))
      .digest("hex");
  }

  sha256ExternalFile(path: string): string {
    return createHash("sha256")
      .update(readFileSync(resolve(path)))
      .digest("hex");
  }

  newId(prefix: string): string {
    return `${prefix}-${randomUUID()}`;
  }

  private assertContained(candidate: string): void {
    if (
      candidate !== this.workDir &&
      !candidate.startsWith(`${this.workDir}${sep}`)
    ) {
      throw new Error(`path escapes run directory: ${candidate}`);
    }
  }

  private readExistingEventSeq(): number {
    const path = resolve(this.workDir, "events.jsonl");
    if (!existsSync(path)) return 0;
    const lines = readFileSync(path, "utf8").trim().split("\n").filter(Boolean);
    if (lines.length === 0) return 0;
    try {
      const last = JSON.parse(lines.at(-1)!) as { seq?: number };
      return Number.isInteger(last.seq) ? Number(last.seq) : lines.length;
    } catch {
      return lines.length;
    }
  }
}

function isImmutableDecisionContextPath(path: string): boolean {
  return /^contexts\/[^/]+\/(?:decision_context\.json|decision_observation\.json|research_memory_snapshot\.json|progress_trajectory_snapshot\.jsonl)$/.test(
    path,
  );
}
