import { randomUUID } from "node:crypto";
import {
  appendFileSync,
  closeSync,
  existsSync,
  linkSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type { PaperCatchRun, ReportManifest } from "./types.ts";

export class PaperCatchStore {
  readonly outputDir: string;
  private lockHeld = false;

  constructor(outputDir: string) {
    this.outputDir = resolve(outputDir);
  }

  initialize(): void {
    mkdirSync(this.outputDir, { recursive: true });
    for (const path of [
      ".state",
      ".state/cache",
      ".state/reports",
      ".runs",
      "logs",
    ]) mkdirSync(this.absolute(path), { recursive: true });
  }

  absolute(path: string): string {
    const candidate = resolve(isAbsolute(path) ? path : resolve(this.outputDir, path));
    if (candidate !== this.outputDir && !candidate.startsWith(`${this.outputDir}${sep}`)) {
      throw new Error(`path escapes paper_catch directory: ${candidate}`);
    }
    return candidate;
  }

  relative(path: string): string {
    return relative(this.outputDir, this.absolute(path)).split(sep).join("/");
  }

  exists(path: string): boolean {
    return existsSync(this.absolute(path));
  }

  readText(path: string): string {
    return readFileSync(this.absolute(path), "utf8");
  }

  readJson<T>(path: string): T {
    return JSON.parse(this.readText(path)) as T;
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
      linkSync(temporary, target);
    } finally {
      if (existsSync(temporary)) unlinkSync(temporary);
    }
  }

  writeImmutableJson(path: string, value: unknown): void {
    this.writeImmutableText(path, `${JSON.stringify(value, null, 2)}\n`);
  }

  appendJsonLine(path: string, value: unknown): void {
    const target = this.absolute(path);
    mkdirSync(dirname(target), { recursive: true });
    appendFileSync(target, `${JSON.stringify(value)}\n`, "utf8");
  }

  acquireLock(): void {
    this.initialize();
    const lockRef = ".state/run.lock";
    const path = this.absolute(lockRef);
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
      if (live) throw new Error(`paper catch is already running: ${path}`);
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
    const path = this.absolute(".state/run.lock");
    if (existsSync(path)) unlinkSync(path);
    this.lockHeld = false;
  }

  writeRun(run: PaperCatchRun): void {
    this.writeJson(`.runs/${run.runId}/run.json`, run);
  }

  readRun(runId: string): PaperCatchRun {
    return this.readJson<PaperCatchRun>(`.runs/${runId}/run.json`);
  }

  activeRunId(): string | null {
    if (!this.exists(".state/active_run.json")) return null;
    const value = this.readJson<{ runId?: string }>(".state/active_run.json");
    return typeof value.runId === "string" ? value.runId : null;
  }

  setActiveRun(runId: string): void {
    this.writeJson(".state/active_run.json", { runId, updatedAt: new Date().toISOString() });
  }

  clearActiveRun(runId: string): void {
    if (this.activeRunId() !== runId) return;
    const path = this.absolute(".state/active_run.json");
    if (existsSync(path)) unlinkSync(path);
  }

  latestReport(): { reportRef: string; manifestRef: string | null; timestamp: string } | null {
    const names = readdirSync(this.outputDir)
      .filter((name) => /^\d{8}_\d{6}\.md$/.test(name))
      .sort();
    const name = names.at(-1);
    if (!name) return null;
    const reportId = name.slice(0, -3);
    const manifestRef = `.state/reports/${reportId}.json`;
    return {
      reportRef: name,
      manifestRef: this.exists(manifestRef) ? manifestRef : null,
      timestamp: reportIdToIso(reportId),
    };
  }

  readLatestManifest(): ReportManifest | null {
    const latest = this.latestReport();
    if (!latest?.manifestRef) return null;
    return this.readJson<ReportManifest>(latest.manifestRef);
  }
}

export function timestampId(date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}${values.month}${values.day}_${values.hour}${values.minute}${values.second}`;
}

export function reportIdToIso(reportId: string): string {
  const match = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(reportId);
  if (!match) throw new Error(`invalid report timestamp id: ${reportId}`);
  return `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}+08:00`;
}
