import fs from "node:fs";
import path from "node:path";

import { canonicalJson, stableHash } from "./stable_ids.ts";
import { ZERO_USAGE } from "./types.ts";
import type {
  JsonValue,
  RunConfig,
  RunState,
  SessionRecord,
  TaskRecord,
} from "./types.ts";

function ensureDirectory(directory: string, mode = 0o755): void {
  fs.mkdirSync(directory, { recursive: true, mode });
}

function atomicWriteText(targetPath: string, text: string, mode = 0o644): void {
  ensureDirectory(path.dirname(targetPath));
  const temporaryPath = path.join(
    path.dirname(targetPath),
    `.${path.basename(targetPath)}.${process.pid}.${Date.now()}.tmp`,
  );
  fs.writeFileSync(temporaryPath, text, { encoding: "utf8", mode });
  fs.renameSync(temporaryPath, targetPath);
}

export function atomicWriteJson(targetPath: string, value: JsonValue | RunState | RunConfig | SessionRecord | TaskRecord): void {
  atomicWriteText(targetPath, `${JSON.stringify(value, null, 2)}\n`);
}

export function appendJsonLine(targetPath: string, value: JsonValue): void {
  ensureDirectory(path.dirname(targetPath));
  fs.appendFileSync(targetPath, `${JSON.stringify(value)}\n`, "utf8");
}

function jsonLines(values: unknown[]): string {
  if (values.length === 0) {
    return "";
  }
  return `${values.map((value) => JSON.stringify(value)).join("\n")}\n`;
}

export function createInitialState(runId: string, config: RunConfig): RunState {
  const now = new Date().toISOString();
  return {
    schemaVersion: 1,
    runId,
    status: "initialized",
    config,
    provider: null,
    stage1: {
      status: "pending",
      round: 0,
      taskCount: 0,
      consecutiveRoundsWithoutNewAnchor: 0,
      acceptedAnchorIds: [],
      stopReason: null,
      anchorSpaceVersion: null,
      controllerLastOutput: null,
    },
    stage2: {
      status: "pending",
      anchorPlanning: {},
      directionReview: {},
    },
    claims: [],
    rejectedClaims: [],
    entities: [],
    baselines: [],
    anchors: [],
    entries: [],
    edges: [],
    directions: [],
    reviews: [],
    sessions: {},
    tasks: {},
    usage: {
      turns: 0,
      total: { ...ZERO_USAGE },
      byRole: {},
    },
    validation: {
      ok: false,
      errors: ["workflow has not been validated"],
      warnings: [],
      checkedAt: null,
    },
    createdAt: now,
    updatedAt: now,
  };
}

export class CanonicalStore {
  readonly workDir: string;
  readonly statePath: string;
  readonly eventsPath: string;

  constructor(workDir: string) {
    this.workDir = path.resolve(workDir);
    this.statePath = path.join(this.workDir, "state.json");
    this.eventsPath = path.join(this.workDir, "events.jsonl");
  }

  initialize(state: RunState): void {
    if (fs.existsSync(this.statePath)) {
      throw new Error(`run already initialized: ${this.workDir}`);
    }
    this.ensureLayout();
    this.save(state, "run_initialized");
  }

  ensureLayout(): void {
    const directories = [
      this.workDir,
      "provider/generated_schema",
      "provider/raw_turns",
      "provider/codex_home",
      "provider/sqlite",
      "provider/role_sandboxes",
      "sessions/direction_planners",
      "sessions/direction_reviewers",
      "tasks/anchor_evidence",
      "tasks/anchor_curation",
      "tasks/review_evidence",
      "evidence",
      "catalog",
      "anchors",
      "directions",
      "reviews",
    ];
    for (const directory of directories) {
      const absolute = path.isAbsolute(directory) ? directory : path.join(this.workDir, directory);
      ensureDirectory(absolute, directory.startsWith("provider") ? 0o700 : 0o755);
    }
  }

  exists(): boolean {
    return fs.existsSync(this.statePath);
  }

  load(): RunState {
    try {
      return JSON.parse(fs.readFileSync(this.statePath, "utf8")) as RunState;
    } catch (error) {
      const recovered = this.recoverFromEvents();
      if (recovered) {
        atomicWriteJson(this.statePath, recovered);
        return recovered;
      }
      throw new Error(`cannot load state and no checkpoint is recoverable: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  recoverFromEvents(): RunState | null {
    if (!fs.existsSync(this.eventsPath)) {
      return null;
    }
    let recovered: RunState | null = null;
    const lines = fs.readFileSync(this.eventsPath, "utf8").split(/\r?\n/);
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as Record<string, unknown>;
        if (event.type === "state_checkpoint" && event.state && typeof event.state === "object") {
          const candidate = event.state as RunState;
          const expectedHash = String(event.stateHash ?? "");
          const actualHash = stableHash(candidate as unknown as JsonValue, 32);
          if (!expectedHash || expectedHash === actualHash) {
            recovered = candidate;
          }
        }
      } catch {
        // A truncated last event is ignored; the most recent complete checkpoint remains valid.
      }
    }
    return recovered;
  }

  save(state: RunState, reason: string): void {
    this.ensureLayout();
    state.updatedAt = new Date().toISOString();
    state.validation = {
      ok: false,
      errors: [`state changed after last validation: ${reason}`],
      warnings: [],
      checkedAt: null,
    };
    atomicWriteJson(path.join(this.workDir, "config.json"), state.config);
    atomicWriteJson(this.statePath, state);
    appendJsonLine(this.eventsPath, {
      type: "state_checkpoint",
      timestamp: state.updatedAt,
      reason,
      stateHash: stableHash(state as unknown as JsonValue, 32),
      state: state as unknown as JsonValue,
    });
    this.materialize(state);
  }

  saveValidatedState(state: RunState): void {
    state.updatedAt = new Date().toISOString();
    atomicWriteJson(this.statePath, state);
    appendJsonLine(this.eventsPath, {
      type: "validation_checkpoint",
      timestamp: state.updatedAt,
      stateHash: stableHash(state as unknown as JsonValue, 32),
      validation: state.validation as unknown as JsonValue,
      state: state as unknown as JsonValue,
    });
    this.materialize(state);
  }

  recordEvent(type: string, payload: JsonValue): void {
    appendJsonLine(this.eventsPath, {
      type,
      timestamp: new Date().toISOString(),
      payload,
    });
  }

  private materialize(state: RunState): void {
    atomicWriteText(path.join(this.workDir, "evidence/claims.jsonl"), jsonLines(
      [...state.claims].sort((left, right) => left.claimId.localeCompare(right.claimId)),
    ));
    atomicWriteText(path.join(this.workDir, "evidence/rejected_claims.jsonl"), jsonLines(state.rejectedClaims));
    atomicWriteText(path.join(this.workDir, "evidence/review_claims.jsonl"), jsonLines(
      state.claims.filter((claim) => claim.taskId.startsWith("RE-")),
    ));
    atomicWriteJson(path.join(this.workDir, "catalog/entities.json"), {
      entities: [...state.entities].sort((left, right) => left.entityId.localeCompare(right.entityId)),
      baselines: [...state.baselines].sort((left, right) => left.baselineId.localeCompare(right.baselineId)),
    });

    const anchors = [...state.anchors].sort((left, right) => left.anchorId.localeCompare(right.anchorId));
    atomicWriteJson(path.join(this.workDir, "anchors/index.json"), anchors.map((anchor) => ({
      anchorId: anchor.anchorId,
      title: anchor.title,
      status: anchor.status,
      signature: anchor.signature,
    })));
    for (const anchor of anchors) {
      atomicWriteJson(path.join(this.workDir, "anchors", `${anchor.anchorId}.json`), {
        anchor,
        baselines: state.baselines.filter((baseline) => anchor.baselineIds.includes(baseline.baselineId)),
        entries: state.entries.filter((entry) => entry.anchorId === anchor.anchorId),
        edges: state.edges.filter((edge) => edge.anchorId === anchor.anchorId),
      });
    }

    const directions = [...state.directions].sort((left, right) => left.directionId.localeCompare(right.directionId));
    atomicWriteJson(path.join(this.workDir, "directions/index.json"), directions.map((direction) => ({
      directionId: direction.directionId,
      anchorId: direction.anchorId,
      status: direction.status,
      hypothesis: direction.hypothesis,
    })));
    for (const direction of directions) {
      atomicWriteJson(path.join(this.workDir, "directions", `${direction.directionId}.json`), direction);
    }
    for (const review of state.reviews) {
      atomicWriteJson(path.join(this.workDir, "reviews", `${review.directionId}.json`), review);
    }

    for (const [key, session] of Object.entries(state.sessions)) {
      const relative = session.role === "anchor_stage_controller"
        ? "sessions/anchor_controller.json"
        : session.role === "direction_planner"
          ? `sessions/direction_planners/${session.scopeId}.json`
          : session.role === "direction_reviewer"
            ? `sessions/direction_reviewers/${session.scopeId}.json`
            : `sessions/${key}.json`;
      atomicWriteJson(path.join(this.workDir, relative), session);
    }
    for (const task of Object.values(state.tasks)) {
      const directory = task.role === "anchor_evidence_worker"
        ? "tasks/anchor_evidence"
        : task.role === "anchor_curator_worker"
          ? "tasks/anchor_curation"
          : "tasks/review_evidence";
      atomicWriteJson(path.join(this.workDir, directory, `${task.taskId}.json`), task);
    }
  }
}

export function stateDigest(state: RunState): string {
  return stableHash(state as unknown as JsonValue, 32);
}

export function canonicalStateText(state: RunState): string {
  return canonicalJson(state as unknown as JsonValue);
}

