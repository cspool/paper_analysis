import { createHash } from "node:crypto";
import { FileLoopStore } from "./store.ts";
import type {
  RuntimeRecoveryRecord,
  StateFile,
  TurnFile,
  TurnTimeoutProfile,
} from "./types.ts";
import { CURRENT_FORMAT_VERSION } from "./types.ts";
import { validateRun } from "./validation.ts";

export interface RuntimeRecoveryOptions {
  token: string;
  timeoutOverride?: Partial<TurnTimeoutProfile>;
}

export interface RuntimeRecoveryAuthorizationResult {
  status: "AUTHORIZED" | "ALREADY_AUTHORIZED" | "ALREADY_CONSUMED";
  recoveryRef: string;
  sourceTurnRef: string;
}

export function authorizeRuntimeRecovery(
  store: FileLoopStore,
  options: RuntimeRecoveryOptions,
): RuntimeRecoveryAuthorizationResult {
  const token = options.token.trim();
  if (!token) throw new Error("recovery token must not be empty");
  const tokenHash = createHash("sha256").update(token).digest("hex");
  const recoveryId = `recovery-${tokenHash.slice(0, 24)}`;
  const recoveryRef = `recoveries/${recoveryId}.json`;

  store.acquireLock();
  try {
    const run = store.readRun();
    const initialState = store.readState();
    if (
      run.formatVersion !== CURRENT_FORMAT_VERSION ||
      initialState.formatVersion !== CURRENT_FORMAT_VERSION
    ) {
      throw new Error(
        `formatVersion ${run.formatVersion}/${initialState.formatVersion} is read-only; runtime recovery requires formatVersion ${CURRENT_FORMAT_VERSION}`,
      );
    }
    if (store.exists(recoveryRef)) {
      const existing = store.readJson<RuntimeRecoveryRecord>(recoveryRef);
      const state = store.readState();
      const consumedTurns = store.turnRefs()
        .filter((ref) => store.readTurn(ref).recoveryRef === recoveryRef)
        .sort((left, right) =>
          store.readTurn(left).startedAt.localeCompare(
            store.readTurn(right).startedAt,
          )
        );
      const consumedRef = consumedTurns.at(-1) ?? null;
      if (
        consumedRef &&
        state.lifecycle === "RUNNING" &&
        (
          state.activeTurnRef === consumedRef ||
          (
            !state.activeTurnRef &&
            state.sequence[0]?.role === existing.role &&
            state.sequence[0]?.bindingRef === existing.taskBindingRef
          )
        )
      ) {
        if (state.activeTurnRef !== consumedRef || state.runtimeRecovery) {
          store.writeState({
            ...state,
            revision: state.revision + 1,
            activeTurnRef: consumedRef,
            activeTaskBindingRef: existing.taskBindingRef,
            runtimeRecovery: null,
          }, "RUNTIME_RECOVERY_ORPHAN_REBOUND");
        }
        return {
          status: "ALREADY_AUTHORIZED",
          recoveryRef,
          sourceTurnRef: existing.sourceTurnRef,
        };
      }
      if (consumedRef) {
        return {
          status: "ALREADY_CONSUMED",
          recoveryRef,
          sourceTurnRef: existing.sourceTurnRef,
        };
      }
      if (state.runtimeRecovery?.recoveryRef === recoveryRef) {
        return {
          status: "ALREADY_AUTHORIZED",
          recoveryRef,
          sourceTurnRef: existing.sourceTurnRef,
        };
      }
      throw new Error("recovery record exists but is not bound to current state");
    }

    const report = validateRun(store.workDir);
    if (!report.valid) {
      const failed = report.checks
        .filter((check) => !check.passed)
        .flatMap((check) =>
          check.details.map((detail) => `${check.name}: ${detail}`)
        );
      throw new Error(`run audit failed: ${failed.join("; ")}`);
    }
    const state = store.readState();
    if (
      state.lifecycle !== "FAILED" ||
      state.failureKind !== "RUNTIME_RETRY_EXHAUSTED"
    ) {
      throw new Error(
        "recover-runtime requires FAILED with failureKind=RUNTIME_RETRY_EXHAUSTED",
      );
    }
    if (state.activeTurnRef) {
      throw new Error("recover-runtime refuses a conflicting active Turn");
    }
    const head = state.sequence[0];
    if (!head) throw new Error("recover-runtime requires a sequence head");
    const source = latestFailedTurn(store, state);
    if (!source) {
      throw new Error("no current RUNTIME_FAILED Turn matches the sequence head");
    }
    const lastTurnRef = store.turnRefs()
      .sort((left, right) =>
        store.readTurn(left).startedAt.localeCompare(
          store.readTurn(right).startedAt,
        )
      )
      .at(-1);
    if (lastTurnRef !== source.ref) {
      throw new Error("a later Turn exists after the recoverable runtime failure");
    }
    const timeoutOverride = normalizeOverride(options.timeoutOverride ?? {});
    const effectiveTimeout = {
      ...store.readRun().budgets.timeoutProfiles[source.turn.role],
      ...timeoutOverride,
    };
    if (effectiveTimeout.hardTimeoutMs < effectiveTimeout.idleTimeoutMs) {
      throw new Error(
        "effective hardTimeoutMs must be >= effective idleTimeoutMs",
      );
    }
    const record: RuntimeRecoveryRecord = {
      formatVersion: CURRENT_FORMAT_VERSION,
      recoveryId,
      recoveryRef,
      tokenHash,
      createdAt: new Date().toISOString(),
      sourceStateRevision: state.revision,
      sourceTurnRef: source.ref,
      role: source.turn.role,
      taskBindingRef: source.turn.taskBindingRef,
      decisionContextRef: source.turn.decisionContextRef,
      timeoutOverride,
    };
    store.writeJson(recoveryRef, record);
    store.appendEvent("RUNTIME_RECOVERY_AUTHORIZED", [
      recoveryRef,
      source.ref,
    ]);
    const next: StateFile = {
      ...state,
      revision: state.revision + 1,
      lifecycle: "RUNNING",
      reason: null,
      failureKind: null,
      node: head.role,
      runtimeRecovery: {
        recoveryRef,
        sourceTurnRef: source.ref,
        role: source.turn.role,
        taskBindingRef: source.turn.taskBindingRef,
        decisionContextRef: source.turn.decisionContextRef,
        timeoutOverride,
      },
    };
    store.writeState(next, "RUNTIME_RECOVERY_STATE_PREPARED");
    return { status: "AUTHORIZED", recoveryRef, sourceTurnRef: source.ref };
  } finally {
    store.releaseLock();
  }
}

function latestFailedTurn(
  store: FileLoopStore,
  state: StateFile,
): { ref: string; turn: TurnFile } | null {
  const head = state.sequence[0];
  if (!head) return null;
  return store.turnRefs()
    .map((ref) => ({ ref, turn: store.readTurn(ref) }))
    .filter(({ turn }) =>
      turn.turnState === "RUNTIME_FAILED" &&
      turn.role === head.role &&
      turn.round === state.round &&
      turn.taskBindingRef === head.bindingRef
    )
    .sort((left, right) =>
      left.turn.startedAt.localeCompare(right.turn.startedAt)
    )
    .at(-1) ?? null;
}

function normalizeOverride(
  value: Partial<TurnTimeoutProfile>,
): Partial<TurnTimeoutProfile> {
  const result: Partial<TurnTimeoutProfile> = {};
  for (const key of [
    "idleTimeoutMs",
    "hardTimeoutMs",
    "interruptGraceMs",
  ] as const) {
    const candidate = value[key];
    if (candidate === undefined) continue;
    if (!Number.isInteger(candidate) || candidate < 1) {
      throw new Error(`${key} override must be a positive integer`);
    }
    result[key] = candidate;
  }
  if (
    result.idleTimeoutMs !== undefined &&
    result.hardTimeoutMs !== undefined &&
    result.hardTimeoutMs < result.idleTimeoutMs
  ) {
    throw new Error("hardTimeoutMs override must be >= idleTimeoutMs override");
  }
  return result;
}
