import type {
  RegisteredRole,
  ReasoningEffort,
} from "../contracts/index.ts";
import type { RuntimeToolEvent } from "../security/no_experiment_guard.ts";

export interface FrozenTurnDispatch {
  attemptId: string;
  taskId: string;
  role: RegisteredRole;
  model: string;
  logicalEffort: ReasoningEffort;
  providerWireEffort: string;
  prompt: string;
  outputSchema: unknown;
  cwd: string;
  timeoutMs: number;
}

export interface TokenUsage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningOutputTokens: number;
  totalTokens: number;
}

export interface RawTurnResult {
  attemptId: string;
  providerThreadId: string;
  providerTurnId: string;
  status: "completed" | "interrupted" | "failed";
  text: string;
  usage: TokenUsage;
  toolEvents: RuntimeToolEvent[];
  rawEvents: unknown[];
  compacted: boolean;
  error: string | null;
  elapsedMs: number;
}

export interface AttemptRecord {
  attemptId: string;
  providerThreadId: string | null;
  providerTurnId: string | null;
  status: string;
}

export interface ReconcileResult {
  status: "completed" | "interrupted" | "unknown";
  recoverableResult: RawTurnResult | null;
}

export interface FreshTurnRuntime {
  run(task: FrozenTurnDispatch): Promise<RawTurnResult>;
  cancel(attemptId: string): Promise<void>;
  reconcile(attempt: AttemptRecord): Promise<ReconcileResult>;
  close?(): Promise<void>;
}

export class ScriptedFreshTurnRuntime implements FreshTurnRuntime {
  private sequence = 0;
  private readonly active = new Map<string, { cancelled: boolean }>();
  private readonly handler: (
    dispatch: FrozenTurnDispatch,
    invocation: number,
  ) => Promise<
    | string
    | Partial<Omit<RawTurnResult, "attemptId" | "providerThreadId" | "providerTurnId">>
  >;

  constructor(
    handler: (
      dispatch: FrozenTurnDispatch,
      invocation: number,
    ) => Promise<
      | string
      | Partial<Omit<RawTurnResult, "attemptId" | "providerThreadId" | "providerTurnId">>
    >,
  ) {
    this.handler = handler;
  }

  async run(dispatch: FrozenTurnDispatch): Promise<RawTurnResult> {
    this.sequence += 1;
    const invocation = this.sequence;
    const marker = { cancelled: false };
    this.active.set(dispatch.attemptId, marker);
    const started = Date.now();
    const output = await this.handler(dispatch, invocation);
    this.active.delete(dispatch.attemptId);
    if (marker.cancelled) {
      return baseResult(
        dispatch,
        invocation,
        "interrupted",
        "",
        Date.now() - started,
        "cancelled",
      );
    }
    if (typeof output === "string") {
      return {
        ...baseResult(
          dispatch,
          invocation,
          "completed",
          output,
          Date.now() - started,
          null,
        ),
      };
    }
    return {
      ...baseResult(
        dispatch,
        invocation,
        output.status ?? "completed",
        output.text ?? "",
        Date.now() - started,
        output.error ?? null,
      ),
      ...output,
      attemptId: dispatch.attemptId,
      providerThreadId: `scripted-thread-${invocation}`,
      providerTurnId: `scripted-turn-${invocation}`,
    };
  }

  async cancel(attemptId: string): Promise<void> {
    const active = this.active.get(attemptId);
    if (active) active.cancelled = true;
  }

  async reconcile(_attempt: AttemptRecord): Promise<ReconcileResult> {
    return { status: "unknown", recoverableResult: null };
  }
}

function baseResult(
  dispatch: FrozenTurnDispatch,
  invocation: number,
  status: RawTurnResult["status"],
  text: string,
  elapsedMs: number,
  error: string | null,
): RawTurnResult {
  return {
    attemptId: dispatch.attemptId,
    providerThreadId: `scripted-thread-${invocation}`,
    providerTurnId: `scripted-turn-${invocation}`,
    status,
    text,
    usage: {
      inputTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      reasoningOutputTokens: 0,
      totalTokens: 0,
    },
    toolEvents: [],
    rawEvents: [],
    compacted: false,
    error,
    elapsedMs,
  };
}
