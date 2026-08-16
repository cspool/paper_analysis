import type { RuntimeLiveEvent } from "./runtime.ts";

export interface ConsoleWritable {
  write(chunk: string): unknown;
}

export class LiveConsoleRenderer {
  private agentOutputActive = false;
  private agentOutputEndsWithNewline = true;
  private readonly stream: ConsoleWritable;
  private readonly prefix: string;

  constructor(
    stream: ConsoleWritable = process.stderr,
    prefix = "simple-loop",
  ) {
    this.stream = stream;
    this.prefix = prefix;
  }

  handle(event: RuntimeLiveEvent): void {
    switch (event.type) {
      case "runtime_started":
        this.line(`runtime:start command=${event.command}`);
        return;
      case "app_server_stderr":
        this.closeAgentOutput();
        for (const line of event.text.split(/\r?\n/).filter(Boolean)) {
          this.line(`app-server:stderr ${line}`);
        }
        return;
      case "turn_starting":
        this.closeAgentOutput();
        this.line(
          `turn:start role=${event.role} turn=${event.turnId} ` +
            `effort=${event.effort} approval=${event.approvalPolicy} ` +
            `sandbox=${event.sandbox}`,
        );
        return;
      case "turn_started":
        this.line(
          `turn:provider-started role=${event.role} ` +
            `thread=${event.threadId} turn=${event.providerTurnId}`,
        );
        return;
      case "agent_output_delta":
        if (!this.agentOutputActive) {
          this.line(
            `agent:stream-start role=${event.role} turn=${event.turnId}`,
          );
          this.agentOutputActive = true;
        }
        this.stream.write(event.delta);
        this.agentOutputEndsWithNewline = event.delta.endsWith("\n");
        return;
      case "agent_message_complete":
        this.closeAgentOutput();
        this.line(
          `agent:message-complete role=${event.role} turn=${event.turnId} ` +
            `item=${event.itemId} phase=${event.phase ?? "unknown"}`,
        );
        return;
      case "tool_status":
        this.closeAgentOutput();
        this.line(
          `tool:${event.phase} role=${event.role} ` +
            `tool=${event.toolName} status=${event.status}`,
        );
        return;
      case "turn_completed":
        this.closeAgentOutput();
        this.line(
          `turn:complete role=${event.role} turn=${event.turnId} ` +
            `status=${event.status} elapsed_ms=${event.elapsedMs} ` +
            `input_tokens=${event.usage.inputTokens} ` +
            `output_tokens=${event.usage.outputTokens} ` +
            `tool_calls=${event.toolCalls}`,
        );
        return;
      case "turn_timeout":
        this.closeAgentOutput();
        this.line(
          `turn:timeout role=${event.role} turn=${event.turnId} ` +
            `kind=${event.kind} timeout_ms=${event.timeoutMs} ` +
            `capture=${event.capture} ` +
            `runtime_ref=turns/${event.turnId}/runtime.jsonl ` +
            `partial_ref=turns/${event.turnId}/partial_output.txt`,
        );
        return;
      case "runtime_error":
        this.closeAgentOutput();
        this.line(`runtime:error ${event.message}`);
        return;
    }
  }

  finish(): void {
    this.closeAgentOutput();
  }

  private closeAgentOutput(): void {
    if (!this.agentOutputActive) return;
    if (!this.agentOutputEndsWithNewline) this.stream.write("\n");
    this.agentOutputActive = false;
    this.agentOutputEndsWithNewline = true;
  }

  private line(message: string): void {
    this.stream.write(`[${this.prefix} ${new Date().toISOString()}] ${message}\n`);
  }
}
