import type {
  AppServerLiveEvent,
} from "./turns/app_server_runtime.ts";

export interface ConsoleWritable {
  write(chunk: string): unknown;
}

export class LiveConsoleRenderer {
  private readonly stream: ConsoleWritable;
  private agentOutputActive = false;
  private agentOutputEndsWithNewline = true;

  constructor(stream: ConsoleWritable = process.stderr) {
    this.stream = stream;
  }

  handle(event: AppServerLiveEvent): void {
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
          `turn:start role=${event.role} attempt=${event.attemptId} ` +
            `effort=${event.effort} approval=${event.approvalPolicy} ` +
            `sandbox=${event.sandbox}`,
        );
        return;
      case "turn_started":
        this.line(
          `turn:provider-started role=${event.role} ` +
            `thread=${event.threadId} turn=${event.turnId}`,
        );
        return;
      case "agent_output_delta":
        if (!this.agentOutputActive) {
          this.line(
            `agent:output role=${event.role} attempt=${event.attemptId}`,
          );
          this.agentOutputActive = true;
        }
        this.stream.write(event.delta);
        this.agentOutputEndsWithNewline = event.delta.endsWith("\n");
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
          `turn:complete role=${event.role} attempt=${event.attemptId} ` +
            `status=${event.status} elapsed_ms=${event.elapsedMs} ` +
            `input_tokens=${event.usage.inputTokens} ` +
            `output_tokens=${event.usage.outputTokens} ` +
            `tool_calls=${event.toolCalls}`,
        );
        return;
      case "turn_timeout":
        this.closeAgentOutput();
        this.line(
          `turn:timeout role=${event.role} attempt=${event.attemptId} ` +
            `timeout_ms=${event.timeoutMs}`,
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
    this.stream.write(`[simple-loop ${new Date().toISOString()}] ${message}\n`);
  }
}
