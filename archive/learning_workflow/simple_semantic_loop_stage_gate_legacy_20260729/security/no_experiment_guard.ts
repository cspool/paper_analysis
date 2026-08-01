import type {
  EvidencePacketEnvelope,
  EvidenceReaderTaskEnvelope,
  ExperimentHandoff,
  StageContractDraft,
  ValidationReport,
} from "../contracts/index.ts";
import { addError, emptyReport } from "../validators/schema_validator.ts";

const FORBIDDEN_TOOL_PATTERNS = [
  /(^|[_.:/-])(shell|bash|command|exec)([_.:/-]|$)/i,
  /(^|[_.:/-])(build|compile|benchmark|profile)([_.:/-]|$)/i,
  /(^|[_.:/-])(gpu|cluster|remote[_-]?job)([_.:/-]|$)/i,
  /(^|[_.:/-])(write|patch|edit|apply_patch)([_.:/-]|$)/i,
  /experiment[_-]?(launch|run|execute)/i,
];

export const ALLOWED_EVIDENCE_TOOLS = Object.freeze([
  "mcp__obsidian__obsidian_search_notes",
  "mcp__obsidian__obsidian_get_note",
] as const);

export function isForbiddenExecutionCapability(value: string): boolean {
  return FORBIDDEN_TOOL_PATTERNS.some((pattern) => pattern.test(value));
}

export function validateNoExperimentStage(
  stage: StageContractDraft,
): ValidationReport {
  const report = emptyReport();
  stage.requestedTools.forEach((tool, index) => {
    if (isForbiddenExecutionCapability(tool)) {
      addError(
        report,
        "security.experiment_or_execution_tool",
        `/requestedTools/${index}`,
        `execution capability is forbidden: ${tool}`,
      );
    }
  });
  stage.prohibitedActions.forEach((action, index) => {
    // The prohibited list may and should name forbidden actions.
    if (!action.trim()) {
      addError(
        report,
        "security.empty_prohibited_action",
        `/prohibitedActions/${index}`,
        "prohibited action must be explicit",
      );
    }
  });
  return report;
}

export function validateExperimentHandoff(
  handoff: ExperimentHandoff | null,
  pointer = "/experimentHandoff",
): ValidationReport {
  const report = emptyReport();
  if (handoff && handoff.executionAuthorized !== false) {
    addError(
      report,
      "no_experiment.execution_authorized",
      `${pointer}/executionAuthorized`,
      "ExperimentHandoff must remain a non-executable handoff",
    );
  }
  if (handoff) {
    const executableText = [
      ...handoff.suggestedEntryPoints,
      ...handoff.acceptanceCriteria,
      ...handoff.failureStopConditions,
    ].find((value) =>
      /(?:^|\s)(?:bash|sh|zsh|python\d*|torchrun|make|cmake|ninja|sbatch|srun|kubectl|docker|nsys|ncu)(?:\s|$)|(?:^|\s)\.\/|```|\$\s/.test(
        value,
      ),
    );
    if (executableText) {
      addError(
        report,
        "no_experiment.executable_handoff_instruction",
        pointer,
        "ExperimentHandoff may describe an artifact and criteria but cannot contain an executable command",
      );
    }
  }
  return report;
}

export interface RuntimeToolEvent {
  toolName: string;
  arguments: Record<string, unknown>;
  status: string;
  resultText: string;
  error: unknown | null;
}

export function validateRuntimeToolEvents(
  role: "workflow_decision" | "evidence_reader" | "direction_reviewer" | "closure_reviewer",
  events: readonly RuntimeToolEvent[],
  evidenceTask?: EvidenceReaderTaskEnvelope,
): ValidationReport {
  const report = emptyReport();
  if (role !== "evidence_reader" && events.length > 0) {
    addError(
      report,
      "security.zero_tool_role_event",
      "/toolEvents",
      `${role} is a zero-tool role`,
    );
    return report;
  }
  events.forEach((event, index) => {
    if (event.status !== "completed" || event.error !== null) {
      addError(
        report,
        "security.tool_call_failed",
        `/toolEvents/${index}`,
        `tool call did not complete successfully: status=${event.status}`,
      );
    }
    if (
      role === "evidence_reader" &&
      !ALLOWED_EVIDENCE_TOOLS.includes(
        event.toolName as (typeof ALLOWED_EVIDENCE_TOOLS)[number],
      )
    ) {
      addError(
        report,
        "security.unregistered_tool",
        `/toolEvents/${index}/toolName`,
        `tool is not admitted for evidence_reader: ${event.toolName}`,
      );
    }
    if (isForbiddenExecutionCapability(event.toolName)) {
      addError(
        report,
        "security.execution_event",
        `/toolEvents/${index}/toolName`,
        `forbidden runtime capability: ${event.toolName}`,
      );
    }
    if (role === "evidence_reader") {
      validateEvidenceToolEvent(event, index, evidenceTask, report);
    }
  });
  if (
    role === "evidence_reader" &&
    evidenceTask &&
    events.length > evidenceTask.payload.budget.maxToolCalls
  ) {
    addError(
      report,
      "security.tool_call_budget",
      "/toolEvents",
      "observed tool calls exceed the frozen Evidence Turn budget",
    );
  }
  return report;
}

export function validateEvidenceRuntimeTrace(
  task: EvidenceReaderTaskEnvelope,
  result: EvidencePacketEnvelope,
  events: readonly RuntimeToolEvent[],
): ValidationReport {
  const report = validateRuntimeToolEvents(
    "evidence_reader",
    events,
    task,
  );
  const searchEvents = events.filter(
    (event) =>
      event.toolName === "mcp__obsidian__obsidian_search_notes",
  );
  const readEvents = events.filter(
    (event) =>
      event.toolName === "mcp__obsidian__obsidian_get_note",
  );
  if (
    searchEvents.length !== result.payload.searches.length ||
    readEvents.length !== result.payload.contextsRead.length
  ) {
    addError(
      report,
      "security.evidence_tool_ledger_count",
      "/toolEvents",
      "observed Obsidian calls must exactly match the EvidencePacket search/read ledger",
    );
    return report;
  }
  const expectedCallOrder = [
    ...result.payload.searches.map((search) => ({
      sequence: search.sequence,
      toolName: "mcp__obsidian__obsidian_search_notes",
    })),
    ...result.payload.contextsRead.map((context) => ({
      sequence: context.sequence,
      toolName: "mcp__obsidian__obsidian_get_note",
    })),
  ].sort((left, right) => left.sequence - right.sequence);
  if (
    events.length !== expectedCallOrder.length ||
    expectedCallOrder.some(
      (expected, index) => events[index]?.toolName !== expected.toolName,
    )
  ) {
    addError(
      report,
      "security.evidence_tool_ledger_order",
      "/toolEvents",
      "observed Obsidian calls must follow the global search/context sequence ledger",
    );
  }
  result.payload.searches.forEach((search, index) => {
    const event = searchEvents[index];
    if (
      !event ||
      event.arguments.mode !== "omnisearch" ||
      event.arguments.query !== search.query ||
      normalizeNullableString(event.arguments.cursor) !== search.cursorUsed
    ) {
      addError(
        report,
        "security.evidence_search_trace",
        `/toolEvents/${index}`,
        "Evidence search ledger does not match the observed Obsidian search call",
      );
    }
  });
  const searchEventById = new Map(
    result.payload.searches.map((search, index) => [
      search.searchId,
      searchEvents[index],
    ]),
  );
  result.payload.hitsConsidered.forEach((hit, index) => {
    if (!hit.selected) return;
      const event = searchEventById.get(hit.searchId);
      if (!event || !event.resultText.includes(hit.path)) {
        addError(
          report,
          "security.evidence_hit_not_observed",
          `/payload/hitsConsidered/${index}`,
          "a selected Evidence hit path must occur in the observed search result",
        );
      }
  });
  result.payload.contextsRead.forEach((context, index) => {
    const event = readEvents[index];
    const offset = searchEvents.length + index;
    if (
      !event ||
      getPathTarget(event.arguments) !== context.path ||
      event.arguments.format !== context.format ||
      (context.format === "section" &&
        !matchesSectionTarget(event.arguments, context.sectionTarget)) ||
      !event.resultText.includes(context.exactContext)
    ) {
      addError(
        report,
        "security.evidence_read_trace",
        `/toolEvents/${offset}`,
        "Evidence context ledger or exact context does not match the observed Obsidian read result",
      );
    }
  });
  return report;
}

function validateEvidenceToolEvent(
  event: RuntimeToolEvent,
  index: number,
  task: EvidenceReaderTaskEnvelope | undefined,
  report: ValidationReport,
): void {
  if (!task) return;
  const allowed = task.payload.permission.allowedPathPrefixes;
  if (event.toolName === "mcp__obsidian__obsidian_search_notes") {
    const query =
      typeof event.arguments.query === "string"
        ? event.arguments.query
        : "";
    const pathTokens = [...query.matchAll(/(?:^|\s)(path:[^\s]+)/g)].map(
      (match) => match[1],
    );
    const admittedPath = allowed.some((prefix) =>
      pathTokens.includes(`path:${prefix}`),
    );
    const allowedKeys = new Set(["mode", "query", "cursor"]);
    const hasUnexpectedArgument = Object.keys(event.arguments).some(
      (key) => !allowedKeys.has(key),
    );
    const cursorIsValid =
      !Object.hasOwn(event.arguments, "cursor") ||
      typeof event.arguments.cursor === "string";
    if (
      event.arguments.mode !== "omnisearch" ||
      pathTokens.length !== 1 ||
      !admittedPath ||
      hasUnexpectedArgument ||
      !cursorIsValid
    ) {
      addError(
        report,
        "security.obsidian_search_arguments",
        `/toolEvents/${index}/arguments`,
        "search must use omnisearch with exactly one task-authorized path filter",
      );
    }
    return;
  }
  if (event.toolName === "mcp__obsidian__obsidian_get_note") {
    const path = getPathTarget(event.arguments);
    const format =
      typeof event.arguments.format === "string"
        ? event.arguments.format
        : "";
    const allowedKeys =
      format === "section"
        ? new Set(["target", "format", "section"])
        : new Set(["target", "format"]);
    const hasUnexpectedArgument = Object.keys(event.arguments).some(
      (key) => !allowedKeys.has(key),
    );
    const validSection =
      format !== "section" ||
      getHeadingTarget(event.arguments.section) !== null;
    if (
      !isContainedVaultPath(path, allowed) ||
      !["document-map", "section", "content", "full"].includes(format) ||
      hasUnexpectedArgument ||
      !validSection
    ) {
      addError(
        report,
        "security.obsidian_read_arguments",
        `/toolEvents/${index}/arguments`,
        "read must target a task-authorized path and an admitted read-only format",
      );
    }
  }
}

function normalizeNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function matchesSectionTarget(
  argumentsValue: Record<string, unknown>,
  expected: string | null,
): boolean {
  if (!expected) return false;
  return getHeadingTarget(argumentsValue.section) === expected;
}

function getPathTarget(argumentsValue: Record<string, unknown>): string {
  const target = argumentsValue.target;
  if (
    typeof target !== "object" ||
    target === null ||
    Array.isArray(target)
  ) {
    return "";
  }
  const record = target as Record<string, unknown>;
  return (
    Object.keys(record).length === 2 &&
    record.type === "path" &&
    typeof record.path === "string"
  )
    ? record.path
    : "";
}

function getHeadingTarget(value: unknown): string | null {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    return null;
  }
  const record = value as Record<string, unknown>;
  return (
    Object.keys(record).length === 2 &&
    record.type === "heading" &&
    typeof record.target === "string"
  )
    ? record.target
    : null;
}

function isContainedVaultPath(
  path: string,
  allowedPrefixes: readonly string[],
): boolean {
  if (
    !path ||
    path.startsWith("/") ||
    path.includes("\\") ||
    path.includes("\0")
  ) {
    return false;
  }
  const segments = path.split("/");
  if (segments.some((segment) => segment === "." || segment === "..")) {
    return false;
  }
  return allowedPrefixes.some(
    (prefix) => path.startsWith(prefix) && path.length > prefix.length,
  );
}
