import type {
  DecisionObservation,
  DecisionProtocolResult,
  ExperimentGoalResult,
  LoopDecision,
  ProgressTrajectoryRecord,
  ResearchMemory,
  ResearchMemoryEntry,
  ReviewResult,
  StateFile,
  TaskAction,
  TaskBinding,
  TokenUsage,
  TurnFile,
  WorkflowGoal,
  WorkResult,
} from "./types.ts";
import { FileLoopStore } from "./store.ts";
import {
  buildNegativeExperimentIndex,
  rebuildNegativeExperimentIndex,
} from "./experiment_history.ts";
import {
  computeRemainingRequirements,
  previewBranchEffects,
} from "./workflow.ts";

const TRAJECTORY_REF = "observations/progress_trajectory.jsonl" as const;
const MEMORY_REF = "observations/research_memory.json" as const;

export function rebuildResearchMemory(
  store: FileLoopStore,
  state: StateFile,
): ResearchMemory {
  const memory = buildResearchMemory(store, state);
  store.writeJson(MEMORY_REF, memory);
  rebuildNegativeExperimentIndex(store, state);
  return memory;
}

export function buildResearchMemory(
  store: FileLoopStore,
  state: StateFile,
): ResearchMemory {
  const memory: ResearchMemory = {
    generatedAt: new Date().toISOString(),
    sourceStateRevision: state.revision,
    accepted: [],
    needsRevision: [],
    rejectedLessons: [],
    openQueryGaps: [],
    coverage: { L1: [], L2: [], L3: [], L4: [], L5: [], L6: [] },
    decisionTrail: [],
    experimentResults: [],
    requirements: computeRemainingRequirements(store, state, false),
  };
  const index = store.readObjects();
  for (const anchor of Object.values(index.anchors)) {
    const revision = anchor.revisions[String(anchor.latestRevision)];
    if (!revision) continue;
    const entry = memoryEntry(store, "ANCHOR", anchor.objectId, revision);
    classify(memory, entry);
    collectQueryGaps(store, memory, entry);
    if (!anchor.rejected) collectCoverage(store, memory, entry);
  }
  for (const direction of Object.values(index.directions)) {
    const revision = direction.revisions[String(direction.latestRevision)];
    if (!revision) continue;
    const entry = memoryEntry(
      store,
      "DIRECTION",
      direction.objectId,
      revision,
    );
    classify(memory, entry);
    collectQueryGaps(store, memory, entry);
  }
  for (const turnRef of sortedTurnRefs(store)) {
    const turn = store.readTurn(turnRef);
    if (turn.role !== "DECISION" || !turn.controlRef) continue;
    const control = store.readJson<{
      role?: unknown;
      decision?: unknown;
      guidance?: unknown;
    }>(turn.controlRef);
    if (control.role !== "DECISION" || typeof control.decision !== "string") {
      continue;
    }
    memory.decisionTrail.push({
      turnRef,
      decision: control.decision as LoopDecision,
      guidance:
        typeof control.guidance === "string" ? control.guidance : null,
    });
  }
  for (const experimentRef of store.experimentRefs().sort()) {
    const record = store.readExperiment(experimentRef);
    if (!store.exists(record.resultRef)) continue;
    const result = store.readJson<ExperimentGoalResult>(record.resultRef);
    memory.experimentResults.push({
      resultRef: record.resultRef,
      anchorWork: result.anchorWork,
      directionWork: result.directionWork,
      goalStatus: result.goalStatus,
      experimentObjective: result.experimentObjective,
      conclusionRef: result.conclusionRef,
    });
  }
  return memory;
}

export function writeDecisionObservation(
  store: FileLoopStore,
  state: StateFile,
  allowed: LoopDecision[],
  observationRef: string,
): DecisionObservation {
  const contextMatch = observationRef.match(
    /^(contexts\/[^/]+)\/decision_observation\.json$/,
  );
  if (!contextMatch) {
    throw new Error(
      `Decision observation must be a Context-local path: ${observationRef}`,
    );
  }
  const contextDir = contextMatch[1]!;
  const researchMemoryRef = `${contextDir}/research_memory_snapshot.json`;
  const trajectoryRef = `${contextDir}/progress_trajectory_snapshot.jsonl`;
  const negativeExperimentHistoryRef =
    `${contextDir}/negative_experiment_history_snapshot.json`;

  const memory = buildResearchMemory(store, state);
  if (!store.exists(TRAJECTORY_REF)) store.writeText(TRAJECTORY_REF, "");
  const trajectoryText = store.readText(TRAJECTORY_REF);
  store.writeImmutableJson(researchMemoryRef, memory);
  store.writeImmutableText(trajectoryRef, trajectoryText);
  store.writeImmutableJson(
    negativeExperimentHistoryRef,
    buildNegativeExperimentIndex(store, state),
  );
  const trajectory = store.readJsonLines<ProgressTrajectoryRecord>(trajectoryRef);
  const observation: DecisionObservation = {
    generatedAt: new Date().toISOString(),
    stateRevision: state.revision,
    round: state.round,
    researchMemoryRef,
    trajectoryRef,
    negativeExperimentHistoryRef,
    trajectoryTail: trajectory.slice(-5),
    branchEffects: previewBranchEffects(store, state, allowed),
    accepted: acceptedCounts(store),
    remainingRequirements: computeRemainingRequirements(store, state, true),
    retries: retryCounts(store, state),
    recentRuntimeFailures: sortedTurnRefs(store)
      .map((turnRef) => ({ turnRef, turn: store.readTurn(turnRef) }))
      .filter(({ turn }) =>
        turn.round === state.round && turn.turnState === "RUNTIME_FAILED"
      )
      .map(({ turnRef, turn }) => ({
        turnRef,
        role: turn.role,
        failureKind: turn.runtimeFailureKind,
        outputCapture: turn.outputCapture,
        partialOutputRef: turn.partialOutputRef,
      })),
  };
  store.writeImmutableJson(observationRef, observation);
  // Keep a mutable latest projection for status/checkpoint. It is not an
  // input source for this or any other historical DecisionContext.
  store.writeJson(MEMORY_REF, memory);
  return observation;
}

export function appendDecisionTrajectory(
  store: FileLoopStore,
  cycleState: StateFile,
  resultingState: StateFile,
  decisionTurnRef: string,
  result: DecisionProtocolResult,
): void {
  const existing = store.readJsonLines<ProgressTrajectoryRecord>(TRAJECTORY_REF);
  if (existing.some((item) => item.decisionTurnRef === decisionTurnRef)) return;
  const pending = cycleState.pending;
  let action: TaskAction | null = null;
  if (pending) {
    action = store.readJson<TaskBinding>(pending.workTaskBindingRef).action;
  }
  const record: ProgressTrajectoryRecord = {
    kind: "DECISION_CYCLE",
    round: cycleState.round,
    decisionTurnRef,
    action,
    workRef: pending?.workRef ?? null,
    workOutcome: pending?.workOutcome ?? null,
    reviewRef: pending?.reviewRef ?? null,
    reviewVerdict: pending?.reviewVerdict ?? null,
    decision: result.decision,
    accepted: acceptedCounts(store),
    remainingRequirements: computeRemainingRequirements(
      store,
      resultingState,
      false,
    ),
    retries: retryCounts(store, cycleState),
    usage: roundUsage(store, cycleState.round),
  };
  store.appendJsonLine(TRAJECTORY_REF, record);
  store.appendEvent("PROGRESS_TRAJECTORY_APPENDED", [
    TRAJECTORY_REF,
    decisionTurnRef,
  ]);
}

export function appendTerminalRuntimeTrajectory(
  store: FileLoopStore,
  state: StateFile,
  turnRef: string,
): void {
  const existing = store.readJsonLines<ProgressTrajectoryRecord>(TRAJECTORY_REF);
  if (existing.some((item) => item.terminalTurnRef === turnRef)) return;
  const turn = store.readTurn(turnRef);
  store.appendJsonLine(TRAJECTORY_REF, {
    kind: "TERMINAL_RUNTIME_FAILURE",
    round: state.round,
    decisionTurnRef: null,
    action: turn.taskBindingRef
      ? store.readJson<TaskBinding>(turn.taskBindingRef).action
      : null,
    workRef: null,
    workOutcome: null,
    reviewRef: null,
    reviewVerdict: null,
    decision: null,
    accepted: acceptedCounts(store),
    remainingRequirements: computeRemainingRequirements(store, state, false),
    retries: retryCounts(store, state),
    usage: roundUsage(store, state.round),
    terminalTurnRef: turnRef,
    runtimeFailureKind: turn.runtimeFailureKind,
    outputCapture: turn.outputCapture,
    partialOutputRef: turn.partialOutputRef,
  } satisfies ProgressTrajectoryRecord);
  store.appendEvent("TERMINAL_TRAJECTORY_APPENDED", [TRAJECTORY_REF, turnRef]);
}

export function writeCheckpoint(
  store: FileLoopStore,
  state: StateFile,
  reason: string,
): string {
  const goal = store.readJson<WorkflowGoal>("workflow_goal.json");
  const memory = rebuildResearchMemory(store, state);
  const tail = store
    .readJsonLines<ProgressTrajectoryRecord>(TRAJECTORY_REF)
    .slice(-5);
  const ref = `observations/checkpoints/round-${state.round}.md`;
  const lines = [
    `# Learning Loop Checkpoint — Round ${state.round}`,
    "",
    `- Lifecycle: ${state.lifecycle}`,
    `- Reason: ${reason}`,
    `- Pause kind: ${state.pauseKind ?? "None"}`,
    `- Authorized through round: ${state.roundBudget?.authorizedThroughRound ?? "legacy"}`,
    `- Topic: ${goal.topic}`,
    `- Objective: ${goal.objective}`,
    `- State revision: ${state.revision}`,
    "",
    "## Accepted objects",
    "",
    ...entryLines(memory.accepted),
    "",
    "## Needs revision",
    "",
    ...entryLines(memory.needsRevision),
    "",
    "## Rejected lessons",
    "",
    ...entryLines(memory.rejectedLessons),
    "",
    "## Dynamic 6L coverage",
    "",
    ...Object.entries(memory.coverage).map(
      ([layer, entries]) =>
        `- ${layer}: ${entries.map((entry) => entry.objectId).join(", ") || "—"}`,
    ),
    "",
    "## Open query gaps",
    "",
    ...(memory.openQueryGaps.length > 0
      ? memory.openQueryGaps.map((entry) =>
        `- ${entry.objectKind} ${entry.objectId} ` +
        `(review: \`${entry.reviewRef}\`): ${JSON.stringify(entry.gap)}`
      )
      : ["- None"]),
    "",
    "## EXP Goal results",
    "",
    ...(memory.experimentResults.length > 0
      ? memory.experimentResults.map((entry) =>
        `- ${entry.goalStatus}: ${entry.experimentObjective} ` +
        `(result: \`${entry.resultRef}\`${
          entry.conclusionRef ? `; conclusion: \`${entry.conclusionRef}\`` : ""
        })`
      )
      : ["- None"]),
    "",
    "## Reviewed negative EXP evidence",
    "",
    ...negativeExperimentLines(store, state),
    "",
    "## Runtime transport failures",
    "",
    ...runtimeFailureLines(store),
    "",
    "## Latest Decision guidance",
    "",
    `- ${state.decisionGuidance ?? "None"}`,
    "",
    "## Mechanical requirements",
    "",
    ...(memory.requirements.length > 0
      ? memory.requirements.map((item) => `- ${item}`)
      : ["- None"]),
    "",
    "## Recent trajectory",
    "",
    "```json",
    JSON.stringify(tail, null, 2),
    "```",
  ];
  store.writeText(ref, `${lines.join("\n")}\n`);
  store.appendEvent("CHECKPOINT_WRITTEN", [ref, MEMORY_REF, TRAJECTORY_REF]);
  return ref;
}

export function readObservationSummary(store: FileLoopStore): unknown {
  return {
    researchMemory: store.exists(MEMORY_REF)
      ? store.readJson<ResearchMemory>(MEMORY_REF)
      : null,
    trajectoryTail: store
      .readJsonLines<ProgressTrajectoryRecord>(TRAJECTORY_REF)
      .slice(-5),
    negativeExperimentIndex: store.exists(
        "observations/negative_experiment_index.json"
      )
      ? store.readJson("observations/negative_experiment_index.json")
      : null,
  };
}

function memoryEntry(
  store: FileLoopStore,
  objectKind: "ANCHOR" | "DIRECTION",
  objectId: string,
  revision: {
    revision: number;
    workRef: string;
    workOutcome: ResearchMemoryEntry["workOutcome"];
    reviewRef: string;
    reviewVerdict: ResearchMemoryEntry["reviewVerdict"];
  },
): ResearchMemoryEntry {
  const work = store.readJson<WorkResult>(revision.workRef);
  const review = store.readJson<ReviewResult>(revision.reviewRef);
  const content = asRecord(work.content);
  const name = stringValue(content?.name);
  const summary = stringValue(review.summary);
  return {
    objectKind,
    objectId,
    revision: revision.revision,
    workRef: revision.workRef,
    reviewRef: revision.reviewRef,
    workOutcome: revision.workOutcome,
    reviewVerdict: revision.reviewVerdict,
    name,
    summary,
    summaryAvailable: name !== null || summary !== null,
  };
}

function classify(memory: ResearchMemory, entry: ResearchMemoryEntry): void {
  if (
    entry.reviewVerdict === "PASS" &&
    entry.workOutcome === "READY_FOR_REVIEW"
  ) {
    memory.accepted.push(entry);
  } else if (entry.reviewVerdict === "REJECT") {
    memory.rejectedLessons.push(entry);
  } else {
    memory.needsRevision.push(entry);
  }
}

function collectQueryGaps(
  store: FileLoopStore,
  memory: ResearchMemory,
  entry: ResearchMemoryEntry,
): void {
  const review = store.readJson<ReviewResult>(entry.reviewRef);
  if (!Array.isArray(review.queryGaps)) return;
  for (const gap of review.queryGaps) {
    memory.openQueryGaps.push({
      objectKind: entry.objectKind,
      objectId: entry.objectId,
      reviewRef: entry.reviewRef,
      gap,
    });
  }
}

function collectCoverage(
  store: FileLoopStore,
  memory: ResearchMemory,
  entry: ResearchMemoryEntry,
): void {
  const work = store.readJson<WorkResult>(entry.workRef);
  const scope = asRecord(asRecord(work.content)?.scope6L);
  if (!scope) return;
  for (const layer of ["L1", "L2", "L3", "L4", "L5", "L6"] as const) {
    const value = stringValue(scope[layer]);
    if (value) {
      memory.coverage[layer].push({
        objectId: entry.objectId,
        workRef: entry.workRef,
        value,
      });
    }
  }
}

function acceptedCounts(store: FileLoopStore): {
  anchors: number;
  directions: number;
} {
  const index = store.readObjects();
  let anchors = 0;
  let directions = 0;
  for (const anchorId of index.activeAnchorIds) {
    const anchor = index.anchors[anchorId];
    if (!anchor || anchor.rejected) continue;
    const revision = anchor.revisions[String(anchor.latestRevision)];
    if (
      revision?.workOutcome === "READY_FOR_REVIEW" &&
      revision.reviewVerdict === "PASS"
    ) anchors += 1;
  }
  for (const direction of Object.values(index.directions)) {
    if (direction.rejected) continue;
    const revision = direction.revisions[String(direction.latestRevision)];
    if (
      revision?.workOutcome === "READY_FOR_REVIEW" &&
      revision.reviewVerdict === "PASS"
    ) directions += 1;
  }
  return { anchors, directions };
}

function retryCounts(store: FileLoopStore, state: StateFile) {
  const turns = sortedTurnRefs(store).map((ref) => store.readTurn(ref));
  return {
    outputCorrection: turns.filter((turn) => turn.turnState === "INVALID_OUTPUT")
      .length,
    semantic: state.semanticRetries.worker + state.semanticRetries.reviewer,
    runtime: turns.filter((turn) => turn.turnState === "RUNTIME_FAILED").length,
  };
}

function roundUsage(
  store: FileLoopStore,
  round: number,
): TokenUsage & { elapsedMs: number } {
  const total = { ...zeroUsage(), elapsedMs: 0 };
  for (const turnRef of sortedTurnRefs(store)) {
    const turn = store.readTurn(turnRef);
    if (turn.round !== round || !store.exists(turn.runtimeRef)) continue;
    const summaries = store.readJsonLines<{
      type?: unknown;
      usage?: Partial<TokenUsage>;
      elapsedMs?: unknown;
    }>(turn.runtimeRef).filter((event) => event.type === "provider_summary");
    const summary = summaries.at(-1);
    if (!summary) continue;
    total.inputTokens += Number(summary.usage?.inputTokens ?? 0);
    total.cachedInputTokens += Number(summary.usage?.cachedInputTokens ?? 0);
    total.outputTokens += Number(summary.usage?.outputTokens ?? 0);
    total.reasoningOutputTokens += Number(
      summary.usage?.reasoningOutputTokens ?? 0,
    );
    total.totalTokens += Number(summary.usage?.totalTokens ?? 0);
    total.elapsedMs += Number(summary.elapsedMs ?? 0);
  }
  return total;
}

function sortedTurnRefs(store: FileLoopStore): string[] {
  return store.turnRefs().sort((left, right) =>
    store.readTurn(left).startedAt.localeCompare(store.readTurn(right).startedAt)
  );
}

function entryLines(entries: ResearchMemoryEntry[]): string[] {
  return entries.length > 0
    ? entries.map((entry) =>
      `- ${entry.objectKind} ${entry.objectId}@${entry.revision}: ` +
      `${entry.name ?? "summary unavailable"} ` +
      `(work: \`${entry.workRef}\`, review: \`${entry.reviewRef}\`)`
    )
    : ["- None"];
}

function runtimeFailureLines(store: FileLoopStore): string[] {
  const failures = sortedTurnRefs(store)
    .map((ref) => ({ ref, turn: store.readTurn(ref) }))
    .filter(({ turn }) => turn.turnState === "RUNTIME_FAILED")
    .slice(-5);
  return failures.length > 0
    ? failures.map(({ ref, turn }) =>
      `- ${turn.role} ${turn.runtimeFailureKind ?? "PROVIDER_ERROR"}, ` +
      `capture=${turn.outputCapture}, turn=\`${ref}\`, ` +
      `runtime=\`${turn.runtimeRef}\`, ` +
      `partial=${turn.partialOutputRef ? `\`${turn.partialOutputRef}\`` : "—"}`
    )
    : ["- None"];
}

function negativeExperimentLines(
  store: FileLoopStore,
  state: StateFile,
): string[] {
  const index = buildNegativeExperimentIndex(store, state);
  return index.entries.length > 0
    ? index.entries.map((entry) =>
      `- Anchor \`${entry.anchorWork}\`; Direction ${
        entry.directionWork ? `\`${entry.directionWork}\`` : "None"
      }; EXP \`${entry.experimentResultRef}\`; review \`${entry.reviewRef}\``
    )
    : ["- None"];
}


function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function zeroUsage(): TokenUsage {
  return {
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningOutputTokens: 0,
    totalTokens: 0,
  };
}
