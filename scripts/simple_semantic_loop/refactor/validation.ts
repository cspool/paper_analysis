import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  validateCoreOutputForAction,
  validateReferenceTemplateForAction,
} from "./schemas.ts";
import {
  validatePublishedReferenceTemplates,
} from "./run_setup.ts";
import type { CodexAppServerRuntime } from "./runtime.ts";
import { FileLoopStore } from "./store.ts";
import {
  CURRENT_FORMAT_VERSION,
  PAUSE_KINDS,
  READABLE_FORMAT_VERSIONS,
  TASK_ACTIONS,
  TURN_STATES,
  OUTPUT_CAPTURE_STATES,
  type CoreControlProjection,
  type CommittedResult,
  type DecisionContext,
  type DecisionObservation,
  type ObjectsIndex,
  type OutputErrorReport,
  type PendingResults,
  type RefCatalog,
  type ReviewResult,
  type RunFile,
  type ProgressTrajectoryRecord,
  type ResearchMemory,
  type RoundAuthorizationRecord,
  type RunFormatVersion,
  type RuntimeRecoveryRecord,
  type StateFile,
  type TaskAction,
  type TaskBinding,
  type TurnFile,
  type TurnTask,
  type WorkflowGoal,
} from "./types.ts";
import { computeRemainingRequirements } from "./workflow.ts";

export interface ValidationReport {
  valid: boolean;
  checks: Array<{
    name: string;
    passed: boolean;
    details: string[];
  }>;
  advisories: Array<{
    name: string;
    details: string[];
  }>;
}

const TASK_INPUT_RULES: Record<
  TaskAction,
  {
    required: Array<keyof TurnTask["inputs"]>;
    allowed: Array<keyof TurnTask["inputs"]>;
  }
> = {
  CREATE_ANCHOR: {
    required: [],
    allowed: ["currentWork", "latestReview", "researchMemory"],
  },
  DEEPEN_ANCHOR: {
    required: ["currentWork", "latestReview"],
    allowed: ["currentWork", "latestReview"],
  },
  CREATE_DIRECTION: {
    required: ["boundAnchor"],
    allowed: ["boundAnchor", "currentWork", "latestReview"],
  },
  DEEPEN_DIRECTION: {
    required: ["boundAnchor", "currentWork", "latestReview"],
    allowed: ["boundAnchor", "currentWork", "latestReview"],
  },
  REVIEW_ANCHOR: {
    required: ["reviewTarget"],
    allowed: ["reviewTarget", "previousReview"],
  },
  REVIEW_DIRECTION: {
    required: ["boundAnchor", "reviewTarget"],
    allowed: ["boundAnchor", "reviewTarget", "previousReview"],
  },
};

export function validateRun(workDir: string): ValidationReport {
  const store = new FileLoopStore(workDir);
  const checks: ValidationReport["checks"] = [];
  const advisories: ValidationReport["advisories"] = [];
  const add = (name: string, details: string[]) =>
    checks.push({ name, passed: details.length === 0, details });
  const advise = (name: string, details: string[]) => {
    if (details.length > 0) advisories.push({ name, details });
  };

  let run: RunFile;
  let state: StateFile;
  let goal: WorkflowGoal;
  let index: ObjectsIndex;
  try {
    run = store.readRun();
    state = store.readState();
    goal = store.readJson<WorkflowGoal>("workflow_goal.json");
    index = store.readObjects();
    add("core-json", []);
  } catch (error) {
    add("core-json", [message(error)]);
    return { valid: false, checks, advisories };
  }

  add("format-version", [
    ...(READABLE_FORMAT_VERSIONS.includes(run.formatVersion)
      ? []
      : [`run format=${run.formatVersion}`]),
    ...(READABLE_FORMAT_VERSIONS.includes(state.formatVersion)
      ? []
      : [`state format=${state.formatVersion}`]),
    ...(run.formatVersion === state.formatVersion
      ? []
      : [
        `run/state format mismatch=${run.formatVersion}/${state.formatVersion}`,
      ]),
  ]);
  add("workflow-goal", validateWorkflowGoal(goal));
  add("event-sequence", validateEventSequence(store));
  add("pinned-skills-and-refs", validatePins(store, run));
  advise(
    "published-reference-templates",
    validatePublishedReferenceTemplates(run.projectRoot),
  );
  add("task-bindings", validateBindings(store));
  add(
    "decision-contexts",
    validateDecisionContexts(store, run.formatVersion),
  );
  add("observations", validateObservations(store));
  add("recoveries", validateRecoveries(store, run.formatVersion));
  add(
    "round-authorizations",
    validateRoundAuthorizations(store, run.formatVersion, state),
  );
  const turnValidation = validateTurns(store);
  add("turn-records", turnValidation.errors);
  advise("result-reference-template", turnValidation.advisories);
  add("object-index", validateObjectIndex(store, index));
  add("canonical-state", validateState(store, state));
  add("final-artifacts", validateFinalArtifacts(store, state));

  return {
    valid: checks.every((check) => check.passed),
    checks,
    advisories,
  };
}

export async function runDoctor(
  projectRoot: string,
  runtime: CodexAppServerRuntime | null,
  model: string,
): Promise<ValidationReport> {
  const checks: ValidationReport["checks"] = [];
  const advisories: ValidationReport["advisories"] = [];
  const major = Number(process.versions.node.split(".")[0]);
  checks.push({
    name: "node",
    passed: major >= 22,
    details: major >= 22 ? [process.version] : [`requires Node >=22, got ${process.version}`],
  });
  const templateErrors = validatePublishedReferenceTemplates(projectRoot);
  if (templateErrors.length > 0) {
    advisories.push({
      name: "published-reference-templates",
      details: templateErrors,
    });
  }

  const skillErrors: string[] = [];
  for (const name of [
    "learning-loop-decision",
    "learning-loop-worker",
    "learning-loop-reviewer",
  ]) {
    const path = resolve(projectRoot, `.codex/skills/${name}/SKILL.md`);
    if (!existsSync(path)) {
      skillErrors.push(`missing ${path}`);
      continue;
    }
    const source = readFileSync(path, "utf8");
    if (!source.startsWith("---\n") || !source.includes(`name: ${name}`)) {
      skillErrors.push(`${name}: invalid frontmatter`);
    }
  }
  checks.push({
    name: "skills",
    passed: skillErrors.length === 0,
    details: skillErrors,
  });

  if (runtime) {
    try {
      const efforts = await runtime.probeModel(model);
      const missing = ["high", "max"].filter(
        (effort) => !efforts.includes(effort),
      );
      checks.push({
        name: "provider-model-effort",
        passed: missing.length === 0,
        details:
          missing.length === 0
            ? [`${model}: high,max`]
            : [`${model} missing ${missing.join(", ")}`],
      });
      const skills = await runtime.probeSkills(projectRoot);
      const missingSkills = [
        "learning-loop-decision",
        "learning-loop-worker",
        "learning-loop-reviewer",
      ].filter((name) => !skills.has(name));
      checks.push({
        name: "provider-skills",
        passed: missingSkills.length === 0,
        details: missingSkills,
      });
    } catch (error) {
      checks.push({
        name: "provider",
        passed: false,
        details: [message(error)],
      });
    } finally {
      await runtime.close();
    }
  }
  return {
    valid: checks.every((check) => check.passed),
    checks,
    advisories,
  };
}

function validateWorkflowGoal(goal: WorkflowGoal): string[] {
  const errors = exactKeys(
    goal,
    ["topic", "objective", "acceptanceCriteria"],
    "workflow_goal.json",
  );
  if (!nonEmpty(goal.topic)) errors.push("workflow_goal.topic is empty");
  if (!nonEmpty(goal.objective)) errors.push("workflow_goal.objective is empty");
  if (
    !Array.isArray(goal.acceptanceCriteria) ||
    goal.acceptanceCriteria.length === 0 ||
    goal.acceptanceCriteria.some((value) => !nonEmpty(value))
  ) {
    errors.push("workflow_goal.acceptanceCriteria must contain non-empty strings");
  }
  return errors;
}

function validateEventSequence(store: FileLoopStore): string[] {
  if (!store.exists("events.jsonl")) return ["events.jsonl is missing"];
  const errors: string[] = [];
  const lines = store.readText("events.jsonl").split("\n").filter(Boolean);
  lines.forEach((line, index) => {
    try {
      const event = JSON.parse(line) as { seq?: number };
      if (event.seq !== index + 1) {
        errors.push(`event ${index + 1} has seq ${String(event.seq)}`);
      }
    } catch (error) {
      errors.push(`event ${index + 1}: ${message(error)}`);
    }
  });
  return errors;
}

function validatePins(store: FileLoopStore, run: RunFile): string[] {
  const errors: string[] = [];
  let catalog: RefCatalog;
  try {
    catalog = store.readJson<RefCatalog>("ref_catalog.json");
  } catch (error) {
    return [message(error)];
  }
  for (const [name, entry] of Object.entries(catalog)) {
    const path = resolve(run.projectRoot, entry.path);
    if (!existsSync(path)) {
      errors.push(`${name}: missing ${path}`);
      continue;
    }
    if (store.sha256ExternalFile(path) !== entry.sha256) {
      errors.push(`${name}: sha256 drift`);
    }
  }
  for (const [role, skill] of Object.entries(run.skills)) {
    const path = resolve(run.projectRoot, skill.path);
    if (!existsSync(path)) errors.push(`${role}: missing Skill ${path}`);
    else if (store.sha256ExternalFile(path) !== skill.sha256) {
      errors.push(`${role}: Skill sha256 drift`);
    }
  }
  return errors;
}

function validateBindings(store: FileLoopStore): string[] {
  const errors: string[] = [];
  for (const ref of jsonFiles(store, "bindings")) {
    let binding: TaskBinding;
    let task: TurnTask;
    try {
      binding = store.readJson<TaskBinding>(ref);
      task = store.readJson<TurnTask>(binding.taskRef);
    } catch (error) {
      errors.push(`${ref}: ${message(error)}`);
      continue;
    }
    if (binding.action !== task.action) {
      errors.push(`${ref}: binding action differs from T01`);
    }
    const expectedRole = task.action.startsWith("REVIEW_")
      ? "REVIEWER"
      : "WORKER";
    if (binding.role !== expectedRole) {
      errors.push(`${ref}: ${task.action} must bind ${expectedRole}`);
    }
    const expectedRef = task.action.includes("ANCHOR")
      ? task.action.startsWith("REVIEW_")
        ? "review-result-v2"
        : "work-result-anchor-v2"
      : task.action.startsWith("REVIEW_")
      ? "review-result-v2"
      : "work-result-direction-v2";
    if (binding.resultRefName !== expectedRef) {
      errors.push(`${ref}: expected ${expectedRef}`);
    }
    errors.push(...validateTurnTask(task).map((item) => `${ref}: ${item}`));
    for (const [name, inputRef] of Object.entries(task.inputs)) {
      if (!nonEmpty(inputRef)) {
        errors.push(`${ref}: inputs.${name} must be a non-empty path`);
      } else if (!store.exists(inputRef)) {
        errors.push(`${ref}: missing input ${inputRef}`);
      }
    }
  }
  return errors;
}

function validateTurnTask(task: TurnTask): string[] {
  const errors = exactKeys(
    task,
    ["goalRef", "action", "objective", "inputs", "requirements", "constraints"],
    "T01",
  );
  if (task.goalRef !== "workflow_goal.json") {
    errors.push("T01 goalRef must be workflow_goal.json");
  }
  if (!TASK_ACTIONS.includes(task.action)) errors.push("unknown T01 action");
  errors.push(
    ...exactKeys(
      task.inputs,
      [
        "boundAnchor",
        "currentWork",
        "latestReview",
        "reviewTarget",
        "previousReview",
        "researchMemory",
      ],
      "T01.inputs",
      true,
    ),
  );
  if (!nonEmpty(task.objective)) errors.push("T01 objective is empty");
  if (!nonEmptyStringList(task.requirements)) {
    errors.push("T01 requirements must contain non-empty strings");
  }
  if (!nonEmptyStringList(task.constraints)) {
    errors.push("T01 constraints must contain non-empty strings");
  }
  if (TASK_ACTIONS.includes(task.action)) {
    const rule = TASK_INPUT_RULES[task.action];
    const actual = Object.keys(task.inputs) as Array<keyof TurnTask["inputs"]>;
    for (const key of actual) {
      if (!rule.allowed.includes(key)) {
        errors.push(`${task.action} does not allow inputs.${key}`);
      }
    }
    for (const key of rule.required) {
      if (!task.inputs[key]) {
        errors.push(`${task.action} requires inputs.${key}`);
      }
    }
    const hasCurrentWork = nonEmpty(task.inputs.currentWork);
    const hasLatestReview = nonEmpty(task.inputs.latestReview);
    if (
      (task.action === "CREATE_ANCHOR" ||
        task.action === "CREATE_DIRECTION") &&
      hasCurrentWork !== hasLatestReview
    ) {
      errors.push(
        `${task.action} replacement context requires both inputs.currentWork and inputs.latestReview`,
      );
    }
  }
  return errors;
}

function validateDecisionContexts(
  store: FileLoopStore,
  formatVersion: RunFormatVersion,
): string[] {
  const errors: string[] = [];
  if (formatVersion >= 6) {
    const contextsRoot = store.absolute("contexts");
    if (existsSync(contextsRoot)) {
      for (
        const entry of readdirSync(contextsRoot, { withFileTypes: true })
      ) {
        if (!entry.isDirectory()) continue;
        const contextDir = `contexts/${entry.name}`;
        for (const name of [
          "decision_context.json",
          "decision_observation.json",
          "research_memory_snapshot.json",
          "progress_trajectory_snapshot.jsonl",
        ]) {
          const expected = `${contextDir}/${name}`;
          if (!store.exists(expected)) {
            errors.push(`${contextDir}: missing frozen Context file ${name}`);
          }
        }
      }
    }
  }
  for (const ref of jsonFiles(store, "contexts")) {
    if (!ref.endsWith("/decision_context.json")) continue;
    let context: DecisionContext;
    try {
      context = store.readJson<DecisionContext>(ref);
    } catch (error) {
      errors.push(`${ref}: ${message(error)}`);
      continue;
    }
    errors.push(
      ...exactKeys(
        context,
        [
          "goalRef",
          "committedResults",
          "pendingResults",
          "remainingRequirementsAfterPendingCommit",
          "observationRef",
        ],
        ref,
      ),
    );
    if (context.goalRef !== "workflow_goal.json") {
      errors.push(`${ref}: invalid goalRef`);
    }
    if (!nonEmpty(context.observationRef) || !store.exists(context.observationRef)) {
      errors.push(`${ref}: missing observationRef ${String(context.observationRef)}`);
    } else {
      try {
        const expectedObservationRef = ref.replace(
          /decision_context\.json$/,
          "decision_observation.json",
        );
        if (context.observationRef !== expectedObservationRef) {
          errors.push(`${ref}: observationRef is not the frozen sibling snapshot`);
        }
        const observation = store.readJson<DecisionObservation>(
          context.observationRef,
        );
        if (!Number.isInteger(observation.stateRevision)) {
          errors.push(`${ref}: observation stateRevision is invalid`);
        }
        for (const sourceRef of [observation.researchMemoryRef, observation.trajectoryRef]) {
          if (!store.exists(sourceRef)) {
            errors.push(`${ref}: observation source missing ${sourceRef}`);
          }
        }
        if (formatVersion >= 6) {
          errors.push(
            ...validateFrozenDecisionSources(
              store,
              ref,
              observation,
            ),
          );
        }
        for (const effect of observation.branchEffects ?? []) {
          if (effect.targetRef && !store.exists(effect.targetRef)) {
            errors.push(`${ref}: branch effect target missing ${effect.targetRef}`);
          }
        }
        for (const failure of observation.recentRuntimeFailures ?? []) {
          if (!store.exists(failure.turnRef)) {
            errors.push(`${ref}: runtime failure Turn missing ${failure.turnRef}`);
          }
          if (failure.partialOutputRef && !store.exists(failure.partialOutputRef)) {
            errors.push(`${ref}: runtime partial missing ${failure.partialOutputRef}`);
          }
        }
      } catch (error) {
        errors.push(`${ref}: invalid observation: ${message(error)}`);
      }
    }
    for (const result of context.committedResults) {
      if (!["ANCHOR", "DIRECTION"].includes(result.objectKind)) {
        errors.push(`${ref}: invalid committed objectKind ${result.objectKind}`);
        continue;
      }
      const keys = result.objectKind === "ANCHOR"
        ? ["objectKind", "work", "review"]
        : ["objectKind", "anchorWork", "work", "review"];
      errors.push(...exactKeys(result, keys, `${ref}.committedResults`));
      for (const path of resultPaths(result)) {
        if (!store.exists(path)) {
          errors.push(`${ref}: missing result ${path}`);
        }
      }
    }
    if (context.pendingResults) {
      if (!["ANCHOR", "DIRECTION"].includes(context.pendingResults.objectKind)) {
        errors.push(
          `${ref}: invalid pending objectKind ${context.pendingResults.objectKind}`,
        );
        continue;
      }
      const keys = context.pendingResults.objectKind === "ANCHOR"
        ? ["objectKind", "workTask", "work", "review"]
        : ["objectKind", "anchorWork", "workTask", "work", "review"];
      errors.push(...exactKeys(context.pendingResults, keys, `${ref}.pendingResults`));
      for (const path of resultPaths(context.pendingResults)) {
        if (!store.exists(path)) {
          errors.push(`${ref}: missing pending result ${path}`);
        }
      }
    }
    for (const requirement of context.remainingRequirementsAfterPendingCommit) {
      if (!validRequirement(requirement)) {
        errors.push(`${ref}: invalid requirement ${requirement}`);
        continue;
      }
      const requirementRef = requirement.includes(":")
        ? requirement.slice(requirement.indexOf(":") + 1)
        : null;
      if (requirementRef && !store.exists(requirementRef)) {
        errors.push(`${ref}: requirement references missing ${requirementRef}`);
      }
    }
  }
  return errors;
}

function validateFrozenDecisionSources(
  store: FileLoopStore,
  contextRef: string,
  observation: DecisionObservation,
): string[] {
  const errors: string[] = [];
  const contextDir = contextRef.replace(/\/decision_context\.json$/, "");
  const expectedMemoryRef = `${contextDir}/research_memory_snapshot.json`;
  const expectedTrajectoryRef =
    `${contextDir}/progress_trajectory_snapshot.jsonl`;
  if (observation.researchMemoryRef !== expectedMemoryRef) {
    errors.push(
      `${contextRef}: researchMemoryRef must be ${expectedMemoryRef}`,
    );
  }
  if (observation.trajectoryRef !== expectedTrajectoryRef) {
    errors.push(
      `${contextRef}: trajectoryRef must be ${expectedTrajectoryRef}`,
    );
  }

  if (store.exists(expectedMemoryRef)) {
    try {
      const memory = store.readJson<ResearchMemory>(expectedMemoryRef);
      if (memory.sourceStateRevision !== observation.stateRevision) {
        errors.push(
          `${contextRef}: memory sourceStateRevision ${memory.sourceStateRevision} differs from observation ${observation.stateRevision}`,
        );
      }
    } catch (error) {
      errors.push(`${contextRef}: invalid memory snapshot: ${message(error)}`);
    }
  }

  if (store.exists(expectedTrajectoryRef)) {
    try {
      const snapshot = store.readJsonLines<ProgressTrajectoryRecord>(
        expectedTrajectoryRef,
      );
      const expectedTail = snapshot.slice(-5);
      if (JSON.stringify(observation.trajectoryTail) !== JSON.stringify(expectedTail)) {
        errors.push(
          `${contextRef}: trajectoryTail differs from frozen trajectory suffix`,
        );
      }

      const globalRef = "observations/progress_trajectory.jsonl";
      if (store.exists(globalRef)) {
        const global = store.readJsonLines<ProgressTrajectoryRecord>(globalRef);
        const prefix = global.slice(0, snapshot.length);
        if (JSON.stringify(prefix) !== JSON.stringify(snapshot)) {
          errors.push(
            `${contextRef}: frozen trajectory is not a prefix of the global trajectory`,
          );
        }
      }

      for (const record of snapshot) {
        if (!record.decisionTurnRef || !store.exists(record.decisionTurnRef)) {
          continue;
        }
        const turn = store.readTurn(record.decisionTurnRef);
        if (turn.decisionContextRef === contextRef) {
          errors.push(
            `${contextRef}: trajectory snapshot contains its own not-yet-made Decision`,
          );
        }
      }
    } catch (error) {
      errors.push(
        `${contextRef}: invalid trajectory snapshot: ${message(error)}`,
      );
    }
  }
  return errors;
}

function validateObservations(store: FileLoopStore): string[] {
  const errors: string[] = [];
  const trajectoryRef = "observations/progress_trajectory.jsonl";
  if (!store.exists(trajectoryRef)) {
    errors.push(`missing ${trajectoryRef}`);
    return errors;
  }
  let records: ProgressTrajectoryRecord[];
  try {
    records = store.readJsonLines<ProgressTrajectoryRecord>(trajectoryRef);
  } catch (error) {
    return [`${trajectoryRef}: ${message(error)}`];
  }
  const decisionRefs = new Set<string>();
  const terminalRefs = new Set<string>();
  for (const [index, record] of records.entries()) {
    const label = `${trajectoryRef}:${index + 1}`;
    if (record.kind === "DECISION_CYCLE") {
      if (!record.decisionTurnRef || !store.exists(record.decisionTurnRef)) {
        errors.push(`${label}: missing decisionTurnRef`);
      } else if (decisionRefs.has(record.decisionTurnRef)) {
        errors.push(`${label}: duplicate decisionTurnRef ${record.decisionTurnRef}`);
      } else {
        decisionRefs.add(record.decisionTurnRef);
      }
    } else if (record.kind === "TERMINAL_RUNTIME_FAILURE") {
      if (!record.terminalTurnRef || !store.exists(record.terminalTurnRef)) {
        errors.push(`${label}: missing terminalTurnRef`);
      } else if (terminalRefs.has(record.terminalTurnRef)) {
        errors.push(`${label}: duplicate terminalTurnRef ${record.terminalTurnRef}`);
      } else {
        terminalRefs.add(record.terminalTurnRef);
      }
    } else {
      errors.push(`${label}: invalid kind ${String(record.kind)}`);
    }
    for (const resultRef of [record.workRef, record.reviewRef]) {
      if (resultRef && !store.exists(resultRef)) {
        errors.push(`${label}: missing result ${resultRef}`);
      }
    }
    if (record.partialOutputRef && !store.exists(record.partialOutputRef)) {
      errors.push(`${label}: missing partial ${record.partialOutputRef}`);
    }
  }
  return errors;
}

function validateRecoveries(
  store: FileLoopStore,
  formatVersion: RunFormatVersion,
): string[] {
  const errors: string[] = [];
  const tokens = new Set<string>();
  for (const ref of jsonFiles(store, "recoveries")) {
    let record: RuntimeRecoveryRecord;
    try {
      record = store.readJson<RuntimeRecoveryRecord>(ref);
    } catch (error) {
      errors.push(`${ref}: ${message(error)}`);
      continue;
    }
    if (record.formatVersion !== formatVersion) {
      errors.push(
        `${ref}: formatVersion ${record.formatVersion} differs from run ${formatVersion}`,
      );
    }
    if (!nonEmpty(record.tokenHash)) errors.push(`${ref}: tokenHash is empty`);
    else if (tokens.has(record.tokenHash)) {
      errors.push(`${ref}: duplicate recovery tokenHash`);
    } else tokens.add(record.tokenHash);
    for (const source of [record.sourceTurnRef, record.recoveryRef]) {
      if (!store.exists(source)) errors.push(`${ref}: missing ${source}`);
    }
    if (record.recoveryRef !== ref) {
      errors.push(`${ref}: recoveryRef does not identify its record`);
    }
    if (!record.sourceTurnRef.startsWith("turns/")) {
      errors.push(`${ref}: invalid sourceTurnRef`);
    }
  }
  return errors;
}

function validateRoundAuthorizations(
  store: FileLoopStore,
  formatVersion: RunFormatVersion,
  state: StateFile,
): string[] {
  if (formatVersion < 7) return [];
  const errors: string[] = [];
  const records = new Map<string, RoundAuthorizationRecord>();
  for (const ref of jsonFiles(store, "authorizations/rounds")) {
    let record: RoundAuthorizationRecord;
    try {
      record = store.readJson<RoundAuthorizationRecord>(ref);
    } catch (error) {
      errors.push(`${ref}: ${message(error)}`);
      continue;
    }
    if (record.formatVersion !== formatVersion) {
      errors.push(
        `${ref}: formatVersion ${record.formatVersion} differs from run ${formatVersion}`,
      );
    }
    if (!nonEmpty(record.authorizationId)) {
      errors.push(`${ref}: authorizationId is empty`);
    }
    const expectedRef =
      `authorizations/rounds/${record.authorizationId}.json`;
    if (ref !== expectedRef) {
      errors.push(`${ref}: authorizationId does not identify its record`);
    }
    if (
      !Number.isInteger(record.sourceStateRevision) ||
      record.sourceStateRevision < 1
    ) {
      errors.push(`${ref}: sourceStateRevision is invalid`);
    }
    if (!Number.isInteger(record.additionalRounds) || record.additionalRounds < 1) {
      errors.push(`${ref}: additionalRounds is invalid`);
    }
    if (
      !Number.isInteger(record.firstAuthorizedRound) ||
      record.firstAuthorizedRound < 1
    ) {
      errors.push(`${ref}: firstAuthorizedRound is invalid`);
    }
    if (
      record.authorizedThroughRound !==
        record.firstAuthorizedRound + record.additionalRounds - 1
    ) {
      errors.push(`${ref}: authorizedThroughRound does not match its grant`);
    }
    records.set(ref, record);
  }

  const lastRef = state.roundBudget?.lastAuthorizationRef ?? null;
  if (lastRef !== null) {
    const last = records.get(lastRef);
    if (!last) {
      errors.push(`state roundBudget references missing ${lastRef}`);
    } else if (
      last.authorizedThroughRound !== state.roundBudget?.authorizedThroughRound
    ) {
      errors.push(
        "state roundBudget authorizedThroughRound differs from its last authorization",
      );
    }
  }
  return errors;
}

function validateTurns(
  store: FileLoopStore,
): { errors: string[]; advisories: string[] } {
  const errors: string[] = [];
  const advisories: string[] = [];
  for (const ref of turnRefs(store)) {
    let turn: TurnFile;
    try {
      turn = store.readTurn(ref);
    } catch (error) {
      errors.push(`${ref}: ${message(error)}`);
      continue;
    }
    if (!TURN_STATES.includes(turn.turnState)) {
      errors.push(`${ref}: invalid turnState ${turn.turnState}`);
    }
    if (!OUTPUT_CAPTURE_STATES.includes(turn.outputCapture)) {
      errors.push(`${ref}: invalid outputCapture ${String(turn.outputCapture)}`);
    }
    for (const required of [
      turn.promptRef,
      turn.validationAuditRef,
      turn.runtimeRef,
    ]) {
      if (!store.exists(required)) errors.push(`${ref}: missing ${required}`);
    }
    if (turn.rawOutputRef && !store.exists(turn.rawOutputRef)) {
      errors.push(`${ref}: missing ${turn.rawOutputRef}`);
    }
    if (turn.partialOutputRef && !store.exists(turn.partialOutputRef)) {
      errors.push(`${ref}: missing ${turn.partialOutputRef}`);
    }
    if (turn.runtimeErrorRef && !store.exists(turn.runtimeErrorRef)) {
      errors.push(`${ref}: missing ${turn.runtimeErrorRef}`);
    }
    if (turn.rawOutputRef && turn.outputCapture !== "COMPLETE") {
      errors.push(`${ref}: rawOutputRef requires outputCapture=COMPLETE`);
    }
    if (turn.outputCapture === "PARTIAL" && turn.rawOutputRef) {
      errors.push(`${ref}: partial output must not be a protocol rawOutputRef`);
    }
    for (const [name, value] of Object.entries(turn.timeoutProfile ?? {})) {
      if (!Number.isInteger(value) || Number(value) < 1) {
        errors.push(`${ref}: timeoutProfile.${name} is invalid`);
      }
    }
    if (
      turn.timeoutProfile &&
      turn.timeoutProfile.hardTimeoutMs < turn.timeoutProfile.idleTimeoutMs
    ) {
      errors.push(`${ref}: hard timeout is lower than idle timeout`);
    }
    if (turn.recoveryRef && !store.exists(turn.recoveryRef)) {
      errors.push(`${ref}: missing recovery ${turn.recoveryRef}`);
    } else if (turn.recoveryRef) {
      try {
        const recovery = store.readJson<RuntimeRecoveryRecord>(turn.recoveryRef);
        if (recovery.sourceTurnRef !== turn.retryOf) {
          errors.push(`${ref}: recovery source differs from retryOf`);
        }
        if (
          recovery.role !== turn.role ||
          recovery.taskBindingRef !== turn.taskBindingRef ||
          recovery.decisionContextRef !== turn.decisionContextRef
        ) {
          errors.push(`${ref}: recovery binding differs from Turn binding`);
        }
      } catch (error) {
        errors.push(`${ref}: invalid recovery record: ${message(error)}`);
      }
    }
    if (turn.retryOf && !store.exists(turn.retryOf)) {
      errors.push(`${ref}: missing retryOf Turn ${turn.retryOf}`);
    } else if (turn.role === "DECISION" && turn.retryOf) {
      try {
        const previous = store.readTurn(turn.retryOf);
        if (
          previous.role === "DECISION" &&
          previous.decisionContextRef !== turn.decisionContextRef
        ) {
          errors.push(
            `${ref}: Decision retry does not reuse ${turn.retryOf} context`,
          );
        }
      } catch (error) {
        errors.push(`${ref}: invalid retryOf Turn: ${message(error)}`);
      }
    }
    if (turn.resultRef && !store.exists(turn.resultRef)) {
      errors.push(`${ref}: missing ${turn.resultRef}`);
    }
    if (turn.controlRef && !store.exists(turn.controlRef)) {
      errors.push(`${ref}: missing ${turn.controlRef}`);
    }
    if (turn.resultRef && !turn.controlRef) {
      errors.push(`${ref}: resultRef lacks core control projection`);
    }
    if (turn.turnState === "INVALID_OUTPUT") {
      if (!turn.errorReportRef || !store.exists(turn.errorReportRef)) {
        errors.push(`${ref}: INVALID_OUTPUT lacks E01`);
      } else {
        errors.push(
          ...validateE01(store.readJson(turn.errorReportRef)).map(
            (item) => `${ref}: ${item}`,
          ),
        );
      }
    }
    if (turn.taskBindingRef && turn.resultRef) {
      try {
        const binding = store.readJson<TaskBinding>(turn.taskBindingRef);
        const output = store.readJson<unknown>(turn.resultRef);
        const result = validateCoreOutputForAction(
          binding.action,
          output,
        );
        for (const error of result.errors) {
          errors.push(`${ref}${error.path}: ${error.message}`);
        }
        if (turn.controlRef && result.control) {
          const stored = store.readJson<CoreControlProjection>(turn.controlRef);
          if (JSON.stringify(stored) !== JSON.stringify(result.control)) {
            errors.push(`${ref}: stored control differs from Agent result`);
          }
        }
        for (
          const advisory of validateReferenceTemplateForAction(
            binding.action,
            output,
          )
        ) {
          advisories.push(`${ref}${advisory.path}: ${advisory.message}`);
        }
      } catch (error) {
        errors.push(`${ref}: ${message(error)}`);
      }
    } else if (turn.role === "DECISION" && turn.resultRef && turn.controlRef) {
      try {
        const control =
          store.readJson<CoreControlProjection>(turn.controlRef);
        if (control.role !== "DECISION") {
          errors.push(`${ref}: Decision control projection has wrong role`);
        }
      } catch (error) {
        errors.push(`${ref}: ${message(error)}`);
      }
    }
  }
  return { errors, advisories };
}

function validateE01(value: OutputErrorReport): string[] {
  const errors = exactKeys(value, ["errors"], "E01");
  if (!Array.isArray(value.errors) || value.errors.length === 0) {
    errors.push("E01 errors must be a non-empty array");
    return errors;
  }
  for (const item of value.errors) {
    errors.push(
      ...exactKeys(item, ["check", "path", "message"], "E01.error"),
    );
    if (
      !["DECISION_PROTOCOL", "JSON_PARSE", "CORE_CONTROL"].includes(
        item.check,
      )
    ) {
      errors.push(`E01 unknown check ${item.check}`);
    }
  }
  return errors;
}

function validateObjectIndex(
  store: FileLoopStore,
  index: ObjectsIndex,
): string[] {
  const errors: string[] = [];
  if (new Set(index.activeAnchorIds).size !== index.activeAnchorIds.length) {
    errors.push("ObjectIndex activeAnchorIds contains duplicates");
  }
  for (const id of index.activeAnchorIds) {
    const anchor = index.anchors[id];
    if (!anchor) errors.push(`active Anchor ${id} is missing`);
    else if (anchor.rejected) errors.push(`active Anchor ${id} is rejected`);
  }
  for (const [id, anchor] of Object.entries(index.anchors)) {
    if (anchor.objectId !== id) errors.push(`Anchor key mismatch ${id}`);
    const latest = anchor.revisions[String(anchor.latestRevision)];
    if (!latest) errors.push(`Anchor ${id} lacks latest revision`);
    else if (anchor.rejected !== (latest.reviewVerdict === "REJECT")) {
      errors.push(`Anchor ${id} rejected flag differs from latest R01`);
    }
    if (anchor.rejected && index.activeAnchorIds.includes(id)) {
      errors.push(`rejected Anchor ${id} remains active`);
    }
    if (new Set(anchor.directionIds).size !== anchor.directionIds.length) {
      errors.push(`Anchor ${id} directionIds contains duplicates`);
    }
    for (const [key, revision] of Object.entries(anchor.revisions)) {
      if (key !== String(revision.revision)) {
        errors.push(`Anchor ${id} revision key mismatch ${key}`);
      }
      errors.push(...validateObjectRevision(store, `Anchor ${id}`, revision));
    }
    for (const directionId of anchor.directionIds) {
      const direction = index.directions[directionId];
      if (!direction) {
        errors.push(`Anchor ${id} lacks Direction ${directionId}`);
      } else if (direction.parentAnchorId !== id) {
        errors.push(`Anchor ${id} lists foreign Direction ${directionId}`);
      }
    }
  }
  for (const [id, direction] of Object.entries(index.directions)) {
    if (direction.objectId !== id) errors.push(`Direction key mismatch ${id}`);
    const parent = index.anchors[direction.parentAnchorId];
    if (!parent) {
      errors.push(`Direction ${id} lacks parent ${direction.parentAnchorId}`);
    } else if (!parent.directionIds.includes(id)) {
      errors.push(`Direction ${id} is absent from parent Anchor index`);
    }
    const latest = direction.revisions[String(direction.latestRevision)];
    if (!latest) errors.push(`Direction ${id} lacks latest revision`);
    else if (direction.rejected !== (latest.reviewVerdict === "REJECT")) {
      errors.push(`Direction ${id} rejected flag differs from latest R01`);
    }
    for (const [key, revision] of Object.entries(direction.revisions)) {
      if (key !== String(revision.revision)) {
        errors.push(`Direction ${id} revision key mismatch ${key}`);
      }
      errors.push(...validateObjectRevision(store, `Direction ${id}`, revision));
    }
  }
  return errors;
}

function validateObjectRevision(
  store: FileLoopStore,
  label: string,
  revision: ObjectsIndex["anchors"][string]["revisions"][string],
): string[] {
  const errors: string[] = [];
  for (const ref of [
    revision.workTaskRef,
    revision.workRef,
    revision.reviewRef,
    revision.committedByDecisionTurnRef,
  ]) {
    if (!store.exists(ref)) errors.push(`${label}: missing ${ref}`);
  }
  if (store.exists(revision.reviewRef)) {
    const review = store.readJson<ReviewResult>(revision.reviewRef);
    if (review.reviewVerdict !== revision.reviewVerdict) {
      errors.push(`${label}: indexed verdict differs from R01`);
    }
  }
  if (store.exists(revision.workRef)) {
    const work = store.readJson<{ workOutcome?: unknown }>(revision.workRef);
    if (work.workOutcome !== revision.workOutcome) {
      errors.push(`${label}: indexed workOutcome differs from W01`);
    }
  }
  return errors;
}

function validateState(store: FileLoopStore, state: StateFile): string[] {
  const errors: string[] = [];
  if (!["RUNNING", "PAUSED", "FINISHED", "FAILED"].includes(state.lifecycle)) {
    errors.push(`invalid lifecycle ${state.lifecycle}`);
  }
  if (!store.exists(`rounds/${state.round}.json`)) {
    errors.push(`missing current Round ${state.round}`);
  }
  if (
    state.failureKind !== null &&
    !["RUNTIME_RETRY_EXHAUSTED", "NON_RECOVERABLE"].includes(
      state.failureKind,
    )
  ) {
    errors.push(`invalid failureKind ${String(state.failureKind)}`);
  }
  if (state.lifecycle !== "FAILED" && state.failureKind !== null) {
    errors.push(`${state.lifecycle} state must have failureKind=null`);
  }
  if (state.formatVersion >= 7) {
    const pauseKind = state.pauseKind ?? null;
    if (state.lifecycle === "PAUSED") {
      if (!pauseKind || !PAUSE_KINDS.includes(pauseKind)) {
        errors.push("PAUSED state requires a valid pauseKind");
      }
    } else if (pauseKind !== null) {
      errors.push(`${state.lifecycle} state must have pauseKind=null`);
    }

    const authorizedThroughRound =
      state.roundBudget?.authorizedThroughRound;
    if (
      !Number.isInteger(authorizedThroughRound) ||
      Number(authorizedThroughRound) < 1
    ) {
      errors.push("formatVersion 7 state requires a valid roundBudget");
    } else {
      const authorized = Number(authorizedThroughRound);
      if (state.lifecycle === "RUNNING" && state.round > authorized) {
        errors.push("RUNNING round exceeds the authorized round budget");
      }
      if (
        pauseKind === "ROUND_BUDGET_EXHAUSTED" &&
        state.round !== authorized + 1
      ) {
        errors.push(
          "ROUND_BUDGET_EXHAUSTED must prepare exactly the first unauthorized round",
        );
      }
    }
  }
  if (state.runtimeRecovery) {
    if (!store.exists(state.runtimeRecovery.recoveryRef)) {
      errors.push(`runtimeRecovery missing ${state.runtimeRecovery.recoveryRef}`);
    }
    if (!store.exists(state.runtimeRecovery.sourceTurnRef)) {
      errors.push(`runtimeRecovery missing ${state.runtimeRecovery.sourceTurnRef}`);
    }
    if (state.lifecycle !== "RUNNING") {
      errors.push("runtimeRecovery requires lifecycle=RUNNING");
    }
  }
  if (state.activeTaskBindingRef && !store.exists(state.activeTaskBindingRef)) {
    errors.push(`missing activeTaskBindingRef ${state.activeTaskBindingRef}`);
  }
  if (state.activeTurnRef && !store.exists(state.activeTurnRef)) {
    errors.push(`missing activeTurnRef ${state.activeTurnRef}`);
  }
  if (state.pending) {
    for (const ref of [
      state.pending.workTaskBindingRef,
      state.pending.workTaskRef,
      state.pending.workTurnRef,
      state.pending.workRef,
      state.pending.reviewTaskBindingRef,
      state.pending.reviewTurnRef,
      state.pending.reviewRef,
    ].filter((value): value is string => Boolean(value))) {
      if (!store.exists(ref)) errors.push(`pending missing ${ref}`);
    }
    if (store.exists(state.pending.workRef)) {
      const work =
        store.readJson<{ workOutcome?: unknown }>(state.pending.workRef);
      if (work.workOutcome !== state.pending.workOutcome) {
        errors.push("pending workOutcome differs from W01");
      }
    }
    if (state.pending.reviewRef && store.exists(state.pending.reviewRef)) {
      const review =
        store.readJson<{ reviewVerdict?: unknown }>(state.pending.reviewRef);
      if (review.reviewVerdict !== state.pending.reviewVerdict) {
        errors.push("pending reviewVerdict differs from R01");
      }
    } else if (state.pending.reviewVerdict !== null) {
      errors.push("pending has reviewVerdict without R01");
    }
  }
  if (state.preReview) {
    for (const ref of [
      state.preReview.workRef,
      state.preReview.reviewTaskBindingRef,
      state.preReview.reviewTurnRef,
      state.preReview.reviewRef,
    ]) {
      if (!store.exists(ref)) errors.push(`preReview missing ${ref}`);
    }
    if (store.exists(state.preReview.workRef)) {
      const work =
        store.readJson<{ workOutcome?: unknown }>(state.preReview.workRef);
      if (work.workOutcome !== state.preReview.workOutcome) {
        errors.push("preReview workOutcome differs from W01");
      }
    }
    if (store.exists(state.preReview.reviewRef)) {
      const review =
        store.readJson<{ reviewVerdict?: unknown }>(state.preReview.reviewRef);
      if (review.reviewVerdict !== state.preReview.reviewVerdict) {
        errors.push("preReview reviewVerdict differs from R01");
      }
    }
    const index = store.readObjects();
    const entry =
      state.preReview.objectKind === "ANCHOR"
        ? index.anchors[state.preReview.objectId]
        : index.directions[state.preReview.objectId];
    const revision = entry?.revisions[String(state.preReview.revision)];
    if (
      !entry ||
      entry.latestRevision !== state.preReview.revision ||
      revision?.workRef !== state.preReview.workRef ||
      revision?.reviewRef !== state.preReview.reviewRef
    ) {
      errors.push("preReview is not the target object's current review");
    }
  }
  if (state.correction) {
    for (const ref of [
      state.correction.retryOfTurnRef,
      state.correction.previousOutputRef,
      state.correction.errorReportRef,
    ]) {
      if (!store.exists(ref)) errors.push(`correction missing ${ref}`);
    }
  }
  if (state.lifecycle === "RUNNING") {
    if (state.node !== (state.sequence[0]?.role ?? null)) {
      errors.push("RUNNING node differs from sequence head");
    }
  } else if (state.node !== null) {
    errors.push(`${state.lifecycle} state must have node=null`);
  }
  if (state.lifecycle === "FINISHED") {
    if (state.sequence.length > 0) {
      errors.push("FINISHED state must have an empty sequence");
    }
    const remaining = computeRemainingRequirements(store, state, false);
    if (remaining.length > 0) {
      errors.push(`FINISHED with open requirements: ${remaining.join(", ")}`);
    }
  }
  return errors;
}

function validateFinalArtifacts(
  store: FileLoopStore,
  state: StateFile,
): string[] {
  if (state.lifecycle === "RUNNING") return [];
  const errors: string[] = [];
  if (state.lifecycle === "FINISHED") {
    for (const ref of [
      "final/report.md",
      "final/manifest.json",
    ]) {
      if (!store.exists(ref)) errors.push(`missing ${ref}`);
    }
  }
  if (!store.exists("final/outcome.json")) {
    errors.push("missing final/outcome.json");
    return errors;
  }
  const outcome = store.readJson<Record<string, unknown>>(
    "final/outcome.json",
  );
  errors.push(
    ...exactKeys(
      outcome,
      ["workflowOutcome", "reportRef", "reason"],
      "O01",
    ),
  );
  if (outcome.workflowOutcome !== state.lifecycle) {
    errors.push("O01 workflowOutcome differs from lifecycle");
  }
  if (state.lifecycle === "FINISHED") {
    if (outcome.reportRef !== "final/report.md" || outcome.reason !== null) {
      errors.push("O01 does not describe a finished run");
    }
  } else if (
    outcome.reportRef !== null ||
    !nonEmpty(outcome.reason)
  ) {
    errors.push(`${state.lifecycle} O01 requires reportRef=null and a reason`);
  }
  return errors;
}

function jsonFiles(store: FileLoopStore, directory: string): string[] {
  const root = store.absolute(directory);
  if (!existsSync(root)) return [];
  const result: string[] = [];
  const walk = (absolute: string, relativePrefix: string) => {
    for (const entry of readdirSync(absolute, { withFileTypes: true })) {
      const ref = `${relativePrefix}/${entry.name}`;
      if (entry.isDirectory()) walk(store.absolute(ref), ref);
      else if (entry.name.endsWith(".json")) result.push(ref);
    }
  };
  walk(root, directory);
  return result;
}

function turnRefs(store: FileLoopStore): string[] {
  const root = store.absolute("turns");
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => `turns/${entry.name}/turn.json`)
    .filter((ref) => store.exists(ref));
}

function exactKeys(
  value: unknown,
  allowed: string[],
  label: string,
  subset = false,
): string[] {
  if (!isRecord(value)) return [`${label} must be an object`];
  const actual = Object.keys(value);
  const unknown = actual.filter((key) => !allowed.includes(key));
  const missing = subset ? [] : allowed.filter((key) => !actual.includes(key));
  return [
    ...unknown.map((key) => `${label} has unknown property ${key}`),
    ...missing.map((key) => `${label} is missing property ${key}`),
  ];
}

function validRequirement(value: string): boolean {
  return (
    value === "ANCHOR_REQUIRED" ||
    /^ANCHOR_REVIEW_PASS_REQUIRED:.+$/.test(value) ||
    /^DIRECTION_REQUIRED:.+$/.test(value) ||
    /^DIRECTION_REVIEW_PASS_REQUIRED:.+$/.test(value)
  );
}

function nonEmpty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function nonEmptyStringList(value: unknown): boolean {
  return Array.isArray(value) && value.length > 0 && value.every(nonEmpty);
}

function resultPaths(
  result: CommittedResult | PendingResults,
): string[] {
  return Object.entries(result)
    .filter(([key, value]) =>
      key !== "objectKind" && typeof value === "string"
    )
    .map(([, value]) => value as string);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
