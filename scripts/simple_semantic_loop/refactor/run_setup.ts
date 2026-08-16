import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
} from "node:fs";
import { relative, resolve, sep } from "node:path";
import {
  REFERENCE_TEMPLATE_SCHEMAS,
} from "./schemas.ts";
import { FileLoopStore } from "./store.ts";
import { rebuildNegativeExperimentIndex } from "./experiment_history.ts";
import type {
  ObjectsIndex,
  RefCatalog,
  RunBudgets,
  RunFile,
  SkillPin,
  StateFile,
  WorkflowGoal,
} from "./types.ts";
import { CURRENT_FORMAT_VERSION } from "./types.ts";

export interface InitOptions {
  projectRoot: string;
  workDir: string;
  topic: string;
  objective?: string;
  acceptanceCriteria?: string[];
  model?: string;
  maxRounds?: number;
  maxExperimentGoals?: number;
  idleTimeoutMs?: number;
  hardTimeoutMs?: number;
  interruptGraceMs?: number;
  experimentIdleTimeoutMs?: number;
  experimentHardTimeoutMs?: number;
  experimentInterruptGraceMs?: number;
}

export interface ContinueOptions {
  projectRoot: string;
  sourceWorkDir: string;
  workDir: string;
  model?: string;
  maxRounds?: number;
  maxExperimentGoals?: number;
  idleTimeoutMs?: number;
  hardTimeoutMs?: number;
  interruptGraceMs?: number;
  experimentIdleTimeoutMs?: number;
  experimentHardTimeoutMs?: number;
  experimentInterruptGraceMs?: number;
  /** Allow a PAUSED source and grant fresh round/EXP authorization. */
  resetBudgets?: boolean;
}

export const DEFAULT_ACCEPTANCE_CRITERIA = [
  "从本地多维知识库形成与 Topic 相关且有来源支撑的性能优化潜力",
  "Topic 的 6L 空间由未被拒绝的 Anchor 集合动态定义",
  "每个最终 Anchor 明确场景、baseline、性能矛盾、6L 区域和约束，并获得独立 Reviewer PASS",
  "每个最终 Anchor 至少有一个 Direction；每个最终 Direction 获得独立 Reviewer PASS",
  "每个 Direction 以最小充分方式明确一个可检验主要变化或诚实的不可分联合包，并给出机制、baseline change、预期影响、权衡、失败条件和可证伪测量计划",
  "普通 Worker 和 Reviewer Turn 不执行新实验；只有 Decision 选择且 Script 授权的 EXP Goal 可以执行有界实验，结果必须回到 Decision 后再决定是否修订对象",
];

export const DEFAULT_BUDGETS: RunBudgets = Object.freeze({
  maxRounds: 12,
  maxExperimentGoals: 5,
  experimentGoalTokenBudget: null,
  maxOutputRetries: 2,
  maxRuntimeRetries: 2,
  maxSemanticRetries: 2,
  timeoutProfiles: {
    DECISION: {
      idleTimeoutMs: 360_000,
      hardTimeoutMs: 900_000,
      interruptGraceMs: 15_000,
    },
    WORKER: {
      idleTimeoutMs: 300_000,
      hardTimeoutMs: 900_000,
      interruptGraceMs: 15_000,
    },
    REVIEWER: {
      idleTimeoutMs: 300_000,
      hardTimeoutMs: 900_000,
      interruptGraceMs: 15_000,
    },
    EXP_GOAL: {
      idleTimeoutMs: 900_000,
      hardTimeoutMs: 21_600_000,
      interruptGraceMs: 15_000,
    },
  },
  maxInputTokens: 96_000,
  maxOutputTokens: 24_000,
});

export function initializeRun(options: InitOptions): RunFile {
  const projectRoot = resolve(options.projectRoot);
  const workDir = resolve(options.workDir);
  const store = new FileLoopStore(workDir);
  if (store.exists("run.json")) {
    throw new Error(`run.json already exists in ${workDir}`);
  }

  store.initializeLayout();

  const maxRounds = options.maxRounds ?? DEFAULT_BUDGETS.maxRounds;
  if (!Number.isInteger(maxRounds) || maxRounds < 1) {
    throw new Error("maxRounds must be a positive integer");
  }
  const maxExperimentGoals =
    options.maxExperimentGoals ?? DEFAULT_BUDGETS.maxExperimentGoals;
  if (!Number.isInteger(maxExperimentGoals) || maxExperimentGoals < 0) {
    throw new Error("maxExperimentGoals must be a non-negative integer");
  }
  const timeoutProfiles = buildTimeoutProfiles(options);
  const topic = options.topic.trim();
  if (!topic) throw new Error("topic must not be empty");
  const objective =
    options.objective?.trim() ||
    "从本地多维知识库识别并形成与该 Topic 相关、可验证的性能优化潜力。";
  const acceptanceCriteria =
    options.acceptanceCriteria?.map((item) => item.trim()).filter(Boolean) ??
    [];

  const goal: WorkflowGoal = {
    topic,
    objective,
    acceptanceCriteria:
      acceptanceCriteria.length > 0
        ? acceptanceCriteria
        : [...DEFAULT_ACCEPTANCE_CRITERIA],
  };
  store.writeJson("workflow_goal.json", goal);

  const run: RunFile = {
    formatVersion: CURRENT_FORMAT_VERSION,
    runId: store.newId("run"),
    createdAt: new Date().toISOString(),
    projectRoot,
    model: options.model ?? "gpt-5.6-sol",
    goalRef: "workflow_goal.json",
    skills: currentSkillPins(store, projectRoot),
    budgets: {
      ...DEFAULT_BUDGETS,
      maxRounds,
      maxExperimentGoals,
      timeoutProfiles,
    },
  };
  store.writeJson("run.json", run);
  store.writeJson(
    "ref_catalog.json",
    buildRefCatalog(store, projectRoot),
  );

  const objects: ObjectsIndex = {
    revision: 0,
    activeAnchorIds: [],
    anchors: {},
    directions: {},
  };
  store.writeJson("objects/index.json", objects);
  store.writeText("observations/progress_trajectory.jsonl", "");
  store.writeRound({
    round: 1,
    branch: "INITIAL",
    turnRefs: [],
    experimentRefs: [],
    committedAt: null,
  });

  const state: StateFile = {
    formatVersion: CURRENT_FORMAT_VERSION,
    revision: 1,
    lifecycle: "RUNNING",
    reason: null,
    pauseKind: null,
    roundBudget: {
      authorizedThroughRound: maxRounds,
      lastAuthorizationRef: null,
    },
    failureKind: null,
    node: "WORKER",
    round: 1,
    transitions: 0,
    sequence: [
      { role: "WORKER", mode: "NORMAL_WORK", bindingRef: null },
      { role: "REVIEWER", mode: "PAIR_REVIEW", bindingRef: null },
      { role: "DECISION", mode: "DECISION", bindingRef: null },
    ],
    activeTaskBindingRef: null,
    activeTurnRef: null,
    activeExperimentRef: null,
    latestExperimentResultRef: null,
    experimentGoalsStarted: 0,
    pending: null,
    preReview: null,
    decisionGuidance: null,
    correction: null,
    runtimeRecovery: null,
    latestDecisionTurnRef: null,
    semanticRetries: { worker: 0, reviewer: 0 },
  };
  store.writeState(state, "RUN_INITIALIZED");
  rebuildNegativeExperimentIndex(store, state);
  return run;
}

export function continueRunFromFinished(options: ContinueOptions): RunFile {
  const projectRoot = resolve(options.projectRoot);
  const sourceWorkDir = resolve(options.sourceWorkDir);
  const workDir = resolve(options.workDir);
  if (sourceWorkDir === workDir) {
    throw new Error("continuation workDir must differ from sourceWorkDir");
  }
  if (isSameOrNested(sourceWorkDir, workDir) || isSameOrNested(workDir, sourceWorkDir)) {
    throw new Error("sourceWorkDir and continuation workDir must not contain one another");
  }
  if (!existsSync(resolve(sourceWorkDir, "run.json"))) {
    throw new Error(`source run.json does not exist in ${sourceWorkDir}`);
  }
  if (existsSync(workDir) && readdirSync(workDir).length > 0) {
    throw new Error(`continuation workDir is not empty: ${workDir}`);
  }

  const sourceStore = new FileLoopStore(sourceWorkDir);
  const sourceRun = sourceStore.readRun();
  const sourceState = sourceStore.readState();
  const resetBudgets = options.resetBudgets === true;
  const sourceLifecycleAllowed = sourceState.lifecycle === "FINISHED" ||
    (resetBudgets && sourceState.lifecycle === "PAUSED");
  if (!sourceLifecycleAllowed) {
    throw new Error(
      resetBudgets
        ? `budget-reset continuation requires a PAUSED or FINISHED source run, got ${sourceState.lifecycle}`
        : `continuation requires a FINISHED source run, got ${sourceState.lifecycle}`,
    );
  }
  if (
    resetBudgets &&
    (sourceState.activeTurnRef ||
      sourceState.activeExperimentRef ||
      sourceState.pending ||
      sourceState.correction)
  ) {
    throw new Error(
      "budget-reset continuation requires a stable boundary without an active Turn, EXP Goal, pending pair, or output correction",
    );
  }
  if (
    sourceRun.formatVersion !== CURRENT_FORMAT_VERSION ||
    sourceState.formatVersion !== CURRENT_FORMAT_VERSION
  ) {
    throw new Error(
      `continuation requires formatVersion ${CURRENT_FORMAT_VERSION}; source is ${sourceRun.formatVersion}/${sourceState.formatVersion}`,
    );
  }
  if (sourceState.lifecycle === "FINISHED") {
    for (const ref of ["final/report.md", "final/manifest.json"]) {
      if (!sourceStore.exists(ref)) {
        throw new Error(`finished source is missing ${ref}`);
      }
    }
  }

  mkdirSync(workDir, { recursive: true });
  for (const entry of readdirSync(sourceWorkDir, { withFileTypes: true })) {
    if (entry.name === ".run.lock" || entry.name === "final") continue;
    cpSync(
      resolve(sourceWorkDir, entry.name),
      resolve(workDir, entry.name),
      { recursive: true },
    );
  }
  const sourceSnapshotRoot = `continuation/${sourceRun.runId}`;
  mkdirSync(resolve(workDir, sourceSnapshotRoot), { recursive: true });
  cpSync(
    resolve(sourceWorkDir, "run.json"),
    resolve(workDir, `${sourceSnapshotRoot}/run.json`),
  );
  cpSync(
    resolve(sourceWorkDir, "state.json"),
    resolve(workDir, `${sourceSnapshotRoot}/state.json`),
  );
  if (existsSync(resolve(sourceWorkDir, "final"))) {
    cpSync(
      resolve(sourceWorkDir, "final"),
      resolve(workDir, `${sourceSnapshotRoot}/final`),
      { recursive: true },
    );
  }

  const store = new FileLoopStore(workDir);
  store.initializeLayout();
  const maxRounds = options.maxRounds ?? DEFAULT_BUDGETS.maxRounds;
  assertPositiveInteger(maxRounds, "maxRounds");
  const copiedExperimentCount = store.experimentRefs().length;
  const maxExperimentGoals = options.maxExperimentGoals ??
    (resetBudgets
      ? DEFAULT_BUDGETS.maxExperimentGoals
      : Math.max(DEFAULT_BUDGETS.maxExperimentGoals, copiedExperimentCount));
  if (
    !Number.isInteger(maxExperimentGoals) ||
    maxExperimentGoals < 0 ||
    (!resetBudgets && maxExperimentGoals < copiedExperimentCount)
  ) {
    throw new Error(
      resetBudgets
        ? "maxExperimentGoals must be a non-negative integer"
        : `maxExperimentGoals must be an integer >= copied experiment count ${copiedExperimentCount}`,
    );
  }
  const timeoutProfiles = buildTimeoutProfiles(options);
  const continuedAt = new Date().toISOString();
  const sourceHasManifest = sourceStore.exists("final/manifest.json");
  const sourceHasReport = sourceStore.exists("final/report.md");
  const continuation = {
    sourceRunId: sourceRun.runId,
    sourceWorkDir,
    sourceStateRevision: sourceState.revision,
    sourceLifecycle: sourceState.lifecycle,
    sourceRunSha256: sourceStore.sha256File("run.json"),
    sourceStateSha256: sourceStore.sha256File("state.json"),
    sourceManifestSha256: sourceHasManifest
      ? sourceStore.sha256File("final/manifest.json")
      : null,
    sourceRunRef: `${sourceSnapshotRoot}/run.json`,
    sourceStateRef: `${sourceSnapshotRoot}/state.json`,
    sourceManifestRef: sourceHasManifest
      ? `${sourceSnapshotRoot}/final/manifest.json`
      : null,
    sourceFinalReportRef: sourceHasReport
      ? `${sourceSnapshotRoot}/final/report.md`
      : null,
    budgetReset: resetBudgets,
    sourceExperimentCount: copiedExperimentCount,
    continuedAt,
  } satisfies NonNullable<RunFile["continuation"]>;
  const run: RunFile = {
    ...sourceRun,
    runId: store.newId("run"),
    createdAt: continuedAt,
    projectRoot,
    model: options.model ?? sourceRun.model,
    skills: currentSkillPins(store, projectRoot),
    budgets: {
      ...DEFAULT_BUDGETS,
      maxRounds,
      maxExperimentGoals,
      timeoutProfiles,
    },
    continuation,
  };
  store.writeJson("run.json", run);
  store.writeJson("ref_catalog.json", buildRefCatalog(store, projectRoot));
  const sourceRecordRef = `${sourceSnapshotRoot}/source.json`;
  store.writeImmutableJson(sourceRecordRef, continuation);

  const nextRound = sourceState.round + 1;
  if (store.exists(`rounds/${nextRound}.json`)) {
    throw new Error(`source already contains unexpected round ${nextRound}`);
  }
  store.writeRound({
    round: nextRound,
    branch: "CONTINUATION",
    turnRefs: [],
    experimentRefs: [],
    committedAt: null,
  });
  const nextState: StateFile = {
    ...sourceState,
    revision: sourceState.revision + 1,
    lifecycle: "RUNNING",
    reason: null,
    pauseKind: null,
    roundBudget: {
      authorizedThroughRound: nextRound + maxRounds - 1,
      lastAuthorizationRef: null,
    },
    failureKind: null,
    node: "DECISION",
    round: nextRound,
    transitions: sourceState.transitions + 1,
    sequence: [{ role: "DECISION", mode: "DECISION", bindingRef: null }],
    activeTaskBindingRef: null,
    activeTurnRef: null,
    activeExperimentRef: null,
    experimentGoalsStarted: resetBudgets ? 0 : copiedExperimentCount,
    pending: null,
    preReview: null,
    decisionGuidance: null,
    correction: null,
    runtimeRecovery: null,
    semanticRetries: { worker: 0, reviewer: 0 },
  };
  store.writeState(
    nextState,
    resetBudgets
      ? "RUN_CONTINUED_WITH_RESET_BUDGETS"
      : "RUN_CONTINUED_FROM_FINISHED",
  );
  rebuildNegativeExperimentIndex(store, nextState);
  store.appendEvent("SOURCE_FINAL_SNAPSHOTTED", [
    sourceRecordRef,
    continuation.sourceRunRef,
    continuation.sourceStateRef,
    ...(continuation.sourceFinalReportRef
      ? [continuation.sourceFinalReportRef]
      : []),
    ...(continuation.sourceManifestRef ? [continuation.sourceManifestRef] : []),
  ]);
  return run;
}

type TimeoutOptions = Pick<
  InitOptions,
  | "idleTimeoutMs"
  | "hardTimeoutMs"
  | "interruptGraceMs"
  | "experimentIdleTimeoutMs"
  | "experimentHardTimeoutMs"
  | "experimentInterruptGraceMs"
>;

function buildTimeoutProfiles(options: TimeoutOptions): RunBudgets["timeoutProfiles"] {
  const profiles = structuredClone(DEFAULT_BUDGETS.timeoutProfiles);
  const regularRoles = ["DECISION", "WORKER", "REVIEWER"] as const;
  if (options.idleTimeoutMs !== undefined) {
    assertPositiveInteger(options.idleTimeoutMs, "idleTimeoutMs");
    for (const role of regularRoles) profiles[role].idleTimeoutMs = options.idleTimeoutMs;
  }
  if (options.hardTimeoutMs !== undefined) {
    assertPositiveInteger(options.hardTimeoutMs, "hardTimeoutMs");
    for (const role of regularRoles) profiles[role].hardTimeoutMs = options.hardTimeoutMs;
  }
  if (options.interruptGraceMs !== undefined) {
    assertPositiveInteger(options.interruptGraceMs, "interruptGraceMs");
    for (const role of regularRoles) {
      profiles[role].interruptGraceMs = options.interruptGraceMs;
    }
  }
  if (options.experimentIdleTimeoutMs !== undefined) {
    assertPositiveInteger(options.experimentIdleTimeoutMs, "experimentIdleTimeoutMs");
    profiles.EXP_GOAL.idleTimeoutMs = options.experimentIdleTimeoutMs;
  }
  if (options.experimentHardTimeoutMs !== undefined) {
    assertPositiveInteger(options.experimentHardTimeoutMs, "experimentHardTimeoutMs");
    profiles.EXP_GOAL.hardTimeoutMs = options.experimentHardTimeoutMs;
  }
  if (options.experimentInterruptGraceMs !== undefined) {
    assertPositiveInteger(
      options.experimentInterruptGraceMs,
      "experimentInterruptGraceMs",
    );
    profiles.EXP_GOAL.interruptGraceMs = options.experimentInterruptGraceMs;
  }
  for (const [role, profile] of Object.entries(profiles)) {
    if (profile.hardTimeoutMs < profile.idleTimeoutMs) {
      throw new Error(`${role} hardTimeoutMs must be >= idleTimeoutMs`);
    }
  }
  return profiles;
}

const SKILL_PATHS = {
  decision: ".codex/skills/learning-loop-decision/SKILL.md",
  worker: ".codex/skills/learning-loop-worker/SKILL.md",
  reviewer: ".codex/skills/learning-loop-reviewer/SKILL.md",
  experiment: ".codex/skills/learning-exp-goal/SKILL.md",
} as const;

function currentSkillPins(
  store: FileLoopStore,
  projectRoot: string,
): RunFile["skills"] {
  for (const path of Object.values(SKILL_PATHS)) {
    if (!existsSync(resolve(projectRoot, path))) {
      throw new Error(`missing required Skill ${path}`);
    }
  }
  return {
    decision: pinSkill(
      store,
      projectRoot,
      "learning-loop-decision",
      SKILL_PATHS.decision,
    ),
    worker: pinSkill(
      store,
      projectRoot,
      "learning-loop-worker",
      SKILL_PATHS.worker,
    ),
    reviewer: pinSkill(
      store,
      projectRoot,
      "learning-loop-reviewer",
      SKILL_PATHS.reviewer,
    ),
    experiment: pinSkill(
      store,
      projectRoot,
      "learning-exp-goal",
      SKILL_PATHS.experiment,
    ),
  };
}

function isSameOrNested(parent: string, candidate: string): boolean {
  const path = relative(parent, candidate);
  return path === "" || (!path.startsWith(`..${sep}`) && path !== "..");
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
}

export function validatePublishedReferenceTemplates(
  projectRoot: string,
): string[] {
  const paths = schemaFilePaths(resolve(projectRoot));
  const errors: string[] = [];
  for (
    const [name, expected] of Object.entries(REFERENCE_TEMPLATE_SCHEMAS)
  ) {
    const path = paths[name as keyof typeof paths];
    if (!existsSync(path)) {
      errors.push(`${name}: missing ${path}`);
      continue;
    }
    let actual: unknown;
    try {
      actual = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
      errors.push(`${name}: invalid JSON: ${String(error)}`);
      continue;
    }
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      errors.push(
        `${name}: published reference template differs from in-code template`,
      );
    }
  }
  return errors;
}

function buildRefCatalog(
  store: FileLoopStore,
  projectRoot: string,
): RefCatalog {
  const paths: Record<
    string,
    { path: string; templateSchemaPath?: string }
  > = {
    "work-result-anchor-v2": {
      path:
        ".codex/skills/learning-loop-worker/references/work_result_anchor_v2.md",
      templateSchemaPath:
        "scripts/simple_semantic_loop/schemas/work_result_anchor_v2.schema.json",
    },
    "work-result-direction-v2": {
      path:
        ".codex/skills/learning-loop-worker/references/work_result_direction_v2.md",
      templateSchemaPath:
        "scripts/simple_semantic_loop/schemas/work_result_direction_v2.schema.json",
    },
    "review-result-v2": {
      path:
        ".codex/skills/learning-loop-reviewer/references/review_result_v2.md",
      templateSchemaPath:
        "scripts/simple_semantic_loop/schemas/review_result_v2.schema.json",
    },
    "decision-line-protocol-v1": {
      path:
        ".codex/skills/learning-loop-decision/references/decision_protocol_v1.md",
    },
    "learning-6l-v1": {
      path:
        ".codex/skills/learning-loop-worker/references/learning_6l_v1.md",
    },
    "knowledge-retrieval-v1": {
      path:
        ".codex/skills/learning-loop-worker/references/knowledge_retrieval_v1.md",
    },
    "review-rubric-v1": {
      path:
        ".codex/skills/learning-loop-reviewer/references/review_rubric_v1.md",
    },
    "optimization-value-questions-v1": {
      path:
        ".codex/skills/learning-loop-reviewer/references/optimization_value_questions_v1.md",
    },
    "experiment-goal-task-v1": {
      path:
        ".codex/skills/learning-exp-goal/references/experiment_goal_task_v1.md",
    },
  };

  const catalog: RefCatalog = {};
  for (const [name, entry] of Object.entries(paths)) {
    const absolute = resolve(projectRoot, entry.path);
    if (!existsSync(absolute)) throw new Error(`missing Ref ${entry.path}`);
    catalog[name] = {
      path: projectRelative(projectRoot, absolute),
      sha256: store.sha256ExternalFile(absolute),
      ...(entry.templateSchemaPath
        ? { templateSchemaPath: entry.templateSchemaPath }
        : {}),
    };
  }
  return catalog;
}

function pinSkill(
  store: FileLoopStore,
  projectRoot: string,
  name: string,
  path: string,
): SkillPin {
  const absolute = resolve(projectRoot, path);
  return {
    name,
    path: projectRelative(projectRoot, absolute),
    sha256: store.sha256ExternalFile(absolute),
  };
}

function schemaFilePaths(projectRoot: string) {
  return {
    "work-result-anchor-v2": resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/work_result_anchor_v2.schema.json",
    ),
    "work-result-direction-v2": resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/work_result_direction_v2.schema.json",
    ),
    "review-result-v2": resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/review_result_v2.schema.json",
    ),
  };
}

function projectRelative(projectRoot: string, path: string): string {
  return relative(projectRoot, path).split(sep).join("/");
}
