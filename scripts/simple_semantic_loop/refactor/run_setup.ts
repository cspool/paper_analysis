import { existsSync, readFileSync } from "node:fs";
import { relative, resolve, sep } from "node:path";
import {
  REFERENCE_TEMPLATE_SCHEMAS,
} from "./schemas.ts";
import { FileLoopStore } from "./store.ts";
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
  idleTimeoutMs?: number;
  hardTimeoutMs?: number;
  interruptGraceMs?: number;
}

export const DEFAULT_ACCEPTANCE_CRITERIA = [
  "从本地多维知识库形成与 Topic 相关且有来源支撑的性能优化潜力",
  "Topic 的 6L 空间由未被拒绝的 Anchor 集合动态定义",
  "每个最终 Anchor 明确场景、baseline、性能矛盾、6L 区域和约束，并获得独立 Reviewer PASS",
  "每个最终 Anchor 至少有一个 Direction；每个最终 Direction 获得独立 Reviewer PASS",
  "每个 Direction 以最小充分方式明确一个可检验主要变化或诚实的不可分联合包，并给出机制、baseline change、预期影响、权衡、失败条件和可证伪测量计划",
  "需要新实验的结论只形成测量计划，不在本工作流内执行实验",
];

export const DEFAULT_BUDGETS: RunBudgets = Object.freeze({
  maxRounds: 12,
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
  const timeoutProfiles = structuredClone(DEFAULT_BUDGETS.timeoutProfiles);
  if (options.idleTimeoutMs !== undefined) {
    assertPositiveInteger(options.idleTimeoutMs, "idleTimeoutMs");
    for (const profile of Object.values(timeoutProfiles)) {
      profile.idleTimeoutMs = options.idleTimeoutMs;
    }
  }
  if (options.hardTimeoutMs !== undefined) {
    assertPositiveInteger(options.hardTimeoutMs, "hardTimeoutMs");
    for (const profile of Object.values(timeoutProfiles)) {
      profile.hardTimeoutMs = options.hardTimeoutMs;
    }
  }
  if (options.interruptGraceMs !== undefined) {
    assertPositiveInteger(options.interruptGraceMs, "interruptGraceMs");
    for (const profile of Object.values(timeoutProfiles)) {
      profile.interruptGraceMs = options.interruptGraceMs;
    }
  }
  for (const [role, profile] of Object.entries(timeoutProfiles)) {
    if (profile.hardTimeoutMs < profile.idleTimeoutMs) {
      throw new Error(`${role} hardTimeoutMs must be >= idleTimeoutMs`);
    }
  }
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

  const skillPaths = {
    decision: ".codex/skills/learning-loop-decision/SKILL.md",
    worker: ".codex/skills/learning-loop-worker/SKILL.md",
    reviewer: ".codex/skills/learning-loop-reviewer/SKILL.md",
  } as const;
  for (const path of Object.values(skillPaths)) {
    if (!existsSync(resolve(projectRoot, path))) {
      throw new Error(`missing required Skill ${path}`);
    }
  }

  const run: RunFile = {
    formatVersion: CURRENT_FORMAT_VERSION,
    runId: store.newId("run"),
    createdAt: new Date().toISOString(),
    projectRoot,
    model: options.model ?? "gpt-5.6-sol",
    goalRef: "workflow_goal.json",
    skills: {
      decision: pinSkill(store, projectRoot, "learning-loop-decision", skillPaths.decision),
      worker: pinSkill(store, projectRoot, "learning-loop-worker", skillPaths.worker),
      reviewer: pinSkill(store, projectRoot, "learning-loop-reviewer", skillPaths.reviewer),
    },
    budgets: {
      ...DEFAULT_BUDGETS,
      maxRounds,
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
    pending: null,
    preReview: null,
    decisionGuidance: null,
    correction: null,
    runtimeRecovery: null,
    latestDecisionTurnRef: null,
    semanticRetries: { worker: 0, reviewer: 0 },
  };
  store.writeState(state, "RUN_INITIALIZED");
  return run;
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
