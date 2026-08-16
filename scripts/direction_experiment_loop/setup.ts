import { createHash, randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { DirectionExperimentStore } from "./store.ts";
import {
  DIRECTION_EXPERIMENT_FORMAT_VERSION,
  type DirectionInputPins,
  type DirectionRunFile,
  type DirectionSourceBinding,
  type DirectionStateFile,
  type EvidenceManifestItem,
  type FilePin,
} from "./types.ts";

export interface DirectionInitOptions {
  projectRoot: string;
  workDir: string;
  directionResultPath: string;
  sourceWorkDir?: string;
  model?: string;
  maxCycles?: number;
  decisionIdleTimeoutMs?: number;
  decisionHardTimeoutMs?: number;
  judgeIdleTimeoutMs?: number;
  judgeHardTimeoutMs?: number;
  labIdleTimeoutMs?: number;
  labHardTimeoutMs?: number;
  interruptGraceMs?: number;
}

interface ImportedTarget {
  binding: DirectionSourceBinding;
  sourceRun: Record<string, unknown>;
  direction: Record<string, unknown>;
  anchor: Record<string, unknown>;
  review: Record<string, unknown>;
  evidence: EvidenceManifestItem[];
  targetMarkdown: string;
}

const SKILL_FILES = {
  decision: ".codex/skills/direction-experiment-decision/SKILL.md",
  decisionMethod:
    ".codex/skills/direction-experiment-decision/references/decision_method.md",
  judge: ".codex/skills/direction-evidence-judge/SKILL.md",
  lab: ".codex/skills/direction-lab-goal/SKILL.md",
  judgeContract:
    ".codex/skills/direction-evidence-judge/references/judgment_contract.md",
  labMethod: ".codex/skills/direction-lab-goal/references/lab_method.md",
} as const;

const FROZEN_SKILL_FILES = {
  decision: "inputs/skills/direction-experiment-decision/SKILL.md",
  decisionMethod: "inputs/skills/direction-experiment-decision/references/decision_method.md",
  judge: "inputs/skills/direction-evidence-judge/SKILL.md",
  lab: "inputs/skills/direction-lab-goal/SKILL.md",
  judgeContract: "inputs/skills/direction-evidence-judge/references/judgment_contract.md",
  labMethod: "inputs/skills/direction-lab-goal/references/lab_method.md",
} as const;

const EXPERIMENT_POLICY_FILE =
  "scripts/direction_experiment_loop/contracts/experiment_policy.md";

export function initializeDirectionExperimentRun(
  options: DirectionInitOptions,
): DirectionRunFile {
  const projectRoot = resolve(options.projectRoot);
  const workDir = resolve(options.workDir);
  const directionResultPath = resolve(options.directionResultPath);
  if (!existsSync(directionResultPath)) {
    throw new Error(`Direction result does not exist: ${directionResultPath}`);
  }
  if (existsSync(workDir) && readdirSync(workDir).length > 0) {
    throw new Error(`workDir must be empty: ${workDir}`);
  }
  const maxCycles = options.maxCycles ?? 5;
  assertPositiveInteger(maxCycles, "maxCycles");
  const interruptGraceMs = options.interruptGraceMs ?? 15_000;
  const target = importTarget(
    directionResultPath,
    options.sourceWorkDir ? resolve(options.sourceWorkDir) : undefined,
  );
  const experimentPolicyPath = resolve(projectRoot, EXPERIMENT_POLICY_FILE);
  if (!existsSync(experimentPolicyPath)) {
    throw new Error(`missing Direction Experiment policy: ${EXPERIMENT_POLICY_FILE}`);
  }

  const store = new DirectionExperimentStore(workDir);
  store.initializeLayout();
  const sharedCacheRoot = resolve(projectRoot, "experiment_cache");
  for (const directory of ["models", "data", "environments"]) {
    mkdirSync(resolve(sharedCacheRoot, directory), { recursive: true });
  }
  store.writeJson("inputs/direction_result.json", target.direction);
  store.writeJson("inputs/parent_anchor_result.json", target.anchor);
  store.writeJson("inputs/source_review_result.json", target.review);
  store.writeJson("inputs/source_run.json", target.sourceRun);
  store.writeJson("inputs/evidence_manifest.json", {
    generatedAt: new Date().toISOString(),
    items: target.evidence,
  });
  store.writeText("inputs/direction_target.md", target.targetMarkdown);
  store.writeText(
    "inputs/experiment_policy.md",
    readFileSync(experimentPolicyPath, "utf8"),
  );
  const skills = snapshotSkills(store, projectRoot);
  store.writeText(
    "workspace/direction/README.md",
    [
      "# Frozen Direction authority",
      "",
      "Do not edit the input bundle. Read these authoritative files:",
      "",
      ...[
        "inputs/direction_result.json",
        "inputs/parent_anchor_result.json",
        "inputs/source_review_result.json",
        "inputs/direction_target.md",
        "inputs/evidence_manifest.json",
        "inputs/experiment_policy.md",
        "inputs/source_run.json",
      ].map((path) => `- ${store.absolute(path)}`),
      "",
    ].join("\n"),
  );

  const inputs: DirectionInputPins = {
    directionResult: inputPin(store, "direction_result", "inputs/direction_result.json"),
    parentAnchorResult: inputPin(
      store,
      "parent_anchor_result",
      "inputs/parent_anchor_result.json",
    ),
    sourceReviewResult: inputPin(
      store,
      "source_review_result",
      "inputs/source_review_result.json",
    ),
    directionTarget: inputPin(
      store,
      "direction_target",
      "inputs/direction_target.md",
    ),
    evidenceManifest: inputPin(
      store,
      "evidence_manifest",
      "inputs/evidence_manifest.json",
    ),
    experimentPolicy: inputPin(
      store,
      "experiment_policy",
      "inputs/experiment_policy.md",
    ),
    sourceRun: inputPin(store, "source_run", "inputs/source_run.json"),
  };
  const labHardTimeoutMs = options.labHardTimeoutMs ?? 21_600_000;
  const labResultReserveMs = Math.min(
    900_000,
    Math.max(60_000, Math.floor(labHardTimeoutMs * 0.1)),
  );
  const run: DirectionRunFile = {
    formatVersion: DIRECTION_EXPERIMENT_FORMAT_VERSION,
    workflow: "DIRECTION_EXPERIMENT_LOOP",
    runId: `direction-run-${randomUUID()}`,
    createdAt: new Date().toISOString(),
    projectRoot,
    model: options.model ?? "gpt-5.6-sol",
    source: target.binding,
    inputs,
    skills,
    storage: {
      sharedCacheRoot,
      cycleRoot: "workspace/cycles",
    },
    budgets: {
      initialLabCycles: maxCycles,
      maxOutputRetries: 2,
      maxRuntimeRetries: 2,
      decision: {
        idleTimeoutMs: options.decisionIdleTimeoutMs ?? 300_000,
        hardTimeoutMs: options.decisionHardTimeoutMs ?? 900_000,
        interruptGraceMs,
      },
      judge: {
        idleTimeoutMs: options.judgeIdleTimeoutMs ?? 300_000,
        hardTimeoutMs: options.judgeHardTimeoutMs ?? 900_000,
        interruptGraceMs,
      },
      lab: {
        idleTimeoutMs: options.labIdleTimeoutMs ?? 900_000,
        hardTimeoutMs: labHardTimeoutMs,
        interruptGraceMs,
      },
      labResultReserveMs,
      controlPollMs: 500,
      maxInputTokens: 96_000,
      maxOutputTokens: 24_000,
    },
  };
  store.writeJson("run.json", run);
  const state: DirectionStateFile = {
    formatVersion: DIRECTION_EXPERIMENT_FORMAT_VERSION,
    revision: 1,
    lifecycle: "RUNNING",
    node: "DECISION",
    cycle: 0,
    authorizedLabCycles: maxCycles,
    transitions: 0,
    reason: null,
    pauseKind: null,
    activeContractRevision: 0,
    activeContractRef: null,
    activeContractHash: null,
    activeGoalRecordRef: null,
    activeLabInvocationRef: null,
    activeJudgeRequestRef: null,
    latestLabResultRef: null,
    latestCheckpointRef: null,
    latestJudgeRef: null,
    latestDecisionRef: null,
    finalDecision: null,
    evidenceScope: null,
  };
  store.writeState(state, "RUN_INITIALIZED");
  store.writeText("history.jsonl", "");
  return run;
}

export function verifyFrozenFiles(
  store: DirectionExperimentStore,
): string[] {
  const run = store.readRun();
  const errors: string[] = [];
  const requiredInputs: Array<keyof DirectionInputPins> = [
    "directionResult",
    "parentAnchorResult",
    "sourceReviewResult",
    "directionTarget",
    "evidenceManifest",
    "experimentPolicy",
    "sourceRun",
  ];
  for (const key of requiredInputs) {
    const pin = run.inputs?.[key];
    if (!pin) {
      errors.push(`missing frozen input pin: ${key}`);
      continue;
    }
    if (!store.exists(pin.path)) errors.push(`missing frozen input: ${pin.path}`);
    else if (store.sha256(pin.path) !== pin.sha256) {
      errors.push(`frozen input hash mismatch: ${pin.path}`);
    }
  }
  for (const pin of Object.values(run.skills ?? {})) {
    if (!store.exists(pin.path)) errors.push(`missing frozen Skill/Ref: ${pin.path}`);
    else if (store.sha256(pin.path) !== pin.sha256) {
      errors.push(`frozen Skill/Ref hash mismatch: ${pin.path}`);
    }
  }
  if (store.exists(run.inputs.evidenceManifest.path)) {
    const manifest = store.readJson<{ items?: EvidenceManifestItem[] }>(
      run.inputs.evidenceManifest.path,
    );
    for (const item of Array.isArray(manifest.items) ? manifest.items : []) {
      if (item.resolution !== "RESOLVED") continue;
      if (!item.resolvedPath || !existsSync(item.resolvedPath)) {
        errors.push(`resolved evidence is unavailable: ${item.sourceRef}`);
      } else if (item.sha256 && sha256External(item.resolvedPath) !== item.sha256) {
        errors.push(`resolved evidence hash mismatch: ${item.sourceRef}`);
      }
    }
  }
  return errors;
}

function importTarget(
  directionPath: string,
  explicitSourceWorkDir?: string,
): ImportedTarget {
  const sourceWorkDir = explicitSourceWorkDir ?? findSourceWorkDir(directionPath);
  const runPath = resolve(sourceWorkDir, "run.json");
  const indexPath = resolve(sourceWorkDir, "objects/index.json");
  if (!existsSync(runPath) || !existsSync(indexPath)) {
    throw new Error(`source workDir lacks run.json or objects/index.json: ${sourceWorkDir}`);
  }
  assertContained(sourceWorkDir, directionPath, "Direction result");
  const sourceRun = readJsonObject(runPath);
  const index = readJsonObject(indexPath);
  const directionRef = normalizeRef(relative(sourceWorkDir, directionPath));
  const match = findDirectionRevision(index, directionRef);
  const direction = readJsonObject(directionPath);
  const task = readOptionalObject(resolveSource(sourceWorkDir, match.workTaskRef));
  const taskInputs = asObject(task?.inputs);
  const boundAnchorRef = typeof taskInputs?.boundAnchor === "string"
    ? normalizeRef(taskInputs.boundAnchor)
    : null;
  const anchorMatch = findAnchorRevision(
    index,
    match.parentAnchorId,
    boundAnchorRef,
  );
  const anchorPath = resolveSource(sourceWorkDir, anchorMatch.workRef);
  const reviewPath = resolveSource(sourceWorkDir, match.reviewRef);
  const anchor = readJsonObject(anchorPath);
  const review = readJsonObject(reviewPath);
  const binding: DirectionSourceBinding = {
    sourceWorkDir,
    sourceRunId: stringValue(sourceRun.runId, "NOT_DECLARED"),
    directionId: match.directionId,
    directionRevision: match.revision,
    directionWorkRef: directionRef,
    directionReviewRef: match.reviewRef,
    parentAnchorId: match.parentAnchorId,
    parentAnchorRevision: anchorMatch.revision,
    parentAnchorWorkRef: anchorMatch.workRef,
  };
  const projectRoot = typeof sourceRun.projectRoot === "string"
    ? resolve(sourceRun.projectRoot)
    : sourceWorkDir;
  const evidence = buildEvidenceManifest(projectRoot, anchor, direction);
  return {
    binding,
    sourceRun,
    direction,
    anchor,
    review,
    evidence,
    targetMarkdown: renderDirectionTarget(binding, direction, anchor, review, evidence),
  };
}

function findSourceWorkDir(path: string): string {
  let current = dirname(path);
  while (true) {
    if (
      existsSync(resolve(current, "run.json")) &&
      existsSync(resolve(current, "objects/index.json"))
    ) return current;
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  throw new Error(
    `cannot find source Learning run above ${path}; pass --source-work-dir explicitly`,
  );
}

function findDirectionRevision(index: Record<string, unknown>, workRef: string) {
  const directions = asObject(index.directions) ?? {};
  const matches: Array<{
    directionId: string;
    revision: number;
    parentAnchorId: string;
    workTaskRef: string;
    reviewRef: string;
  }> = [];
  for (const [directionId, rawDirection] of Object.entries(directions)) {
    const direction = asObject(rawDirection);
    if (!direction) continue;
    const parentAnchorId = stringValue(direction.parentAnchorId, "");
    const revisions = asObject(direction.revisions) ?? {};
    for (const [rawRevision, rawValue] of Object.entries(revisions)) {
      const value = asObject(rawValue);
      if (!value || normalizeRef(stringValue(value.workRef, "")) !== workRef) continue;
      matches.push({
        directionId,
        revision: Number(value.revision ?? rawRevision),
        parentAnchorId,
        workTaskRef: stringValue(value.workTaskRef, ""),
        reviewRef: normalizeRef(stringValue(value.reviewRef, "")),
      });
    }
  }
  if (matches.length !== 1) {
    throw new Error(
      `Direction result must match exactly one objects/index.json revision; matched ${matches.length}: ${workRef}`,
    );
  }
  const match = matches[0]!;
  if (!match.parentAnchorId || !match.reviewRef) {
    throw new Error("Direction index entry lacks parent Anchor or review reference");
  }
  return match;
}

function findAnchorRevision(
  index: Record<string, unknown>,
  anchorId: string,
  preferredWorkRef: string | null,
) {
  const anchors = asObject(index.anchors) ?? {};
  const anchor = asObject(anchors[anchorId]);
  if (!anchor) throw new Error(`parent Anchor missing from object index: ${anchorId}`);
  const revisions = asObject(anchor.revisions) ?? {};
  const candidates = Object.entries(revisions).map(([rawRevision, rawValue]) => {
    const value = asObject(rawValue) ?? {};
    return {
      revision: Number(value.revision ?? rawRevision),
      workRef: normalizeRef(stringValue(value.workRef, "")),
    };
  }).filter((item) => item.workRef);
  if (preferredWorkRef) {
    const exact = candidates.find((item) => item.workRef === preferredWorkRef);
    if (exact) return exact;
  }
  const latest = Number(anchor.latestRevision);
  const fallback = candidates.find((item) => item.revision === latest);
  if (!fallback) throw new Error(`parent Anchor has no resolvable revision: ${anchorId}`);
  return fallback;
}

function buildEvidenceManifest(
  projectRoot: string,
  anchor: Record<string, unknown>,
  direction: Record<string, unknown>,
): EvidenceManifestItem[] {
  const items: EvidenceManifestItem[] = [];
  for (const [owner, result] of [
    ["ANCHOR", anchor],
    ["DIRECTION", direction],
  ] as const) {
    for (const raw of Array.isArray(result.evidence) ? result.evidence : []) {
      const evidence = asObject(raw) ?? {};
      const sourceRef = stringValue(evidence.sourceRef, "NOT_DECLARED");
      const supports = stringValue(evidence.supports, "NOT_DECLARED");
      const { pathPart, sourceUnit } = splitSourceRef(sourceRef);
      let resolvedPath: string | null = null;
      let sha256: string | null = null;
      if (pathPart && !/^[a-z]+:\/\//i.test(pathPart)) {
        const candidate = resolve(isAbsolute(pathPart) ? pathPart : resolve(projectRoot, pathPart));
        if (existsSync(candidate) && statSync(candidate).isFile()) {
          resolvedPath = candidate;
          sha256 = sha256External(candidate);
        }
      }
      items.push({
        owner,
        sourceRef,
        supports,
        resolvedPath,
        sourceUnit,
        sha256,
        resolution: resolvedPath ? "RESOLVED" : "UNRESOLVED",
      });
    }
  }
  return items;
}

export function renderDirectionTarget(
  binding: DirectionSourceBinding,
  direction: Record<string, unknown>,
  anchor: Record<string, unknown>,
  review: Record<string, unknown>,
  evidence: EvidenceManifestItem[],
): string {
  const directionContent = asObject(direction.content) ?? {};
  const anchorContent = asObject(anchor.content) ?? {};
  const scope = asObject(anchorContent.scope6L) ?? {};
  const sections = [
    "# Frozen Direction Target",
    "",
    "Original JSON files are authoritative. This file is a mechanical projection; missing fields are `NOT_DECLARED`.",
    "",
    "## Identity",
    line("Direction name", directionContent.name),
    line("Direction id", binding.directionId),
    line("Direction revision", binding.directionRevision),
    line("Source ref", binding.directionWorkRef),
    line("Parent Anchor", `${binding.parentAnchorId} revision ${binding.parentAnchorRevision}`),
    "",
    "## Optimization direction",
    block("Target performance problem", anchorContent.performanceTension),
    block("Proposed mechanism", directionContent.mechanism),
    block("Target metrics and applicable conditions", directionContent.expectedEffects),
    "",
    "## Baselines and frozen change",
    block("Parent execution baseline", anchorContent.baseline),
    block(
      "Closest/strongest baseline and unique change",
      directionContent.baselineChange,
    ),
    block("Declared invariants", directionContent.invariants),
    block("Declared frozen variables and interfaces", directionContent.frozenInterfaces),
    "",
    "## Optimization layers and concrete objects",
    ...["L1", "L2", "L3", "L4", "L5", "L6"].flatMap((layer) => [
      `### ${layer}`,
      formatValue(scope[layer]),
      "",
    ]),
    block("Cross-layer interfaces", directionContent.crossLayerInterfaces),
    block("Execution events", directionContent.executionEvents),
    "",
    "## Evidence-backed headroom and causal hypothesis",
    block("Observed performance tension / headroom", anchorContent.performanceTension),
    block("Direction hypothesis", directionContent.mechanism),
    "",
    "## Expected effects, constraints and tradeoffs",
    block("Expected effects", directionContent.expectedEffects),
    block("Parent constraints", anchorContent.constraints),
    block("Direction tradeoffs", directionContent.tradeoffs),
    "",
    "## Failure conditions and counterexamples",
    block("Failure conditions", directionContent.failureConditions),
    "",
    "## Measurement and ablation plan",
    block("Measurement plan", directionContent.measurementPlan),
    "",
    "## Source review",
    line("Verdict", review.reviewVerdict),
    block("Summary", review.summary),
    block("Findings", review.findings),
    block("Query gaps", review.queryGaps),
    "",
    "## Declared evidence",
    ...(evidence.length > 0
      ? evidence.map((item) =>
        `- [${item.owner}] ${item.sourceRef} — ${item.supports} (${item.resolution})`
      )
      : ["NOT_DECLARED"]),
    "",
  ];
  return sections.join("\n");
}

function line(label: string, value: unknown): string {
  return `- ${label}: ${inlineValue(value)}`;
}

function block(label: string, value: unknown): string {
  return [`### ${label}`, formatValue(value)].join("\n\n");
}

function inlineValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "NOT_DECLARED";
  if (["string", "number", "boolean"].includes(typeof value)) return String(value);
  return `see structured value below`;
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "NOT_DECLARED";
  if (typeof value === "string") return value;
  return `\n\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``;
}

function inputPin(store: DirectionExperimentStore, name: string, path: string): FilePin {
  return { name, path, sha256: store.sha256(path) };
}

function snapshotSkills(
  store: DirectionExperimentStore,
  projectRoot: string,
): DirectionRunFile["skills"] {
  const result = {} as DirectionRunFile["skills"];
  for (const key of Object.keys(SKILL_FILES) as Array<keyof typeof SKILL_FILES>) {
    const source = resolve(projectRoot, SKILL_FILES[key]);
    if (!existsSync(source)) {
      throw new Error(`missing required Skill/Ref: ${SKILL_FILES[key]}`);
    }
    const destination = FROZEN_SKILL_FILES[key];
    store.writeText(destination, readFileSync(source, "utf8"));
    result[key] = {
      name: `frozen-${key}`,
      path: destination,
      sha256: store.sha256(destination),
    };
  }
  return result;
}

function resolveSource(root: string, ref: string): string {
  const path = resolve(root, ref);
  assertContained(root, path, "source reference");
  if (!existsSync(path)) throw new Error(`source reference does not exist: ${ref}`);
  return path;
}

function assertContained(root: string, path: string, label: string): void {
  const normalizedRoot = resolve(root);
  const normalizedPath = resolve(path);
  if (
    normalizedPath !== normalizedRoot &&
    !normalizedPath.startsWith(`${normalizedRoot}${sep}`)
  ) throw new Error(`${label} escapes source workDir: ${path}`);
}

function readJsonObject(path: string): Record<string, unknown> {
  const value = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`expected JSON object: ${path}`);
  }
  return value as Record<string, unknown>;
}

function readOptionalObject(path: string): Record<string, unknown> | null {
  return existsSync(path) ? readJsonObject(path) : null;
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function normalizeRef(value: string): string {
  return value.split(sep).join("/").replace(/^\.\//, "");
}

function splitSourceRef(sourceRef: string): {
  pathPart: string;
  sourceUnit: string | null;
} {
  const index = sourceRef.indexOf("#");
  return index < 0
    ? { pathPart: sourceRef, sourceUnit: null }
    : {
      pathPart: sourceRef.slice(0, index),
      sourceUnit: sourceRef.slice(index + 1) || null,
    };
}

function sha256External(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
}
