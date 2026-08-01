import type {
  Anchor,
  Direction,
  KnowledgeDimension,
  RegisteredRole,
  SearchIntent,
  SearchNeed,
  StageContractDraft,
  TopicFrame,
  TurnBudget,
  ValidationReport,
} from "../contracts/index.ts";
import {
  ALLOWED_EVIDENCE_TOOLS,
  validateNoExperimentStage,
} from "../security/no_experiment_guard.ts";
import {
  addError,
  emptyReport,
  mergeReports,
} from "./schema_validator.ts";
import { canonicalEqual } from "../contracts/index.ts";

export const INTENT_DIMENSION_ROUTES: Readonly<
  Record<
    SearchIntent,
    {
      primary: KnowledgeDimension;
      auxiliary: readonly KnowledgeDimension[];
    }
  >
> = Object.freeze({
  discover_anchor: { primary: "idea", auxiliary: ["human"] },
  define_baseline: { primary: "idea", auxiliary: ["experiment"] },
  find_modification: { primary: "idea", auxiliary: ["knowledge"] },
  explain_mechanism: { primary: "knowledge", auxiliary: ["idea"] },
  find_implementation: { primary: "experiment", auxiliary: ["knowledge"] },
  design_measurement: { primary: "experiment", auxiliary: ["knowledge"] },
  challenge_direction: {
    primary: "knowledge",
    auxiliary: ["experiment", "human"],
  },
  verify_primary_source: { primary: "paper", auxiliary: [] },
});

export const DIMENSION_PATHS: Readonly<Record<KnowledgeDimension, string>> =
  Object.freeze({
    idea: "idea_notes/",
    knowledge: "knowledge_notes/",
    experiment: "experiment_notes/",
    human: "human_notes/",
    paper: "paper_secs/",
  });

export function validateTopicFrame(topic: TopicFrame): ValidationReport {
  const report = emptyReport();
  if (
    topic.workloads.length === 0 ||
    topic.phases.length === 0 ||
    topic.regimes.length === 0 ||
    topic.stackScope.length === 0 ||
    topic.layerScope.length === 0 ||
    topic.targetMetrics.length === 0
  ) {
    addError(
      report,
      "domain.topic_scope_incomplete",
      "/",
      "TopicFrame must preserve explicit workload, phase, regime, stack, layer, and metric scope",
    );
  }
  if (new Set(topic.layerScope).size !== topic.layerScope.length) {
    addError(
      report,
      "domain.topic_layer_scope_duplicate",
      "/layerScope",
      "TopicFrame layerScope must be a unique L1-L6 subset",
    );
  }
  topic.scopeAudit.changes.forEach((change, index) => {
    if (change.changeType === "narrow" && !change.userAuthorized) {
      addError(
        report,
        "domain.unauthorized_scope_narrowing",
        `/scopeAudit/changes/${index}/userAuthorized`,
        "scope narrowing requires explicit user authorization",
      );
    }
  });
  return report;
}

export function validateAnchor(anchor: Anchor): ValidationReport {
  const report = emptyReport();
  const invariantParts = [
    anchor.scenario.workload,
    anchor.scenario.phase,
    anchor.scenario.regime,
    ...anchor.scenario.stack,
    anchor.baseline.name,
    ...anchor.baseline.executionPath,
    ...anchor.baseline.comparisonScope,
    anchor.performanceTension.symptom,
    anchor.performanceTension.suspectedMechanism,
    ...anchor.performanceTension.bottleneckResources,
    ...anchor.performanceTension.targetMetrics,
  ];
  if (invariantParts.some((part) => part.trim() === "")) {
    addError(
      report,
      "domain.anchor_invariant_incomplete",
      "/",
      "Anchor requires scenario boundary, baseline execution path, and performance tension",
    );
  }
  if (anchor.status === "saturated" && !anchor.saturationReason?.trim()) {
    addError(
      report,
      "domain.anchor_saturation_reason_missing",
      "/saturationReason",
      "saturated Anchor requires a saturation reason",
    );
  }
  if (anchor.status === "rejected" && !anchor.statusReason.trim()) {
    addError(
      report,
      "domain.anchor_rejection_reason_missing",
      "/statusReason",
      "rejected Anchor requires a status reason",
    );
  }
  return report;
}

export function validateDirection(
  direction: Direction,
  topic?: TopicFrame,
): ValidationReport {
  const report = emptyReport();
  if (!direction.changes.some((change) => change.role === "primary")) {
    addError(
      report,
      "domain.direction_primary_change_missing",
      "/changes",
      "Direction requires at least one primary ModificationAtom",
    );
  }
  if (new Set(direction.changes.map((change) => change.atomId)).size !==
      direction.changes.length) {
    addError(
      report,
      "domain.direction_duplicate_atom",
      "/changes",
      "Direction ModificationAtom IDs must be unique",
    );
  }
  if (topic) {
    const allowedLayers = new Set(topic.layerScope);
    direction.changes.forEach((change, index) => {
      if (!allowedLayers.has(change.layer)) {
        addError(
          report,
          "domain.direction_layer_outside_topic",
          `/changes/${index}/layer`,
          `Direction layer ${change.layer} is outside Topic layerScope`,
        );
      }
    });
  }
  if (direction.causalLinks.length === 0 || direction.expectedEffects.length === 0) {
    addError(
      report,
      "domain.direction_causal_or_effect_missing",
      "/",
      "Direction requires a causal chain and expected metric effect",
    );
  }
  if (
    ["testable", "experiment_required", "rejected"].includes(direction.status) &&
    !direction.statusReason.trim()
  ) {
    addError(
      report,
      "domain.direction_terminal_reason_missing",
      "/statusReason",
      "terminal Direction requires a status reason",
    );
  }
  if (
    direction.status === "experiment_required" &&
    !direction.experimentHandoffId
  ) {
    addError(
      report,
      "domain.direction_handoff_missing",
      "/experimentHandoffId",
      "experiment_required Direction requires a handoff ID",
    );
  }
  if (
    direction.status !== "experiment_required" &&
    direction.experimentHandoffId !== null
  ) {
    addError(
      report,
      "domain.direction_handoff_for_non_experiment",
      "/experimentHandoffId",
      "only experiment_required Direction may reference a handoff",
    );
  }
  return report;
}

const OPEN_ENDED_TASK = /(?:继续研究|更多资料|search\s+more|continue\s+research)/i;

export function validateSearchNeed(need: SearchNeed): ValidationReport {
  const report = emptyReport();
  const route = INTENT_DIMENSION_ROUTES[need.intent];
  if (need.primaryDimension !== route.primary) {
    addError(
      report,
      "domain.search_route_primary",
      "/primaryDimension",
      `${need.intent} requires primary dimension ${route.primary}`,
    );
  }
  if (
    need.auxiliaryDimension !== null &&
    !route.auxiliary.includes(need.auxiliaryDimension)
  ) {
    addError(
      report,
      "domain.search_route_auxiliary",
      "/auxiliaryDimension",
      `${need.auxiliaryDimension} is not an admitted auxiliary dimension for ${need.intent}`,
    );
  }
  const expected = [
    need.primaryDimension,
    ...(need.auxiliaryDimension ? [need.auxiliaryDimension] : []),
  ];
  if (!canonicalEqual(need.targetDimensions, expected)) {
    addError(
      report,
      "domain.target_dimensions",
      "/targetDimensions",
      "targetDimensions must exactly equal primary plus the optional auxiliary dimension",
    );
  }
  if (OPEN_ENDED_TASK.test(need.question)) {
    addError(
      report,
      "domain.open_ended_search_need",
      "/question",
      "SearchNeed must ask one bounded question",
    );
  }
  if (need.successCriteria.length === 0) {
    addError(
      report,
      "domain.search_success_criteria_missing",
      "/successCriteria",
      "SearchNeed requires at least one success criterion",
    );
  }
  return report;
}

export function validateTurnBudget(
  role: RegisteredRole | null,
  budget: TurnBudget,
): ValidationReport {
  const report = emptyReport();
  for (const [field, value] of Object.entries(budget)) {
    if (field !== "evidenceRead" && (!Number.isInteger(value) || value < 0)) {
      addError(
        report,
        "budget.invalid_integer",
        `/${field}`,
        "budget values must be non-negative integers",
      );
    }
  }
  if (
    budget.timeoutMs < 1 ||
    budget.maxInputTokens < 1 ||
    budget.maxOutputTokens < 1
  ) {
    addError(
      report,
      "budget.nonpositive_core_limit",
      "/",
      "timeout and token limits must be positive",
    );
  }
  if (role === "evidence_reader") {
    const evidence = budget.evidenceRead;
    if (budget.maxToolCalls <= 0 || evidence === null) {
      addError(
        report,
        "budget.evidence_tools_required",
        "/",
        "evidence_reader requires positive tool budget and evidenceRead limits",
      );
      return report;
    }
    if (
      evidence.maxLogicalQueries < 1 ||
      evidence.maxLogicalQueries > 3 ||
      evidence.maxSearchToolCalls < evidence.maxLogicalQueries ||
      evidence.maxSearchToolCalls > budget.maxToolCalls ||
      evidence.maxHitsConsidered < 1 ||
      evidence.maxHitsConsidered > 50 ||
      evidence.maxSelectedSources < 1 ||
      evidence.maxSelectedSources > evidence.maxHitsConsidered ||
      evidence.maxContextsRead < evidence.maxSelectedSources
    ) {
      addError(
        report,
        "budget.evidence_relationship",
        "/evidenceRead",
        "Evidence Q1-Q3, search-call, hit, selected-source, context, and tool limits are inconsistent",
      );
    }
  } else if (budget.maxToolCalls !== 0 || budget.evidenceRead !== null) {
    addError(
      report,
      "budget.zero_tool_role",
      "/",
      "workflow, reviewers, and script transitions require maxToolCalls=0 and evidenceRead=null",
    );
  }
  return report;
}

export function validateStageContractDraft(
  stage: StageContractDraft,
): ValidationReport {
  const report = emptyReport();
  const stageRegistry = (
    awaitableStageRegistry as Record<
      string,
      {
        executionKind: string;
        role: RegisteredRole | null;
        output: string | null;
      }
    >
  )[stage.stageType];
  if (!stageRegistry) {
    addError(
      report,
      "stage.unknown_type",
      "/stageType",
      "Stage type is not registered",
    );
    return report;
  }
  if (
    stage.executionKind !== stageRegistry.executionKind ||
    stage.role !== stageRegistry.role ||
    stage.expectedOutputMessageType !== stageRegistry.output
  ) {
    addError(
      report,
      "stage.registry_mismatch",
      "/",
      "executionKind, role, and expected output must exactly match the Stage registry",
    );
  }
  if (stage.role === "evidence_reader") {
    const allowed = new Set<string>(ALLOWED_EVIDENCE_TOOLS);
    if (
      stage.requestedTools.length !== ALLOWED_EVIDENCE_TOOLS.length ||
      stage.requestedTools.some((tool) => !allowed.has(tool))
    ) {
      addError(
        report,
        "stage.evidence_tool_set",
        "/requestedTools",
        "Evidence stage requires exactly the registered read-only Obsidian tools",
      );
    }
  } else if (stage.requestedTools.length > 0) {
    addError(
      report,
      "stage.zero_tool_role",
      "/requestedTools",
      "this Stage is not permitted to request tools",
    );
  }
  return mergeReports(
    report,
    validateTurnBudget(stage.role, stage.budget),
    validateNoExperimentStage(stage),
  );
}

// Avoid a value-level cycle through contracts/index.ts while keeping registry
// values immutable in one source. Node initializes this after module linking.
import { STAGE_REGISTRY as awaitableStageRegistry } from "../contracts/registries.ts";

export function validateBudgetEquality(
  taskBudget: TurnBudget,
  frozenBudget: TurnBudget,
): ValidationReport {
  const report = emptyReport();
  if (!canonicalEqual(taskBudget, frozenBudget)) {
    addError(
      report,
      "budget.frozen_contract_mismatch",
      "/budget",
      "task budget must be canonical-equal to the frozen StageContract budget",
    );
  }
  return report;
}
