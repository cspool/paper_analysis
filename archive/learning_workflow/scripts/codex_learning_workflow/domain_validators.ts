import {
  anchorId as makeAnchorId,
  normalizeAnchorSignature,
  normalizeIdentityText,
  stableHash,
  stableId,
} from "./stable_ids.ts";
import { REVIEW_DIMENSIONS } from "./role_profiles.ts";
import type {
  Anchor,
  AnchorSignature,
  Baseline,
  BaselineKind,
  Confidence,
  CrossLayerEdge,
  Direction,
  EdgeRelation,
  EntryRole,
  ExpertReview,
  GlobalEntity,
  JsonValue,
  Layer,
  LayerEntry,
  ReviewDimension,
  ReviewQuestionAnswer,
  RunState,
} from "./types.ts";

function asObject(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value.trim() : fallback;
}

function stringArray(value: unknown): string[] {
  return [...new Set(arrayValue(value).map((item) => stringValue(item)).filter(Boolean))];
}

function snakeOrCamel(object: Record<string, unknown>, snake: string, camel: string): unknown {
  return object[snake] ?? object[camel];
}

function claimRefs(value: unknown, knownClaims: Set<string>): string[] {
  return stringArray(value).filter((id) => knownClaims.has(id)).sort();
}

function confidence(value: unknown): Confidence {
  const candidate = stringValue(value, "middle");
  return candidate === "low" || candidate === "high" ? candidate : "middle";
}

function baselineKind(value: unknown): BaselineKind {
  const candidate = stringValue(value);
  const aliases: Record<string, BaselineKind> = {
    current_practice: "current_practice",
    current: "current_practice",
    b0: "current_practice",
    strong_comparison: "strong_comparison",
    strong: "strong_comparison",
    b1: "strong_comparison",
    tool_evaluation: "tool_evaluation",
    evaluation: "tool_evaluation",
    tool: "tool_evaluation",
    b2: "tool_evaluation",
    reusable_implementation: "reusable_implementation",
    implementation: "reusable_implementation",
    b3: "reusable_implementation",
  };
  return aliases[candidate.toLowerCase()] ?? "current_practice";
}

function entryRole(value: unknown): EntryRole {
  const candidate = stringValue(value);
  const allowed = new Set<EntryRole>([
    "baseline_behavior",
    "opportunity",
    "method",
    "implementation",
    "constraint",
    "evaluation",
  ]);
  return allowed.has(candidate as EntryRole) ? candidate as EntryRole : "opportunity";
}

function layer(value: unknown): Layer | null {
  const candidate = stringValue(value).toUpperCase();
  return /^L[1-6]$/.test(candidate) ? candidate as Layer : null;
}

function edgeRelation(value: unknown): EdgeRelation {
  const candidate = stringValue(value);
  const allowed = new Set<EdgeRelation>([
    "controls",
    "depends_on",
    "enables",
    "complements",
    "conflicts",
    "substitutes",
    "incompatible",
  ]);
  return allowed.has(candidate as EdgeRelation) ? candidate as EdgeRelation : "depends_on";
}

function signatureFromCandidate(object: Record<string, unknown>): AnchorSignature | null {
  const signature = asObject(object.signature) ?? object;
  const targetMetrics = stringArray(snakeOrCamel(signature, "target_metrics", "targetMetrics"));
  const parsed: AnchorSignature = {
    workload: stringValue(signature.workload),
    phase: stringValue(signature.phase),
    regime: stringValue(signature.regime),
    backend: stringValue(signature.backend),
    bottleneck: stringValue(signature.bottleneck),
    primaryBaselineExecutionPath: stringValue(
      snakeOrCamel(signature, "primary_baseline_execution_path", "primaryBaselineExecutionPath")
        ?? snakeOrCamel(signature, "primary_baseline", "primaryBaseline"),
    ),
    targetMetrics,
  };
  if (
    !parsed.workload
    || !parsed.phase
    || !parsed.regime
    || !parsed.backend
    || !parsed.bottleneck
    || !parsed.primaryBaselineExecutionPath
    || parsed.targetMetrics.length === 0
  ) {
    return null;
  }
  return normalizeAnchorSignature(parsed);
}

function mergeStringArrays(left: string[], right: string[]): string[] {
  return [...new Set([...left, ...right])].sort();
}

function upsertById<T>(items: T[], id: string, idOf: (item: T) => string, value: T, merge: (existing: T, incoming: T) => T): void {
  const index = items.findIndex((item) => idOf(item) === id);
  if (index < 0) {
    items.push(value);
  } else {
    items[index] = merge(items[index], value);
  }
}

function anchorPriority(state: RunState, anchor: Anchor): number[] {
  const evidence = new Set(anchor.evidenceRefs);
  const claims = state.claims.filter((claim) => evidence.has(claim.claimId));
  const entries = state.entries.filter((entry) => entry.anchorId === anchor.anchorId);
  const baselines = state.baselines.filter((baseline) => baseline.anchorId === anchor.anchorId);
  return [
    claims.filter((claim) => /scenario|opportunity|bottleneck|acceleration/i.test(claim.claimType)).length,
    entries.filter((entry) => entry.role === "opportunity" && Boolean(entry.modifiableObject)).length,
    evidence.size,
    baselines.length,
    entries.filter((entry) => entry.role === "implementation").length,
    entries.filter((entry) => entry.role === "method").length,
  ];
}

function comparePriority(left: Anchor, right: Anchor): number {
  const length = Math.max(left.priority.length, right.priority.length);
  for (let index = 0; index < length; index += 1) {
    const difference = (right.priority[index] ?? 0) - (left.priority[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return left.anchorId.localeCompare(right.anchorId);
}

export interface CuratorMergeResult {
  newAcceptedAnchorIds: string[];
  touchedAnchorIds: string[];
  rejectedObjects: Array<{ kind: string; reason: string; candidate: JsonValue }>;
}

export function mergeCuratorDelta(
  state: RunState,
  payload: JsonValue,
  round: number,
): CuratorMergeResult {
  const root = asObject(payload);
  if (!root) throw new Error("curator payload must be an object");
  const knownClaims = new Set(state.claims.map((claim) => claim.claimId));
  const previouslyAccepted = new Set(state.anchors.filter((anchor) => anchor.status === "accepted").map((anchor) => anchor.anchorId));
  const touched = new Set<string>();
  const rejectedObjects: CuratorMergeResult["rejectedObjects"] = [];

  for (const rawEntity of arrayValue(root.entities)) {
    const candidate = asObject(rawEntity);
    if (!candidate) continue;
    const name = stringValue(candidate.name);
    const kindRaw = stringValue(candidate.kind, "other");
    const kind = ["method", "implementation", "tool", "software", "hardware", "other"].includes(kindRaw)
      ? kindRaw as GlobalEntity["kind"]
      : "other";
    const evidenceRefs = claimRefs(snakeOrCamel(candidate, "evidence_refs", "evidenceRefs"), knownClaims);
    if (!name || evidenceRefs.length === 0) {
      rejectedObjects.push({ kind: "entity", reason: "missing name or verified evidence", candidate: rawEntity as JsonValue });
      continue;
    }
    const entityId = stableId("G", [kind, normalizeIdentityText(name)], 14);
    const entity: GlobalEntity = {
      entityId,
      kind,
      name,
      description: stringValue(candidate.description),
      evidenceRefs,
    };
    upsertById(state.entities, entityId, (item) => item.entityId, entity, (existing, incoming) => ({
      ...existing,
      description: existing.description || incoming.description,
      evidenceRefs: mergeStringArrays(existing.evidenceRefs, incoming.evidenceRefs),
    }));
  }

  for (const rawAnchor of arrayValue(root.anchors)) {
    const candidate = asObject(rawAnchor);
    if (!candidate) {
      rejectedObjects.push({ kind: "anchor", reason: "not an object", candidate: rawAnchor as JsonValue });
      continue;
    }
    const signature = signatureFromCandidate(candidate);
    const evidenceRefs = claimRefs(snakeOrCamel(candidate, "evidence_refs", "evidenceRefs"), knownClaims);
    if (!signature || evidenceRefs.length === 0) {
      rejectedObjects.push({
        kind: "anchor",
        reason: !signature ? "incomplete anchor signature" : "anchor has no verified evidence",
        candidate: rawAnchor as JsonValue,
      });
      continue;
    }
    const anchorId = makeAnchorId(signature);
    touched.add(anchorId);
    const existingAnchor = state.anchors.find((anchor) => anchor.anchorId === anchorId);
    const localBaselineMap = new Map<string, string>();
    const localEntryMap = new Map<string, string>();
    const baselineIds: string[] = [];
    const entryIds: string[] = [];
    const edgeIds: string[] = [];

    for (const rawBaseline of arrayValue(candidate.baselines)) {
      const baselineCandidate = asObject(rawBaseline);
      if (!baselineCandidate) continue;
      const name = stringValue(baselineCandidate.name);
      const executionPath = stringValue(snakeOrCamel(baselineCandidate, "execution_path", "executionPath"));
      const baselineEvidence = claimRefs(
        snakeOrCamel(baselineCandidate, "evidence_refs", "evidenceRefs"),
        knownClaims,
      );
      if (!name || !executionPath || baselineEvidence.length === 0) {
        rejectedObjects.push({
          kind: "baseline",
          reason: "missing name, execution_path, or verified evidence",
          candidate: rawBaseline as JsonValue,
        });
        continue;
      }
      const kind = baselineKind(baselineCandidate.kind);
      const baselineId = stableId("B", [
        anchorId,
        kind,
        normalizeIdentityText(name),
        normalizeIdentityText(executionPath),
      ], 14);
      const baseline: Baseline = {
        baselineId,
        anchorId,
        kind,
        name,
        executionPath,
        implementation: stringValue(baselineCandidate.implementation),
        comparisonScope: stringValue(snakeOrCamel(baselineCandidate, "comparison_scope", "comparisonScope")),
        evidenceRefs: baselineEvidence,
        explorationValue: ["low", "middle", "high"].includes(stringValue(
          snakeOrCamel(baselineCandidate, "exploration_value", "explorationValue"),
        ))
          ? stringValue(snakeOrCamel(baselineCandidate, "exploration_value", "explorationValue")) as Baseline["explorationValue"]
          : "unknown",
        status: "active",
      };
      upsertById(state.baselines, baselineId, (item) => item.baselineId, baseline, (existing, incoming) => ({
        ...existing,
        implementation: existing.implementation || incoming.implementation,
        comparisonScope: existing.comparisonScope || incoming.comparisonScope,
        evidenceRefs: mergeStringArrays(existing.evidenceRefs, incoming.evidenceRefs),
      }));
      baselineIds.push(baselineId);
      const localId = stringValue(snakeOrCamel(baselineCandidate, "local_id", "localId"));
      if (localId) localBaselineMap.set(localId, baselineId);
      localBaselineMap.set(name, baselineId);
    }

    const anchorEntityMap = new Map<string, string>();
    for (const entity of state.entities) {
      anchorEntityMap.set(normalizeIdentityText(entity.name), entity.entityId);
    }
    for (const rawEntry of arrayValue(candidate.entries)) {
      const entryCandidate = asObject(rawEntry);
      if (!entryCandidate) continue;
      const parsedLayer = layer(entryCandidate.layer);
      const claim = stringValue(entryCandidate.claim);
      const modifiableObject = stringValue(
        snakeOrCamel(entryCandidate, "modifiable_object", "modifiableObject"),
      );
      const entryEvidence = claimRefs(
        snakeOrCamel(entryCandidate, "evidence_refs", "evidenceRefs"),
        knownClaims,
      );
      if (!parsedLayer || !claim || entryEvidence.length === 0) {
        rejectedObjects.push({
          kind: "entry",
          reason: "missing layer, atomic claim, or verified evidence",
          candidate: rawEntry as JsonValue,
        });
        continue;
      }
      const role = entryRole(entryCandidate.role);
      const entryId = stableId("E", [
        anchorId,
        parsedLayer,
        role,
        normalizeIdentityText(claim),
        normalizeIdentityText(modifiableObject),
      ], 16);
      const baselineReferences = stringArray(
        snakeOrCamel(entryCandidate, "applicable_baselines", "applicableBaselines"),
      )
        .map((reference) => localBaselineMap.get(reference) ?? reference)
        .filter((reference) => state.baselines.some((baseline) => baseline.baselineId === reference && baseline.anchorId === anchorId));
      const entityName = stringValue(snakeOrCamel(entryCandidate, "entity_name", "entityName"));
      const explicitEntityId = stringValue(snakeOrCamel(entryCandidate, "entity_id", "entityId"));
      const entityId = state.entities.some((entity) => entity.entityId === explicitEntityId)
        ? explicitEntityId
        : anchorEntityMap.get(normalizeIdentityText(entityName)) ?? null;
      const entry: LayerEntry = {
        entryId,
        entityId,
        anchorId,
        layer: parsedLayer,
        role,
        claim,
        modifiableObject,
        applicableBaselineIds: [...new Set(baselineReferences)].sort(),
        preconditions: stringArray(entryCandidate.preconditions),
        expectedEffect: stringValue(snakeOrCamel(entryCandidate, "expected_effect", "expectedEffect")),
        evidenceRefs: entryEvidence,
        confidence: confidence(entryCandidate.confidence),
        status: "active",
      };
      upsertById(state.entries, entryId, (item) => item.entryId, entry, (existing, incoming) => ({
        ...existing,
        entityId: existing.entityId ?? incoming.entityId,
        modifiableObject: existing.modifiableObject || incoming.modifiableObject,
        applicableBaselineIds: mergeStringArrays(existing.applicableBaselineIds, incoming.applicableBaselineIds),
        preconditions: mergeStringArrays(existing.preconditions, incoming.preconditions),
        expectedEffect: existing.expectedEffect || incoming.expectedEffect,
        evidenceRefs: mergeStringArrays(existing.evidenceRefs, incoming.evidenceRefs),
      }));
      entryIds.push(entryId);
      const localId = stringValue(snakeOrCamel(entryCandidate, "local_id", "localId"));
      if (localId) localEntryMap.set(localId, entryId);
      localEntryMap.set(entryId, entryId);
    }

    for (const rawEdge of arrayValue(candidate.edges)) {
      const edgeCandidate = asObject(rawEdge);
      if (!edgeCandidate) continue;
      const fromReference = stringValue(snakeOrCamel(edgeCandidate, "from_entry", "fromEntry"));
      const toReference = stringValue(snakeOrCamel(edgeCandidate, "to_entry", "toEntry"));
      const fromEntryId = localEntryMap.get(fromReference) ?? fromReference;
      const toEntryId = localEntryMap.get(toReference) ?? toReference;
      const relation = edgeRelation(edgeCandidate.relation);
      const compatibilityRaw = stringValue(edgeCandidate.compatibility);
      const compatibility = relation === "conflicts" || relation === "substitutes" || relation === "incompatible"
        ? "conflict"
        : compatibilityRaw === "conditional"
          ? "conditional"
          : "compatible";
      const condition = stringValue(edgeCandidate.condition);
      const interfaceText = stringValue(edgeCandidate.interface);
      const edgeEvidence = claimRefs(
        snakeOrCamel(edgeCandidate, "evidence_refs", "evidenceRefs"),
        knownClaims,
      );
      const endpointsValid = state.entries.some((entry) => entry.entryId === fromEntryId && entry.anchorId === anchorId)
        && state.entries.some((entry) => entry.entryId === toEntryId && entry.anchorId === anchorId)
        && fromEntryId !== toEntryId;
      if (!endpointsValid || !interfaceText || edgeEvidence.length === 0 || (compatibility === "conditional" && !condition)) {
        rejectedObjects.push({
          kind: "edge",
          reason: "invalid endpoints, missing interface/evidence, or conditional edge without condition",
          candidate: rawEdge as JsonValue,
        });
        continue;
      }
      const edgeId = stableId("X", [
        anchorId,
        fromEntryId,
        toEntryId,
        relation,
        normalizeIdentityText(interfaceText),
        normalizeIdentityText(condition),
      ], 16);
      const edge: CrossLayerEdge = {
        edgeId,
        anchorId,
        fromEntryId,
        toEntryId,
        relation,
        interface: interfaceText,
        compatibility,
        condition,
        evidenceRefs: edgeEvidence,
        confidence: confidence(edgeCandidate.confidence),
      };
      upsertById(state.edges, edgeId, (item) => item.edgeId, edge, (existing, incoming) => ({
        ...existing,
        evidenceRefs: mergeStringArrays(existing.evidenceRefs, incoming.evidenceRefs),
      }));
      edgeIds.push(edgeId);
    }

    const incoming: Anchor = {
      anchorId,
      title: stringValue(candidate.title) || `${signature.workload}: ${signature.bottleneck}`,
      scenario: stringValue(candidate.scenario),
      signature,
      evidenceRefs,
      baselineIds: [...new Set(baselineIds)].sort(),
      entryIds: [...new Set(entryIds)].sort(),
      edgeIds: [...new Set(edgeIds)].sort(),
      gaps: stringArray(candidate.gaps),
      status: "accepted",
      firstSeenRound: existingAnchor?.firstSeenRound ?? round,
      lastUpdatedRound: round,
      priority: [],
    };
    upsertById(state.anchors, anchorId, (item) => item.anchorId, incoming, (existing, fresh) => ({
      ...existing,
      title: existing.title || fresh.title,
      scenario: existing.scenario || fresh.scenario,
      evidenceRefs: mergeStringArrays(existing.evidenceRefs, fresh.evidenceRefs),
      baselineIds: mergeStringArrays(existing.baselineIds, fresh.baselineIds),
      entryIds: mergeStringArrays(existing.entryIds, fresh.entryIds),
      edgeIds: mergeStringArrays(existing.edgeIds, fresh.edgeIds),
      gaps: mergeStringArrays(existing.gaps, fresh.gaps),
      lastUpdatedRound: round,
    }));
  }

  for (const anchor of state.anchors) {
    anchor.priority = anchorPriority(state, anchor);
  }
  const ranked = [...state.anchors].sort(comparePriority);
  const acceptedIds = new Set(ranked.slice(0, state.config.maxAnchors).map((anchor) => anchor.anchorId));
  for (const anchor of state.anchors) {
    anchor.status = acceptedIds.has(anchor.anchorId) ? "accepted" : "rejected_cap";
  }
  for (const baseline of state.baselines) {
    baseline.status = baseline.anchorId && acceptedIds.has(baseline.anchorId) ? "active" : "reference_only";
  }
  state.stage1.acceptedAnchorIds = [...acceptedIds].sort();
  const newAcceptedAnchorIds = [...acceptedIds].filter((id) => !previouslyAccepted.has(id)).sort();
  return {
    newAcceptedAnchorIds,
    touchedAnchorIds: [...touched].sort(),
    rejectedObjects,
  };
}

function graphConnected(selectedEntries: string[], selectedEdges: CrossLayerEdge[]): boolean {
  if (selectedEntries.length <= 1) return true;
  const adjacency = new Map<string, Set<string>>();
  for (const entry of selectedEntries) adjacency.set(entry, new Set());
  for (const edge of selectedEdges) {
    adjacency.get(edge.fromEntryId)?.add(edge.toEntryId);
    adjacency.get(edge.toEntryId)?.add(edge.fromEntryId);
  }
  const visited = new Set<string>();
  const queue = [selectedEntries[0]];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);
    for (const neighbor of adjacency.get(current) ?? []) queue.push(neighbor);
  }
  return visited.size === selectedEntries.length;
}

export function validateDirectionProposal(
  state: RunState,
  anchorId: string,
  payload: JsonValue,
  proposalIndex: number,
): { direction: Direction | null; errors: string[] } {
  const object = asObject(payload);
  if (!object) return { direction: null, errors: ["direction payload is not an object"] };
  const errors: string[] = [];
  const anchor = state.anchors.find((item) => item.anchorId === anchorId && item.status === "accepted");
  if (!anchor) errors.push("anchor is not accepted");
  const selectedEntryIds = stringArray(snakeOrCamel(object, "selected_entry_ids", "selectedEntryIds")).sort();
  const selectedEdgeIds = stringArray(snakeOrCamel(object, "selected_edge_ids", "selectedEdgeIds")).sort();
  const baselineIds = stringArray(snakeOrCamel(object, "baseline_ids", "baselineIds")).sort();
  const hypothesis = stringValue(object.hypothesis);
  if (selectedEntryIds.length === 0) errors.push("direction selects no entry");
  if (!hypothesis) errors.push("direction hypothesis is empty");

  const selectedEntries = selectedEntryIds
    .map((id) => state.entries.find((entry) => entry.entryId === id))
    .filter((entry): entry is LayerEntry => Boolean(entry));
  if (selectedEntries.length !== selectedEntryIds.length) errors.push("direction references unknown entries");
  if (selectedEntries.some((entry) => entry.anchorId !== anchorId)) errors.push("direction contains entry from another anchor");
  const selectedEdges = selectedEdgeIds
    .map((id) => state.edges.find((edge) => edge.edgeId === id))
    .filter((edge): edge is CrossLayerEdge => Boolean(edge));
  if (selectedEdges.length !== selectedEdgeIds.length) errors.push("direction references unknown edges");
  for (const edge of selectedEdges) {
    if (edge.anchorId !== anchorId) errors.push(`edge ${edge.edgeId} belongs to another anchor`);
    if (!selectedEntryIds.includes(edge.fromEntryId) || !selectedEntryIds.includes(edge.toEntryId)) {
      errors.push(`edge ${edge.edgeId} endpoint is not selected`);
    }
    if (edge.compatibility === "conflict" || ["conflicts", "substitutes", "incompatible"].includes(edge.relation)) {
      errors.push(`edge ${edge.edgeId} is a conflict/substitute edge`);
    }
    if (edge.compatibility === "conditional" && !edge.condition) {
      errors.push(`conditional edge ${edge.edgeId} has no condition`);
    }
  }
  const allAnchorEdgesBetweenSelected = state.edges.filter(
    (edge) => edge.anchorId === anchorId
      && selectedEntryIds.includes(edge.fromEntryId)
      && selectedEntryIds.includes(edge.toEntryId),
  );
  if (allAnchorEdgesBetweenSelected.some((edge) => edge.compatibility === "conflict")) {
    errors.push("selected entries have a declared conflict/substitution relationship");
  }
  if (!graphConnected(selectedEntryIds, selectedEdges)) {
    errors.push("selected entry/edge subgraph is disconnected");
  }
  if (baselineIds.length === 0) errors.push("direction selects no baseline");
  if (baselineIds.some((id) => !state.baselines.some((baseline) => baseline.baselineId === id && baseline.anchorId === anchorId))) {
    errors.push("direction references a baseline outside the anchor");
  }

  const signature = stableHash([
    anchorId,
    selectedEntryIds,
    selectedEdgeIds,
    normalizeIdentityText(hypothesis),
  ], 24);
  const duplicate = state.directions.some((direction) =>
    direction.anchorId === anchorId
      && stableHash([
        direction.anchorId,
        [...direction.selectedEntryIds].sort(),
        [...direction.selectedEdgeIds].sort(),
        normalizeIdentityText(direction.hypothesis),
      ], 24) === signature
  );
  if (duplicate) errors.push("direction duplicates an accepted subgraph/hypothesis");
  if (errors.length > 0 || !anchor) return { direction: null, errors };

  const evidenceRefs = mergeStringArrays(
    selectedEntries.flatMap((entry) => entry.evidenceRefs),
    mergeStringArrays(
      selectedEdges.flatMap((edge) => edge.evidenceRefs),
      state.baselines.filter((baseline) => baselineIds.includes(baseline.baselineId)).flatMap((baseline) => baseline.evidenceRefs),
    ),
  );
  const directionId = stableId("D", [anchorId, signature], 16);
  const ablationPlan = stringArray(snakeOrCamel(object, "ablation_plan", "ablationPlan"));
  const implementationPlan = stringArray(snakeOrCamel(object, "implementation_plan", "implementationPlan"));
  return {
    direction: {
      directionId,
      anchorId,
      selectedEntryIds,
      selectedEdgeIds,
      baselineIds,
      hypothesis,
      ablationPlan: ablationPlan.length > 0
        ? ablationPlan
        : ["primary baseline", "each selected entry alone", "full selected subgraph"],
      metrics: anchor.signature.targetMetrics,
      implementationPlan,
      evidenceRefs,
      status: "accepted",
      rejectionReasons: [],
      proposalIndex,
    },
    errors: [],
  };
}

function enumValue<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  const candidate = stringValue(value);
  return allowed.includes(candidate as T) ? candidate as T : fallback;
}

function deterministicDecision(
  exploration: ExpertReview["explorationValue"],
  implementation: ExpertReview["implementationReuse"],
  method: ExpertReview["methodReference"],
  baseline: ExpertReview["baselineQuality"],
  crossLayer: ExpertReview["crossLayerValidity"],
  readiness: ExpertReview["experimentReadiness"],
  gaps: string[],
): ExpertReview["decision"] {
  if (baseline === "invalid" || crossLayer === "invalid") return "rejected";
  if (exploration === "low" && (implementation !== "low" || method !== "low" || baseline === "strong")) {
    return "baseline_reference";
  }
  if (gaps.length > 0 || readiness === "not_ready" || readiness === "unknown") {
    return "needs_evidence";
  }
  if (
    (exploration === "middle" || exploration === "high")
    && (baseline === "fair" || baseline === "strong")
    && readiness === "ready"
  ) {
    return "experiment_candidate";
  }
  return "needs_evidence";
}

export function buildExpertReview(
  directionId: string,
  payload: JsonValue,
  questionAnswers: ReviewQuestionAnswer[],
  referenceKeysUsed: string[],
): { review: ExpertReview | null; errors: string[] } {
  const object = asObject(payload);
  if (!object) return { review: null, errors: ["review payload is not an object"] };
  const covered = new Set(questionAnswers.map((qa) => qa.dimension));
  const missing = REVIEW_DIMENSIONS.filter((dimension) => !covered.has(dimension));
  if (missing.length > 0) {
    return { review: null, errors: [`missing review dimensions: ${missing.join(", ")}`] };
  }
  const explorationValue = enumValue(
    snakeOrCamel(object, "exploration_value", "explorationValue"),
    ["low", "middle", "high", "unknown"] as const,
    "unknown",
  );
  const implementationReuse = enumValue(
    snakeOrCamel(object, "implementation_reuse", "implementationReuse"),
    ["low", "middle", "high", "unknown"] as const,
    "unknown",
  );
  const methodReference = enumValue(
    snakeOrCamel(object, "method_reference", "methodReference"),
    ["low", "middle", "high", "unknown"] as const,
    "unknown",
  );
  const baselineQuality = enumValue(
    snakeOrCamel(object, "baseline_quality", "baselineQuality"),
    ["invalid", "weak", "fair", "strong", "unknown"] as const,
    "unknown",
  );
  const crossLayerValidity = enumValue(
    snakeOrCamel(object, "cross_layer_validity", "crossLayerValidity"),
    ["invalid", "weak", "conditional", "valid", "unknown"] as const,
    "unknown",
  );
  const experimentReadiness = enumValue(
    snakeOrCamel(object, "experiment_readiness", "experimentReadiness"),
    ["not_ready", "partial", "ready", "unknown"] as const,
    "unknown",
  );
  const gaps = mergeStringArrays(
    stringArray(object.gaps),
    questionAnswers.flatMap((qa) => qa.gaps),
  );
  const decision = deterministicDecision(
    explorationValue,
    implementationReuse,
    methodReference,
    baselineQuality,
    crossLayerValidity,
    experimentReadiness,
    gaps,
  );
  const review: ExpertReview = {
    directionId,
    status: "complete",
    pendingReason: null,
    explorationValue,
    implementationReuse,
    methodReference,
    baselineQuality,
    crossLayerValidity,
    experimentReadiness,
    decision,
    rationale: stringValue(object.rationale),
    minimumImplementationPlan: stringArray(
      snakeOrCamel(object, "minimum_implementation_plan", "minimumImplementationPlan"),
    ),
    baselineAblationMatrix: stringArray(
      snakeOrCamel(object, "baseline_ablation_matrix", "baselineAblationMatrix"),
    ),
    metricsTools: stringArray(snakeOrCamel(object, "metrics_tools", "metricsTools")),
    failureStopConditions: stringArray(
      snakeOrCamel(object, "failure_stop_conditions", "failureStopConditions"),
    ),
    selectedRefs: stringArray(snakeOrCamel(object, "selected_refs", "selectedRefs")),
    alternativeRefs: stringArray(snakeOrCamel(object, "alternative_refs", "alternativeRefs")),
    gaps,
    questionAnswers,
    referenceKeysUsed: [...new Set(referenceKeysUsed)].sort(),
  };
  return { review, errors: [] };
}

export function reviewDimension(value: unknown): ReviewDimension | null {
  const candidate = stringValue(value);
  return REVIEW_DIMENSIONS.includes(candidate as ReviewDimension) ? candidate as ReviewDimension : null;
}
