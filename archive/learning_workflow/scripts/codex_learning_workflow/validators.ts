import fs from "node:fs";

import { anchorId as expectedAnchorId } from "./stable_ids.ts";
import { REVIEW_DIMENSIONS } from "./role_profiles.ts";
import { revalidateClaim } from "./source_validator.ts";
import type {
  CrossLayerEdge,
  Direction,
  RunState,
  ValidationReport,
} from "./types.ts";

function graphConnected(direction: Direction, edges: CrossLayerEdge[]): boolean {
  if (direction.selectedEntryIds.length <= 1) return true;
  const adjacency = new Map<string, Set<string>>(
    direction.selectedEntryIds.map((entryId) => [entryId, new Set<string>()]),
  );
  for (const edge of edges) {
    adjacency.get(edge.fromEntryId)?.add(edge.toEntryId);
    adjacency.get(edge.toEntryId)?.add(edge.fromEntryId);
  }
  const seen = new Set<string>();
  const queue = [direction.selectedEntryIds[0]];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const neighbor of adjacency.get(current) ?? []) queue.push(neighbor);
  }
  return seen.size === direction.selectedEntryIds.length;
}

export function validateRunState(state: RunState): ValidationReport {
  const errors: string[] = [];
  const warnings: string[] = [];
  const claimIds = new Set(state.claims.map((claim) => claim.claimId));
  const anchorIds = new Set(state.anchors.map((anchor) => anchor.anchorId));
  const acceptedAnchorIds = new Set(state.stage1.acceptedAnchorIds);
  const baselineIds = new Set(state.baselines.map((baseline) => baseline.baselineId));
  const entryIds = new Set(state.entries.map((entry) => entry.entryId));
  const edgeIds = new Set(state.edges.map((edge) => edge.edgeId));
  const directionIds = new Set(state.directions.map((direction) => direction.directionId));

  if (state.config.model !== "gpt-5.6-sol") errors.push("config model is not gpt-5.6-sol");
  if (Object.values(state.config.roleEffort).some((effort) => effort === ("ultra" as never))) {
    errors.push("role effort contains forbidden ultra");
  }
  if (!state.provider?.modelVerified) errors.push("provider/model doctor was not recorded");
  if (state.stage1.status !== "complete") errors.push("Stage 1 is not complete");
  if (state.stage2.status !== "complete") errors.push("Stage 2 is not complete");
  if (state.status !== "complete") errors.push("run status is not complete");
  if (acceptedAnchorIds.size > state.config.maxAnchors) {
    errors.push(`accepted Anchor count exceeds cap ${state.config.maxAnchors}`);
  }
  for (const id of acceptedAnchorIds) {
    const anchor = state.anchors.find((candidate) => candidate.anchorId === id);
    if (!anchor || anchor.status !== "accepted") errors.push(`accepted_anchor_ids contains nonaccepted ${id}`);
  }

  const stopReason = state.stage1.stopReason;
  if (stopReason === "target_reached" && acceptedAnchorIds.size < state.config.maxAnchors) {
    errors.push("Stage 1 target_reached without reaching maxAnchors");
  } else if (
    stopReason === "no_new_anchor_streak"
    && state.stage1.consecutiveRoundsWithoutNewAnchor < state.config.noNewAnchorStop
  ) {
    errors.push("Stage 1 no_new_anchor_streak is below configured threshold");
  } else if (stopReason === "round_budget_exhausted" && state.stage1.round < state.config.maxStage1Rounds) {
    errors.push("Stage 1 round budget stop is inconsistent");
  } else if (stopReason === "task_budget_exhausted" && state.stage1.taskCount < state.config.maxStage1Tasks) {
    errors.push("Stage 1 task budget stop is inconsistent");
  } else if (stopReason === null) {
    errors.push("Stage 1 has no stop reason");
  }
  if (!state.stage1.anchorSpaceVersion) errors.push("Stage 1 has no immutable anchor_space_version");

  for (const claim of state.claims) {
    const reasons = revalidateClaim(
      claim,
      state.config.vaultRoot,
      state.config.evidenceRoots,
    );
    if (reasons.length > 0) {
      errors.push(`claim ${claim.claimId} failed source validation: ${reasons.join("; ")}`);
    }
  }
  if (claimIds.size !== state.claims.length) errors.push("duplicate claim IDs");

  for (const anchor of state.anchors) {
    if (anchor.anchorId !== expectedAnchorId(anchor.signature)) {
      errors.push(`anchor ${anchor.anchorId} does not match its normalized signature`);
    }
    if (anchor.evidenceRefs.some((id) => !claimIds.has(id))) {
      errors.push(`anchor ${anchor.anchorId} references unknown claim`);
    }
    if (anchor.baselineIds.some((id) => !baselineIds.has(id))) {
      errors.push(`anchor ${anchor.anchorId} references unknown baseline`);
    }
    if (anchor.entryIds.some((id) => !entryIds.has(id))) {
      errors.push(`anchor ${anchor.anchorId} references unknown entry`);
    }
    if (anchor.edgeIds.some((id) => !edgeIds.has(id))) {
      errors.push(`anchor ${anchor.anchorId} references unknown edge`);
    }
    if (anchor.status === "accepted") {
      const kinds = new Set(
        state.baselines.filter((baseline) => baseline.anchorId === anchor.anchorId).map((baseline) => baseline.kind),
      );
      for (const required of ["current_practice", "strong_comparison", "tool_evaluation", "reusable_implementation"]) {
        if (!kinds.has(required as never)) warnings.push(`anchor ${anchor.anchorId} baseline gap: ${required}`);
      }
    }
  }

  for (const baseline of state.baselines) {
    if (baseline.anchorId && !anchorIds.has(baseline.anchorId)) {
      errors.push(`baseline ${baseline.baselineId} references unknown anchor`);
    }
    if (baseline.evidenceRefs.length === 0 || baseline.evidenceRefs.some((id) => !claimIds.has(id))) {
      errors.push(`baseline ${baseline.baselineId} lacks valid evidence`);
    }
    if (baseline.explorationValue === "low" && baseline.status !== "active" && baseline.anchorId && acceptedAnchorIds.has(baseline.anchorId)) {
      errors.push(`low-exploration baseline ${baseline.baselineId} was not retained`);
    }
  }

  for (const entry of state.entries) {
    if (!anchorIds.has(entry.anchorId)) errors.push(`entry ${entry.entryId} references unknown anchor`);
    if (entry.evidenceRefs.length === 0 || entry.evidenceRefs.some((id) => !claimIds.has(id))) {
      errors.push(`entry ${entry.entryId} lacks valid evidence`);
    }
    if (entry.applicableBaselineIds.some((id) => !baselineIds.has(id))) {
      errors.push(`entry ${entry.entryId} references unknown baseline`);
    }
  }
  for (const edge of state.edges) {
    const from = state.entries.find((entry) => entry.entryId === edge.fromEntryId);
    const to = state.entries.find((entry) => entry.entryId === edge.toEntryId);
    if (!from || !to) {
      errors.push(`edge ${edge.edgeId} has unknown endpoint`);
      continue;
    }
    if (from.anchorId !== edge.anchorId || to.anchorId !== edge.anchorId) {
      errors.push(`edge ${edge.edgeId} crosses Anchor boundary`);
    }
    if (edge.compatibility === "conditional" && !edge.condition) {
      errors.push(`conditional edge ${edge.edgeId} has no condition`);
    }
    if (edge.evidenceRefs.length === 0 || edge.evidenceRefs.some((id) => !claimIds.has(id))) {
      errors.push(`edge ${edge.edgeId} lacks valid evidence`);
    }
  }

  for (const direction of state.directions) {
    const anchor = state.anchors.find((candidate) => candidate.anchorId === direction.anchorId);
    if (!anchor || anchor.status !== "accepted") errors.push(`direction ${direction.directionId} belongs to nonaccepted anchor`);
    const entries = direction.selectedEntryIds
      .map((id) => state.entries.find((entry) => entry.entryId === id))
      .filter(Boolean);
    if (entries.length !== direction.selectedEntryIds.length) {
      errors.push(`direction ${direction.directionId} references unknown entries`);
    }
    if (entries.some((entry) => entry?.anchorId !== direction.anchorId)) {
      errors.push(`direction ${direction.directionId} crosses Anchor boundary`);
    }
    const edges = direction.selectedEdgeIds
      .map((id) => state.edges.find((edge) => edge.edgeId === id))
      .filter((edge): edge is CrossLayerEdge => Boolean(edge));
    if (edges.length !== direction.selectedEdgeIds.length) {
      errors.push(`direction ${direction.directionId} references unknown edges`);
    }
    if (edges.some((edge) =>
      edge.compatibility === "conflict"
      || ["conflicts", "substitutes", "incompatible"].includes(edge.relation)
    )) {
      errors.push(`direction ${direction.directionId} selects conflict/substitute edge`);
    }
    const declaredConflicts = state.edges.some((edge) =>
      edge.anchorId === direction.anchorId
      && direction.selectedEntryIds.includes(edge.fromEntryId)
      && direction.selectedEntryIds.includes(edge.toEntryId)
      && edge.compatibility === "conflict"
    );
    if (declaredConflicts) errors.push(`direction ${direction.directionId} selects conflicting entries`);
    if (!graphConnected(direction, edges)) errors.push(`direction ${direction.directionId} subgraph is disconnected`);
    if (direction.baselineIds.length === 0 || direction.baselineIds.some((id) => !baselineIds.has(id))) {
      errors.push(`direction ${direction.directionId} lacks valid baseline`);
    }
    if (direction.evidenceRefs.some((id) => !claimIds.has(id))) {
      errors.push(`direction ${direction.directionId} references unknown evidence`);
    }
  }
  if (directionIds.size !== state.directions.length) errors.push("duplicate direction IDs");

  for (const anchorId of acceptedAnchorIds) {
    const planning = state.stage2.anchorPlanning[anchorId];
    if (!planning || !["complete", "pending_budget"].includes(planning.status)) {
      errors.push(`anchor ${anchorId} planning is not terminal`);
    }
  }
  for (const direction of state.directions.filter((item) => item.status === "accepted")) {
    const review = state.reviews.find((candidate) => candidate.directionId === direction.directionId);
    if (!review) {
      errors.push(`direction ${direction.directionId} has no review/pending record`);
      continue;
    }
    if (review.status === "pending" && !review.pendingReason) {
      errors.push(`pending review ${direction.directionId} has no reason`);
    }
    if (review.status === "complete") {
      const covered = new Set(review.questionAnswers.map((qa) => qa.dimension));
      const missing = REVIEW_DIMENSIONS.filter((dimension) => !covered.has(dimension));
      if (missing.length > 0) errors.push(`review ${direction.directionId} misses dimensions: ${missing.join(", ")}`);
    }
    for (const qa of review.questionAnswers) {
      if (qa.evidenceRefs.some((id) => !claimIds.has(id))) {
        errors.push(`review ${direction.directionId} QA ${qa.questionId} references unknown evidence`);
      }
      if (qa.evidenceRefs.length === 0 && qa.gaps.length === 0 && qa.conclusion !== "not_applicable") {
        errors.push(`review ${direction.directionId} QA ${qa.questionId} has neither evidence nor explicit gap`);
      }
    }
  }

  if (!fs.existsSync(state.config.skillRoot)) errors.push("configured skill root does not exist");
  return { ok: errors.length === 0, errors, warnings };
}

