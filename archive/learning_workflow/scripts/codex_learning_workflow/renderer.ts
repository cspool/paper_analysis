import fs from "node:fs";
import path from "node:path";

import { atomicWriteJson } from "./canonical_store.ts";
import type {
  Anchor,
  Direction,
  ExpertReview,
  Layer,
  RunState,
} from "./types.ts";

const LAYERS: Layer[] = ["L1", "L2", "L3", "L4", "L5", "L6"];

function cell(value: unknown): string {
  return String(value ?? "")
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|")
    .trim();
}

function idList(values: string[]): string {
  return values.length > 0 ? values.map((value) => `\`${value}\``).join(", ") : "—";
}

function sourceLink(state: RunState, claimId: string): string {
  const claim = state.claims.find((item) => item.claimId === claimId);
  if (!claim) return `\`${claimId}\` (missing)`;
  const absolute = path.resolve(state.config.vaultRoot, claim.sourcePath);
  return `\`${claimId}\` → [${cell(claim.sourcePath)}](${absolute}#L${claim.lineStart}) L${claim.lineStart}–L${claim.lineEnd}`;
}

function renderReview(state: RunState, direction: Direction, review: ExpertReview): string {
  const lines: string[] = [];
  lines.push(`# Expert Review: ${direction.directionId}`, "");
  lines.push(`- Status: \`${review.status}\`${review.pendingReason ? ` — ${review.pendingReason}` : ""}`);
  lines.push(`- Decision: \`${review.decision}\``);
  lines.push(`- Exploration value: \`${review.explorationValue}\``);
  lines.push(`- Implementation reuse: \`${review.implementationReuse}\``);
  lines.push(`- Method reference: \`${review.methodReference}\``);
  lines.push(`- Baseline quality: \`${review.baselineQuality}\``);
  lines.push(`- Cross-layer validity: \`${review.crossLayerValidity}\``);
  lines.push(`- Experiment readiness: \`${review.experimentReadiness}\``);
  lines.push("", review.rationale || "No terminal rationale was available.", "");
  lines.push("## Review ledger", "");
  lines.push("| Dimension | Question | Conclusion | Evidence | Gaps |");
  lines.push("|---|---|---|---|---|");
  for (const qa of review.questionAnswers) {
    lines.push(`| ${qa.dimension} | ${cell(qa.question)} | ${qa.conclusion} | ${idList(qa.evidenceRefs)} | ${cell(qa.gaps.join("; ")) || "—"} |`);
  }
  lines.push("", "## Minimum implementation plan", "");
  lines.push(...(review.minimumImplementationPlan.length ? review.minimumImplementationPlan.map((item) => `- ${item}`) : ["- Pending"]));
  lines.push("", "## Baseline and ablation matrix", "");
  lines.push(...(review.baselineAblationMatrix.length ? review.baselineAblationMatrix.map((item) => `- ${item}`) : ["- Pending"]));
  lines.push("", "## Metrics and tools", "");
  lines.push(...(review.metricsTools.length ? review.metricsTools.map((item) => `- ${item}`) : ["- Pending"]));
  lines.push("", "## Failure / stop conditions", "");
  lines.push(...(review.failureStopConditions.length ? review.failureStopConditions.map((item) => `- ${item}`) : ["- Pending"]));
  lines.push("", "## Evidence trace", "");
  const refs = [...new Set(review.questionAnswers.flatMap((qa) => qa.evidenceRefs))].sort();
  lines.push(...(refs.length ? refs.map((claimId) => `- ${sourceLink(state, claimId)}`) : ["- No direct evidence; see explicit gaps above."]));
  lines.push("");
  return lines.join("\n");
}

function renderAnchor(state: RunState, anchor: Anchor): string[] {
  const lines: string[] = [];
  lines.push(`## ${anchor.anchorId}: ${anchor.title}`, "");
  lines.push(anchor.scenario || "Scenario is represented by the exact signature below.", "");
  lines.push("```json", JSON.stringify(anchor.signature, null, 2), "```", "");
  lines.push("### Baseline set", "");
  lines.push("| ID | Kind | Name | Execution path | Implementation | Exploration | Evidence |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const baseline of state.baselines
    .filter((item) => item.anchorId === anchor.anchorId)
    .sort((left, right) => left.baselineId.localeCompare(right.baselineId))) {
    lines.push(`| ${baseline.baselineId} | ${baseline.kind} | ${cell(baseline.name)} | ${cell(baseline.executionPath)} | ${cell(baseline.implementation) || "—"} | ${baseline.explorationValue} | ${idList(baseline.evidenceRefs)} |`);
  }
  lines.push("", "### L1–L6 Intervention Map", "");
  lines.push("| Layer | Entry | Role | Atomic claim | Modifiable object | Baselines | Preconditions | Expected effect | Evidence |");
  lines.push("|---|---|---|---|---|---|---|---|---|");
  for (const layer of LAYERS) {
    const entries = state.entries
      .filter((entry) => entry.anchorId === anchor.anchorId && entry.layer === layer)
      .sort((left, right) => left.entryId.localeCompare(right.entryId));
    if (entries.length === 0) {
      lines.push(`| ${layer} | — | gap | — | — | — | — | — | — |`);
      continue;
    }
    for (const entry of entries) {
      lines.push(`| ${layer} | ${entry.entryId} | ${entry.role} | ${cell(entry.claim)} | ${cell(entry.modifiableObject) || "—"} | ${idList(entry.applicableBaselineIds)} | ${cell(entry.preconditions.join("; ")) || "—"} | ${cell(entry.expectedEffect) || "—"} | ${idList(entry.evidenceRefs)} |`);
    }
  }
  lines.push("", "### Entry-level cross-layer edges", "");
  lines.push("| Edge | From | To | Relation | Compatibility | Interface | Condition | Evidence |");
  lines.push("|---|---|---|---|---|---|---|---|");
  const edges = state.edges
    .filter((edge) => edge.anchorId === anchor.anchorId)
    .sort((left, right) => left.edgeId.localeCompare(right.edgeId));
  if (edges.length === 0) {
    lines.push("| — | — | — | — | — | No verified edge | — | — |");
  } else {
    for (const edge of edges) {
      lines.push(`| ${edge.edgeId} | ${edge.fromEntryId} | ${edge.toEntryId} | ${edge.relation} | ${edge.compatibility} | ${cell(edge.interface)} | ${cell(edge.condition) || "—"} | ${idList(edge.evidenceRefs)} |`);
    }
  }
  const directions = state.directions
    .filter((direction) => direction.anchorId === anchor.anchorId && direction.status === "accepted")
    .sort((left, right) => left.directionId.localeCompare(right.directionId));
  lines.push("", "### Direction Bundles", "");
  if (directions.length === 0) {
    const planning = state.stage2.anchorPlanning[anchor.anchorId];
    lines.push(`No accepted Direction. Planning terminal state: \`${planning?.status ?? "missing"}\` (${planning?.reason ?? "no reason"}).`, "");
  }
  for (const direction of directions) {
    const review = state.reviews.find((item) => item.directionId === direction.directionId);
    lines.push(`#### ${direction.directionId}`, "");
    lines.push(`- Hypothesis: ${direction.hypothesis}`);
    lines.push(`- Selected entries: ${idList(direction.selectedEntryIds)}`);
    lines.push(`- Selected edges: ${idList(direction.selectedEdgeIds)}`);
    lines.push(`- Baselines: ${idList(direction.baselineIds)}`);
    lines.push(`- Metrics: ${cell(direction.metrics.join(", "))}`);
    lines.push(`- Ablations: ${cell(direction.ablationPlan.join("; "))}`);
    lines.push(`- Review: \`${review?.decision ?? "missing"}\` / \`${review?.status ?? "missing"}\`${review?.pendingReason ? ` (${review.pendingReason})` : ""}`);
    lines.push(`- Review file: [${direction.directionId}.md](reviews/${direction.directionId}.md)`, "");
  }
  if (anchor.gaps.length > 0) {
    lines.push("### Anchor gaps", "", ...anchor.gaps.map((gap) => `- ${gap}`), "");
  }
  return lines;
}

export function renderRun(state: RunState, workDir: string): string {
  if (!state.validation.ok) {
    throw new Error("formal render is forbidden until validation succeeds");
  }
  const lines: string[] = [];
  lines.push("# Codex Learning Workflow Result", "");
  lines.push(`- Topic: ${state.config.topic}`);
  lines.push(`- Run: \`${state.runId}\``);
  lines.push(`- Model: \`${state.config.model}\``);
  lines.push(`- Stage 1 stop: \`${state.stage1.stopReason}\``);
  lines.push(`- Anchor space version: \`${state.stage1.anchorSpaceVersion}\``);
  lines.push(`- Accepted Anchors: ${state.stage1.acceptedAnchorIds.length}`);
  lines.push(`- Accepted Directions: ${state.directions.filter((direction) => direction.status === "accepted").length}`);
  lines.push("");
  lines.push("> Value order: exploratory scenario / acceleration opportunity > reusable implementation/tool/software > paper method. Valid baselines remain an independent mandatory lane.", "");

  lines.push("## Global Layer Catalog", "");
  lines.push("| Layer | Entry | Anchor | Role | Claim | Modifiable object | Evidence |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const layer of LAYERS) {
    const entries = state.entries
      .filter((entry) => entry.layer === layer && state.stage1.acceptedAnchorIds.includes(entry.anchorId))
      .sort((left, right) => left.entryId.localeCompare(right.entryId));
    for (const entry of entries) {
      lines.push(`| ${layer} | ${entry.entryId} | ${entry.anchorId} | ${entry.role} | ${cell(entry.claim)} | ${cell(entry.modifiableObject) || "—"} | ${idList(entry.evidenceRefs)} |`);
    }
  }
  lines.push("", "## Anchor Layer Maps", "");
  for (const anchor of state.anchors
    .filter((item) => item.status === "accepted")
    .sort((left, right) => left.anchorId.localeCompare(right.anchorId))) {
    lines.push(...renderAnchor(state, anchor));
  }

  lines.push("## Baseline / Reference Registry", "");
  lines.push("| Baseline | Anchor | Status | Kind | Name | Execution path | Implementation | Evidence |");
  lines.push("|---|---|---|---|---|---|---|---|");
  for (const baseline of [...state.baselines].sort((left, right) => left.baselineId.localeCompare(right.baselineId))) {
    lines.push(`| ${baseline.baselineId} | ${baseline.anchorId ?? "global"} | ${baseline.status} | ${baseline.kind} | ${cell(baseline.name)} | ${cell(baseline.executionPath)} | ${cell(baseline.implementation) || "—"} | ${idList(baseline.evidenceRefs)} |`);
  }
  lines.push("", "## Global Entity Registry", "");
  lines.push("| Entity | Kind | Name | Description | Evidence |");
  lines.push("|---|---|---|---|---|");
  for (const entity of [...state.entities].sort((left, right) => left.entityId.localeCompare(right.entityId))) {
    lines.push(`| ${entity.entityId} | ${entity.kind} | ${cell(entity.name)} | ${cell(entity.description)} | ${idList(entity.evidenceRefs)} |`);
  }

  lines.push("", "## Evidence Index", "");
  for (const claim of [...state.claims].sort((left, right) => left.claimId.localeCompare(right.claimId))) {
    lines.push(`- ${sourceLink(state, claim.claimId)} — ${cell(claim.statement)} (\`${claim.evidenceKind}\`)`);
  }
  lines.push("", "## Unresolved gaps", "");
  const gaps = [
    ...state.anchors.flatMap((anchor) => anchor.gaps.map((gap) => `${anchor.anchorId}: ${gap}`)),
    ...state.reviews.flatMap((review) => review.gaps.map((gap) => `${review.directionId}: ${gap}`)),
    ...state.reviews
      .filter((review) => review.pendingReason)
      .map((review) => `${review.directionId}: ${review.pendingReason}`),
  ];
  lines.push(...(gaps.length ? [...new Set(gaps)].sort().map((gap) => `- ${gap}`) : ["- None recorded."]));
  lines.push("");

  const output = lines.join("\n");
  for (const direction of state.directions.filter((item) => item.status === "accepted")) {
    const review = state.reviews.find((item) => item.directionId === direction.directionId);
    if (review) {
      fs.writeFileSync(
        path.join(workDir, "reviews", `${direction.directionId}.md`),
        renderReview(state, direction, review),
        "utf8",
      );
    }
  }
  fs.writeFileSync(path.join(workDir, "final.md"), output, "utf8");
  atomicWriteJson(path.join(workDir, "validation.json"), state.validation);
  return output;
}

