import type {
  Anchor,
  Direction,
  EvidencePacket,
  ReviewDelta,
  SearchNeed,
  TopicFrame,
} from "./contracts/index.ts";

export interface FinalRenderModel {
  runId: string;
  canonicalRevision: number;
  topic: TopicFrame;
  anchors: Anchor[];
  directions: Direction[];
  evidencePackets: EvidencePacket[];
  directionReviews: ReviewDelta[];
  unresolvedNeeds: SearchNeed[];
  contradictions: Array<{
    contradictionId: string;
    summary: string;
    disposition: string | null;
  }>;
  experimentHandoffs: Array<{
    handoffId: string;
    directionId: string;
    reason: string;
    executionAuthorized: false;
  }>;
}

export const FINAL_SECTION_HEADINGS = Object.freeze([
  "Topic scope",
  "Performance opportunity overview",
  "Anchor summaries",
  "Direction statuses",
  "Evidence provenance",
  "Contradictions and limits",
  "Experiment handoffs",
  "Unresolved questions",
] as const);

export function renderFinalMarkdown(model: FinalRenderModel): string {
  const anchors = [...model.anchors].sort((a, b) =>
    a.anchorId.localeCompare(b.anchorId),
  );
  const directions = [...model.directions].sort((a, b) =>
    a.directionId.localeCompare(b.directionId),
  );
  const evidence = model.evidencePackets
    .flatMap((packet) => packet.findings)
    .sort((a, b) => a.evidenceId.localeCompare(b.evidenceId));
  const lines = [
    "# Simple Semantic Loop result",
    "",
    `Run: \`${escapeInline(model.runId)}\`  `,
    `Canonical revision: \`${model.canonicalRevision}\``,
    "",
    "## Topic scope",
    "",
    `**Topic ID:** \`${escapeInline(model.topic.topicId)}\``,
    "",
    `**Topic:** ${escapeMarkdown(model.topic.userTopic)}`,
    "",
    `**Objective:** ${escapeMarkdown(model.topic.objective)}`,
    "",
    `- Workloads: ${join(model.topic.workloads)}`,
    `- Phases: ${join(model.topic.phases)}`,
    `- Regimes: ${join(model.topic.regimes)}`,
    `- Stack: ${join(model.topic.stackScope)}`,
    `- Layers: ${join(model.topic.layerScope)}`,
    `- Target metrics: ${join(model.topic.targetMetrics)}`,
    `- Invariants: ${join(model.topic.invariants)}`,
    `- Exclusions: ${join(model.topic.exclusions)}`,
    "",
    "## Performance opportunity overview",
    "",
    `- Canonical Anchors: ${anchors.length}`,
    `- Canonical Directions: ${directions.length}`,
    `- Testable Directions: ${directions.filter((direction) => direction.status === "testable").length}`,
    `- Experiment-required Directions: ${directions.filter((direction) => direction.status === "experiment_required").length}`,
    `- Rejected Directions: ${directions.filter((direction) => direction.status === "rejected").length}`,
    ...(directions.length
      ? directions.map(
          (direction) =>
            `- \`${escapeInline(direction.directionId)}\` ${escapeMarkdown(direction.title)} — ${escapeMarkdown(direction.status)}`,
        )
      : [
          "- No optimization Direction survived the bounded search and review closure.",
        ]),
    "",
    "## Anchor summaries",
    "",
    ...(anchors.length
      ? anchors.flatMap((anchor) =>
          renderAnchor(
            anchor,
            directions.filter(
              (direction) => direction.anchorId === anchor.anchorId,
            ),
          ),
        )
      : ["No canonical Anchors.", ""]),
    "## Direction statuses",
    "",
    ...(directions.length
      ? directions.flatMap((direction) =>
          renderDirection(
            direction,
            model.directionReviews.filter(
              (review) => review.directionId === direction.directionId,
            ),
          ),
        )
      : ["No canonical Directions.", ""]),
    "## Evidence provenance",
    "",
    ...(evidence.length
      ? evidence.flatMap((finding) => [
          `- \`${escapeInline(finding.evidenceId)}\` ${escapeMarkdown(finding.claim)} — \`${escapeInline(finding.sourcePath)}#${escapeInline(finding.heading)}\` (\`${escapeInline(finding.sourceUnitId)}\`)`,
        ])
      : ["No committed Evidence findings."]),
    "",
    "## Contradictions and limits",
    "",
    ...(model.contradictions.length
      ? [...model.contradictions]
          .sort((a, b) =>
            a.contradictionId.localeCompare(b.contradictionId),
          )
          .map(
            (item) =>
              `- \`${escapeInline(item.contradictionId)}\` ${escapeMarkdown(item.summary)}; disposition: ${escapeMarkdown(item.disposition ?? "unresolved")}`,
          )
      : ["No committed critical contradictions."]),
    "",
    "## Experiment handoffs",
    "",
    ...(model.experimentHandoffs.length
      ? [...model.experimentHandoffs]
          .sort((a, b) => a.handoffId.localeCompare(b.handoffId))
          .map(
            (handoff) =>
              `- \`${escapeInline(handoff.handoffId)}\` for \`${escapeInline(handoff.directionId)}\`: ${escapeMarkdown(handoff.reason)} (executionAuthorized=\`false\`)`,
          )
      : ["No experiment handoffs."]),
    "",
    "## Unresolved questions",
    "",
    ...(model.unresolvedNeeds.length
      ? [...model.unresolvedNeeds]
          .sort((a, b) => a.needId.localeCompare(b.needId))
          .map(
            (need) =>
              `- \`${escapeInline(need.needId)}\` ${escapeMarkdown(need.question)} [${escapeMarkdown(need.answerability)} / ${escapeMarkdown(need.status)}]`,
          )
      : ["No unresolved canonical SearchNeeds."]),
    "",
  ];
  return `${lines.join("\n").replace(/\n{3,}/g, "\n\n")}\n`;
}

export function validateRenderedCoverage(
  markdown: string,
  model: FinalRenderModel,
): string[] {
  const errors: string[] = [];
  for (const heading of FINAL_SECTION_HEADINGS) {
    if (!markdown.includes(`## ${heading}`)) {
      errors.push(`missing final output section: ${heading}`);
    }
  }
  const requiredIds = [
    model.topic.topicId,
    ...model.anchors.map((anchor) => anchor.anchorId),
    ...model.directions.map((direction) => direction.directionId),
    ...model.directions.flatMap((direction) =>
      direction.changes.map((change) => change.atomId),
    ),
    ...model.evidencePackets.flatMap((packet) =>
      packet.findings.map((finding) => finding.evidenceId),
    ),
    ...model.directionReviews.map((review) => review.reviewId),
    ...model.experimentHandoffs.map((handoff) => handoff.handoffId),
  ];
  for (const id of requiredIds) {
    if (!markdown.includes(id)) {
      errors.push(`canonical object is not traceable in final output: ${id}`);
    }
  }
  return errors;
}

function renderAnchor(
  anchor: Anchor,
  directions: Direction[],
): string[] {
  return [
    `### ${escapeMarkdown(anchor.anchorId)} — ${escapeMarkdown(anchor.status)}`,
    "",
    `**Scenario:** ${escapeMarkdown(anchor.scenario.workload)} / ${escapeMarkdown(anchor.scenario.phase)} / ${escapeMarkdown(anchor.scenario.regime)}`,
    "",
    `- Stack boundary: ${join(anchor.scenario.stack)}`,
    `- Baseline: ${escapeMarkdown(anchor.baseline.name)}`,
    `- Baseline execution path: ${joinArrow(anchor.baseline.executionPath)}`,
    `- Baseline configuration: ${join(anchor.baseline.configuration)}`,
    `- Fair-comparison scope: ${join(anchor.baseline.comparisonScope)}`,
    `- Performance symptom: ${escapeMarkdown(anchor.performanceTension.symptom)}`,
    `- Suspected mechanism: ${escapeMarkdown(anchor.performanceTension.suspectedMechanism)}`,
    `- Bottleneck resources: ${join(anchor.performanceTension.bottleneckResources)}`,
    `- Target metrics: ${join(anchor.performanceTension.targetMetrics)}`,
    `- Constraints: ${join(anchor.constraints)}`,
    `- Evidence: ${joinCode(anchor.evidenceRefs)}`,
    `- Open Needs: ${joinCode(anchor.openNeedIds)}`,
    `- Directions: ${joinCode(directions.map((direction) => direction.directionId))}`,
    `- Closure/status reason: ${escapeMarkdown(anchor.saturationReason ?? (anchor.statusReason || "not supplied"))}`,
    "",
  ];
}

function renderDirection(
  direction: Direction,
  reviews: ReviewDelta[],
): string[] {
  return [
    `### ${escapeMarkdown(direction.directionId)} — ${escapeMarkdown(direction.status)}`,
    "",
    `**${escapeMarkdown(direction.title)}.** ${escapeMarkdown(direction.hypothesis)}`,
    "",
    `- Anchor: \`${escapeInline(direction.anchorId)}\``,
    `- Status reason: ${escapeMarkdown(direction.statusReason || "not supplied")}`,
    "",
    "#### Modification atoms",
    "",
    ...direction.changes.map(
      (change) =>
        `- \`${escapeInline(change.atomId)}\` **${escapeMarkdown(change.layer)} / ${escapeMarkdown(change.role)}** — ${escapeMarkdown(change.object)}: ${escapeMarkdown(change.fromState)} → ${escapeMarkdown(change.toState)}; conditions: ${join(change.conditions)}; evidence: ${joinCode(change.evidenceRefs)}`,
    ),
    "",
    "#### Causal chain and expected effects",
    "",
    ...direction.causalLinks.map(
      (link) =>
        `- ${escapeMarkdown(link.from)} —${escapeMarkdown(link.relation)}→ ${escapeMarkdown(link.to)} when ${escapeMarkdown(link.condition)} [${escapeMarkdown(link.directness)}; evidence: ${joinCode(link.evidenceRefs)}]`,
    ),
    ...direction.expectedEffects.map(
      (effect) =>
        `- Expected ${escapeMarkdown(effect.metric)} to **${escapeMarkdown(effect.expectedDirection)}**: ${escapeMarkdown(effect.rationale)}`,
    ),
    "",
    "#### Fair comparison, implementation, and falsification",
    "",
    `- Comparison baseline: ${escapeMarkdown(direction.comparison.baseline)}`,
    `- Controlled variables: ${join(direction.comparison.controlledVariables)}`,
    `- Ablations: ${join(direction.comparison.ablations)}`,
    `- Target components: ${join(direction.implementation.targetComponents)}`,
    `- Known entry points: ${join(direction.implementation.knownEntryPoints)}`,
    `- Unresolved interfaces: ${join(direction.implementation.unresolvedInterfaces)}`,
    `- Falsifiers: ${join(direction.falsifiers)}`,
    `- Degradation conditions: ${join(direction.degradationConditions)}`,
    `- Supporting evidence: ${joinCode(direction.supportingEvidenceRefs)}`,
    `- Contradicting evidence: ${joinCode(direction.contradictingEvidenceRefs)}`,
    `- Inferred claims: ${join(direction.inferredClaims)}`,
    `- Unresolved Needs: ${joinCode(direction.unresolvedNeedIds)}`,
    `- Experiment handoff: ${direction.experimentHandoffId ? `\`${escapeInline(direction.experimentHandoffId)}\`` : "none"}`,
    "",
    "#### Independent reviews",
    "",
    ...(reviews.length
      ? [...reviews]
          .sort((left, right) => left.reviewId.localeCompare(right.reviewId))
          .map(
            (review) =>
              `- \`${escapeInline(review.reviewId)}\` — **${escapeMarkdown(review.decision)}**: ${escapeMarkdown(review.rationale)}; weakest link: ${escapeMarkdown(review.weakestCausalLink ?? "none")}; strongest counterexample: ${escapeMarkdown(review.strongestCounterexample ?? "none")}`,
          )
      : ["- No committed independent review."]),
    "",
  ];
}

function join(values: readonly string[]): string {
  return values.length
    ? values.map(escapeMarkdown).join(", ")
    : "not supplied";
}

function joinArrow(values: readonly string[]): string {
  return values.length
    ? values.map(escapeMarkdown).join(" → ")
    : "not supplied";
}

function joinCode(values: readonly string[]): string {
  return values.length
    ? values.map((value) => `\`${escapeInline(value)}\``).join(", ")
    : "none";
}

function escapeInline(value: string): string {
  return value.replaceAll("`", "\\`");
}

function escapeMarkdown(value: string): string {
  return value
    .replaceAll("\\", "\\\\")
    .replaceAll("*", "\\*")
    .replaceAll("_", "\\_")
    .replaceAll("[", "\\[")
    .replaceAll("]", "\\]")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
