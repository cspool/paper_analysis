import { FileLoopStore } from "./store.ts";
import { buildNegativeExperimentIndex } from "./experiment_history.ts";
import type {
  ExperimentGoalResult,
  ObjectRevision,
  ReviewResult,
  RunOutcome,
  WorkflowGoal,
  WorkResult,
} from "./types.ts";

interface RawAppendix {
  label: string;
  ref: string;
  value: unknown;
}

export function renderFinalReport(store: FileLoopStore): void {
  const goal = store.readJson<WorkflowGoal>("workflow_goal.json");
  const index = store.readObjects();
  const lines: string[] = [
    "# Learning Workflow Report",
    "",
    "## Topic",
    "",
    goal.topic,
    "",
    "## Objective",
    "",
    goal.objective,
    "",
    "## Acceptance criteria",
    "",
    ...goal.acceptanceCriteria.map((criterion) => `- ${criterion}`),
    "",
    "## Results",
    "",
  ];
  const sourceResultRefs = new Set<string>();
  const experimentResultRefs: string[] = [];
  const rawAppendices: RawAppendix[] = [];
  let anchorNumber = 0;

  for (const anchorId of index.activeAnchorIds) {
    const anchorEntry = index.anchors[anchorId];
    if (!anchorEntry || anchorEntry.rejected) continue;
    const anchorRevision =
      anchorEntry.revisions[String(anchorEntry.latestRevision)];
    if (!anchorRevision) continue;

    anchorNumber += 1;
    const anchorWork = store.readJson<WorkResult>(anchorRevision.workRef);
    const anchorReview =
      store.readJson<ReviewResult>(anchorRevision.reviewRef);
    sourceResultRefs.add(anchorRevision.workRef);
    sourceResultRefs.add(anchorRevision.reviewRef);
    renderAnchor(
      lines,
      anchorNumber,
      anchorId,
      anchorRevision,
      anchorWork,
      anchorReview,
      rawAppendices,
    );

    let directionNumber = 0;
    for (const directionId of anchorEntry.directionIds) {
      const directionEntry = index.directions[directionId];
      if (!directionEntry || directionEntry.rejected) continue;
      const revision =
        directionEntry.revisions[String(directionEntry.latestRevision)];
      if (!revision || !isBoundToAnchor(store, revision, anchorRevision.workRef)) {
        continue;
      }

      directionNumber += 1;
      const work = store.readJson<WorkResult>(revision.workRef);
      const review = store.readJson<ReviewResult>(revision.reviewRef);
      sourceResultRefs.add(revision.workRef);
      sourceResultRefs.add(revision.reviewRef);
      renderDirection(
        lines,
        anchorNumber,
        directionNumber,
        directionId,
        revision,
        work,
        review,
        rawAppendices,
      );
    }
    if (directionNumber === 0) {
      lines.push(
        "",
        "_No current non-rejected Direction was available for rendering._",
      );
    }
    lines.push("");
  }

  if (anchorNumber === 0) {
    lines.push("_No current non-rejected Anchor was available for rendering._", "");
  }
  for (const experimentRef of store.experimentRefs()) {
    const record = store.readExperiment(experimentRef);
    if (!store.exists(record.resultRef)) continue;
    const result = store.readJson<ExperimentGoalResult>(record.resultRef);
    experimentResultRefs.push(record.resultRef);
    if (experimentResultRefs.length === 1) {
      lines.push("", "## EXP Goal evidence", "");
    }
    lines.push(
      `### ${result.experimentId}`,
      "",
      `- Status: ${result.goalStatus}`,
      `- Objective: ${result.experimentObjective}`,
      `- Anchor: \`${result.anchorWork}\``,
      `- Direction: ${result.directionWork ? `\`${result.directionWork}\`` : "None"}`,
      `- Result: \`${record.resultRef}\``,
      `- Workspace: \`${result.workspaceRef}\``,
    );
    if (result.conclusionRef && store.exists(result.conclusionRef)) {
      lines.push(
        `- Conclusion: \`${result.conclusionRef}\``,
        "",
        store.readText(result.conclusionRef).trim(),
        "",
      );
    } else if (result.error) {
      lines.push(`- Runtime limitation: ${result.error}`, "");
    }
  }
  const negativeIndex = buildNegativeExperimentIndex(store, store.readState());
  if (negativeIndex.entries.length > 0) {
    lines.push("", "## Reviewed negative EXP evidence", "");
    for (const entry of negativeIndex.entries) {
      const review = store.readJson<ReviewResult>(entry.reviewRef);
      lines.push(
        `- EXP \`${entry.experimentResultRef}\` → ${entry.reviewVerdict}; ` +
          `Anchor \`${entry.anchorWork}\`; Direction ${
            entry.directionWork ? `\`${entry.directionWork}\`` : "None"
          }; review \`${entry.reviewRef}\`${
            review.summary ? `: ${review.summary}` : ""
          }`,
      );
    }
  }
  appendRawAppendices(lines, rawAppendices);

  store.writeText("final/report.md", `${lines.join("\n").trim()}\n`);
  store.writeJson("final/manifest.json", {
    generatedAt: new Date().toISOString(),
    goalRef: "workflow_goal.json",
    sourceResultRefs: [...sourceResultRefs],
    experimentResultRefs,
    negativeExperimentEntries: negativeIndex.entries,
  });
  store.writeJson("final/outcome.json", {
    workflowOutcome: "FINISHED",
    reportRef: "final/report.md",
    reason: null,
  } satisfies RunOutcome);
  store.appendEvent("FINAL_ARTIFACT_WRITTEN", [
    "final/report.md",
    "final/manifest.json",
    "final/outcome.json",
  ]);
}

function renderAnchor(
  lines: string[],
  number: number,
  objectId: string,
  revision: ObjectRevision,
  work: WorkResult,
  review: ReviewResult,
  appendices: RawAppendix[],
): void {
  const content = asRecord(work.content);
  const title = stringValue(content?.name) ?? objectId;
  lines.push(
    `### Anchor ${number}: ${title}`,
    "",
    `- Object: \`${objectId}\` revision ${revision.revision}`,
    `- Work outcome: ${work.workOutcome}`,
    `- Review verdict: ${review.reviewVerdict}`,
  );
  appendScalar(lines, "Scenario", content?.scenario);
  appendScalar(lines, "Baseline", content?.baseline);
  appendScalar(lines, "Performance tension", content?.performanceTension);
  appendScope6L(lines, content?.scope6L);
  appendScalar(lines, "Independent review", review.summary);
  appendStringList(lines, "Constraints", content?.constraints);
  appendEvidence(lines, work.evidence);
  appendReviewCaveats(lines, review.findings);

  if (!isStandardAnchorResult(work)) {
    appendices.push({
      label: `Anchor ${number} Work Result`,
      ref: revision.workRef,
      value: work,
    });
  }
  if (!isStandardReviewResult(review)) {
    appendices.push({
      label: `Anchor ${number} Review Result`,
      ref: revision.reviewRef,
      value: review,
    });
  }
}

function renderDirection(
  lines: string[],
  anchorNumber: number,
  directionNumber: number,
  objectId: string,
  revision: ObjectRevision,
  work: WorkResult,
  review: ReviewResult,
  appendices: RawAppendix[],
): void {
  const content = asRecord(work.content);
  const title = stringValue(content?.name) ?? objectId;
  lines.push(
    "",
    `#### Direction ${anchorNumber}.${directionNumber}: ${title}`,
    "",
    `- Object: \`${objectId}\` revision ${revision.revision}`,
    `- Work outcome: ${work.workOutcome}`,
    `- Review verdict: ${review.reviewVerdict}`,
  );
  appendScalar(lines, "Mechanism", content?.mechanism);
  appendScalar(lines, "Baseline change", content?.baselineChange);
  appendScalar(lines, "Independent review", review.summary);
  appendExpectedEffects(lines, content?.expectedEffects);
  appendStringList(lines, "Tradeoffs", content?.tradeoffs);
  appendStringList(lines, "Failure conditions", content?.failureConditions);
  appendStringList(lines, "Measurement plan", content?.measurementPlan);
  appendEvidence(lines, work.evidence);
  appendReviewCaveats(lines, review.findings);

  if (!isStandardDirectionResult(work)) {
    appendices.push({
      label: `Direction ${anchorNumber}.${directionNumber} Work Result`,
      ref: revision.workRef,
      value: work,
    });
  }
  if (!isStandardReviewResult(review)) {
    appendices.push({
      label: `Direction ${anchorNumber}.${directionNumber} Review Result`,
      ref: revision.reviewRef,
      value: review,
    });
  }
}

function isBoundToAnchor(
  store: FileLoopStore,
  revision: ObjectRevision,
  anchorWorkRef: string,
): boolean {
  try {
    const task = store.readJson<{ inputs?: { boundAnchor?: unknown } }>(
      revision.workTaskRef,
    );
    return task.inputs?.boundAnchor === anchorWorkRef;
  } catch {
    return false;
  }
}

function appendScalar(
  lines: string[],
  label: string,
  value: unknown,
): void {
  const text = stringValue(value);
  if (text !== null) lines.push(`- ${label}: ${text}`);
}

function appendScope6L(lines: string[], value: unknown): void {
  const scope = asRecord(value);
  if (!scope) return;
  const regions = ["L1", "L2", "L3", "L4", "L5", "L6"]
    .map((layer) => {
      const region = stringValue(scope[layer]);
      return region === null ? null : `${layer}: ${region}`;
    })
    .filter((item): item is string => item !== null);
  if (regions.length > 0) lines.push(`- 6L scope: ${regions.join("; ")}`);
}

function appendStringList(
  lines: string[],
  label: string,
  value: unknown,
): void {
  const values = stringList(value);
  if (values.length === 0) return;
  lines.push("", `${label}:`, "", ...values.map((item) => `- ${item}`));
}

function appendExpectedEffects(lines: string[], value: unknown): void {
  if (!Array.isArray(value)) return;
  const effects = value.flatMap((raw) => {
    const item = asRecord(raw);
    if (!item) return [];
    const metric = stringValue(item.metric);
    const effect = stringValue(item.effect);
    const conditions = stringValue(item.conditions);
    if (metric === null && effect === null && conditions === null) return [];
    const core = [metric, effect].filter((part): part is string => part !== null)
      .join(": ");
    return [
      conditions === null ? core : `${core} (conditions: ${conditions})`,
    ];
  });
  if (effects.length > 0) {
    lines.push("", "Expected effects:", "", ...effects.map((item) => `- ${item}`));
  }
}

function appendEvidence(lines: string[], value: unknown): void {
  if (!Array.isArray(value)) return;
  const entries = value.flatMap((raw) => {
    const item = asRecord(raw);
    if (!item) return [];
    const sourceRef = stringValue(item.sourceRef);
    const supports = stringValue(item.supports);
    if (sourceRef === null && supports === null) return [];
    if (sourceRef === null) return [supports!];
    if (supports === null) return [`\`${sourceRef}\``];
    return [`\`${sourceRef}\`: ${supports}`];
  });
  if (entries.length > 0) {
    lines.push("", "Evidence:", "", ...entries.map((item) => `- ${item}`));
  }
}

function appendReviewCaveats(lines: string[], value: unknown): void {
  if (!Array.isArray(value)) return;
  const caveats = value.flatMap((raw) => {
    const finding = asRecord(raw);
    if (!finding || finding.severity !== "NON_BLOCKING") return [];
    const issue = stringValue(finding.issue);
    const basis = stringValue(finding.basis);
    const expected = stringValue(finding.expected);
    const parts = [
      issue,
      basis === null ? null : `basis: ${basis}`,
      expected === null ? null : `expected: ${expected}`,
    ].filter((part): part is string => part !== null);
    return parts.length === 0 ? [] : [parts.join("; ")];
  });
  if (caveats.length > 0) {
    lines.push(
      "",
      "Non-blocking review caveats:",
      "",
      ...caveats.map((item) => `- ${item}`),
    );
  }
}

function appendRawAppendices(
  lines: string[],
  appendices: RawAppendix[],
): void {
  if (appendices.length === 0) return;
  lines.push(
    "",
    "## Raw non-standard Agent results",
    "",
    "These results passed the Controller core-field gate but did not match the recommended Ref template. They are preserved verbatim for audit.",
  );
  for (const appendix of appendices) {
    lines.push(
      "",
      `### ${appendix.label}`,
      "",
      `Source: \`${appendix.ref}\``,
      "",
      "```json",
      safeJson(appendix.value),
      "```",
    );
  }
}

function isStandardAnchorResult(value: WorkResult): boolean {
  const content = asRecord(value.content);
  const scope = asRecord(content?.scope6L);
  return (
    hasOnlyKeys(value, ["workOutcome", "content", "evidence", "unresolved"]) &&
    Boolean(
      content &&
        hasOnlyKeys(content, [
          "name",
          "scenario",
          "baseline",
          "performanceTension",
          "scope6L",
          "constraints",
        ]) &&
        stringValue(content.name) !== null &&
        stringValue(content.scenario) !== null &&
        stringValue(content.baseline) !== null &&
        stringValue(content.performanceTension) !== null &&
        scope &&
        hasOnlyKeys(scope, ["L1", "L2", "L3", "L4", "L5", "L6"]) &&
        Array.isArray(content.constraints) &&
        Array.isArray(value.evidence) &&
        Array.isArray(value.unresolved),
    )
  );
}

function isStandardDirectionResult(value: WorkResult): boolean {
  const content = asRecord(value.content);
  return (
    hasOnlyKeys(value, ["workOutcome", "content", "evidence", "unresolved"]) &&
    Boolean(
      content &&
        hasOnlyKeys(content, [
          "name",
          "mechanism",
          "baselineChange",
          "expectedEffects",
          "tradeoffs",
          "failureConditions",
          "measurementPlan",
        ]) &&
        stringValue(content.name) !== null &&
        stringValue(content.mechanism) !== null &&
        stringValue(content.baselineChange) !== null &&
        Array.isArray(content.expectedEffects) &&
        Array.isArray(content.tradeoffs) &&
        Array.isArray(content.failureConditions) &&
        Array.isArray(content.measurementPlan) &&
        Array.isArray(value.evidence) &&
        Array.isArray(value.unresolved),
    )
  );
}

function isStandardReviewResult(value: ReviewResult): boolean {
  return (
    hasOnlyKeys(value, ["reviewVerdict", "summary", "findings", "queryGaps"]) &&
    stringValue(value.summary) !== null &&
    Array.isArray(value.findings) &&
    Array.isArray(value.queryGaps)
  );
}

function hasOnlyKeys(
  value: Record<string, unknown>,
  keys: string[],
): boolean {
  const allowed = new Set(keys);
  return Object.keys(value).every((key) => allowed.has(key));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.flatMap((item) => {
      const text = stringValue(item);
      return text === null ? [] : [text];
    })
    : [];
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? "null";
  } catch {
    return JSON.stringify({ unrenderable: String(value) }, null, 2);
  }
}
