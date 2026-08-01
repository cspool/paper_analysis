import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import type {
  ClosureReviewEnvelope,
  ClosureReviewTaskEnvelope,
  DirectionReviewTaskEnvelope,
  EvidencePacket,
  EvidenceReaderTaskEnvelope,
  ReviewDelta,
} from "../contracts/index.ts";
import { sha256Bytes } from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";
import { atomicWriteText, exportRun } from "../exporter.ts";
import {
  renderFinalMarkdown,
  validateRenderedCoverage,
  type FinalRenderModel,
} from "../renderer.ts";
import {
  validateAnchor,
  validateDirection,
  validateSearchNeed,
  validateTopicFrame,
} from "../validators/domain_validator.ts";
import { validateClosureReview } from "../validators/closure_review_validator.ts";
import { validateEvidencePacket } from "../validators/evidence_packet_validator.ts";
import { validateReviewDelta } from "../validators/review_delta_validator.ts";
import { validateExperimentHandoff } from "../security/no_experiment_guard.ts";

export interface FinalizationResult {
  completed: boolean;
  finalPath: string;
  finalSha256: string;
  coverageErrors: string[];
}

export function finalizeAcceptedClosure(
  store: WorkflowStore,
  workDir: string,
  task: ClosureReviewTaskEnvelope,
  review: ClosureReviewEnvelope,
): FinalizationResult {
  const reviewReport = validateClosureReview(review, task);
  if (
    !reviewReport.valid ||
    review.payload.decision !== "accept" ||
    !review.payload.allowsFinalization
  ) {
    throw new Error("ClosureReview does not authorize finalization");
  }
  const before = store.stateBinding(task.runId);
  if (
    before.canonicalRevision !== review.payload.canonicalRevision ||
    before.canonicalRevision !== task.payload.currentCanonicalRevision
  ) {
    throw new Error("canonical revision changed after Closure acceptance");
  }
  validateCanonicalProjection(store, task);
  const model = buildFinalRenderModel(store, task);
  const markdown = renderFinalMarkdown(model);
  const coverageErrors = validateRenderedCoverage(markdown, model);
  if (coverageErrors.length) {
    return {
      completed: false,
      finalPath: resolve(workDir, "final.md"),
      finalSha256: sha256Bytes(markdown),
      coverageErrors,
    };
  }
  const finalPath = resolve(workDir, "final.md");
  atomicWriteText(finalPath, markdown);
  const finalSha256 = sha256Bytes(readFileSync(finalPath));
  const artifactId = `final-${finalSha256.slice(0, 24)}`;
  const existing = store.db
    .prepare(
      "SELECT artifact_id FROM artifact_manifests WHERE run_id = ? AND artifact_id = ?",
    )
    .get(task.runId, artifactId);
  if (!existing) {
    store.registerArtifact(task.runId, {
      artifactId,
      kind: "final_markdown",
      relativePath: "final.md",
      sha256: finalSha256,
      sizeBytes: statSync(finalPath).size,
      trustClass: "canonical",
    });
  }
  const unchanged = store.stateBinding(task.runId);
  if (
    unchanged.snapshotVersion !== before.snapshotVersion ||
    unchanged.canonicalRevision !== before.canonicalRevision ||
    unchanged.eventCursor !== before.eventCursor ||
    unchanged.workflowPlanRevision !== before.workflowPlanRevision
  ) {
    throw new Error("state changed during deterministic final rendering");
  }
  store.markCompleted(task.runId, unchanged, artifactId);
  exportRun(store, task.runId, workDir);
  return { completed: true, finalPath, finalSha256, coverageErrors: [] };
}

function validateCanonicalProjection(
  store: WorkflowStore,
  task: ClosureReviewTaskEnvelope,
): void {
  const canonical = store.query(
    `SELECT object_type, object_id, revision, object_json
     FROM canonical_objects
     WHERE run_id = ? AND active = 1
     ORDER BY object_type, object_id`,
    task.runId,
  );
  const reports = [];
  const refKeys = new Set<string>();
  for (const row of canonical) {
    const type = String(row.object_type);
    const value = JSON.parse(String(row.object_json));
    refKeys.add(
      `${type}\0${String(row.object_id)}\0${Number(row.revision)}`,
    );
    if (type === "topic") reports.push(validateTopicFrame(value));
    if (type === "anchor") reports.push(validateAnchor(value));
    if (type === "direction") {
      reports.push(validateDirection(value, task.payload.topic));
    }
    if (type === "search_need") reports.push(validateSearchNeed(value));
    if (type === "experiment_handoff") {
      reports.push(validateExperimentHandoff(value));
    }
  }
  reports.push(validateTopicFrame(task.payload.topic));
  for (const anchor of task.payload.anchors) {
    if (!refKeys.has(refKey(anchor.anchorRef))) {
      throw new Error(`unresolved Anchor ref ${anchor.anchorRef.objectId}`);
    }
  }
  for (const direction of task.payload.directions) {
    if (!refKeys.has(refKey(direction.directionRef))) {
      throw new Error(`unresolved Direction ref ${direction.directionRef.objectId}`);
    }
  }
  const results = store.query(
    `SELECT r.message_type, r.payload_json, t.task_json
     FROM turn_results r
     JOIN tasks t ON t.task_id = r.task_id
     WHERE r.run_id = ? AND r.status = 'committed'
     ORDER BY r.result_id`,
    task.runId,
  );
  for (const row of results) {
    const result = JSON.parse(String(row.payload_json));
    const sourceTask = JSON.parse(String(row.task_json));
    if (row.message_type === "EVIDENCE_PACKET") {
      reports.push(
        validateEvidencePacket(
          result,
          sourceTask as EvidenceReaderTaskEnvelope,
        ),
      );
    } else if (row.message_type === "REVIEW_DELTA") {
      reports.push(
        validateReviewDelta(
          result,
          sourceTask as DirectionReviewTaskEnvelope,
        ),
      );
    } else if (row.message_type === "CLOSURE_REVIEW") {
      reports.push(
        validateClosureReview(
          result,
          sourceTask as ClosureReviewTaskEnvelope,
        ),
      );
    }
  }
  const unresolved = store.query(
    `SELECT validation_report_id FROM validation_reports
     WHERE run_id = ? AND valid = 0 AND resolved_by_id IS NULL`,
    task.runId,
  );
  if (unresolved.length) {
    throw new Error(
      `unresolved validation failures: ${unresolved
        .map((row) => row.validation_report_id)
        .join(", ")}`,
    );
  }
  const errors = reports.flatMap((report) => report.errors);
  if (errors.length) {
    throw new Error(`full canonical validation failed: ${JSON.stringify(errors)}`);
  }
}

function refKey(ref: {
  objectType: string;
  objectId: string;
  revision: number;
}): string {
  return `${ref.objectType}\0${ref.objectId}\0${ref.revision}`;
}

function buildFinalRenderModel(
  store: WorkflowStore,
  task: ClosureReviewTaskEnvelope,
): FinalRenderModel {
  const activeObjects = store.query(
    `SELECT object_type, object_json FROM canonical_objects
     WHERE run_id = ? AND active = 1 ORDER BY object_type, object_id`,
    task.runId,
  );
  const objectsByType = new Map<string, unknown[]>();
  for (const row of activeObjects) {
    const type = String(row.object_type);
    objectsByType.set(type, [
      ...(objectsByType.get(type) ?? []),
      JSON.parse(String(row.object_json)),
    ]);
  }
  const results = store.query(
    `SELECT message_type, payload_json FROM turn_results
     WHERE run_id = ? AND status = 'committed' ORDER BY result_id`,
    task.runId,
  );
  const evidencePackets = results
    .filter((row) => row.message_type === "EVIDENCE_PACKET")
    .map((row) => {
      const envelope = JSON.parse(String(row.payload_json));
      return (envelope.payload ?? envelope) as EvidencePacket;
    });
  const directionReviews = results
    .filter((row) => row.message_type === "REVIEW_DELTA")
    .map((row) => {
      const envelope = JSON.parse(String(row.payload_json));
      return (envelope.payload ?? envelope) as ReviewDelta;
    });
  return {
    runId: task.runId,
    canonicalRevision: task.payload.currentCanonicalRevision,
    topic:
      (objectsByType.get("topic")?.[0] as FinalRenderModel["topic"]) ??
      task.payload.topic,
    anchors: (objectsByType.get("anchor") ?? []) as FinalRenderModel["anchors"],
    directions: (objectsByType.get("direction") ??
      []) as FinalRenderModel["directions"],
    evidencePackets,
    directionReviews,
    unresolvedNeeds: (objectsByType.get("search_need") ?? []).filter(
      (need) => (need as { status: string }).status === "pending",
    ) as FinalRenderModel["unresolvedNeeds"],
    contradictions: (objectsByType.get("contradiction") ?? []).map(
      (item) => {
        const value = item as Record<string, unknown>;
        return {
          contradictionId: String(value.contradictionId),
          summary: String(value.summary ?? ""),
          disposition:
            value.disposition === null || value.disposition === undefined
              ? null
              : String(value.disposition),
        };
      },
    ),
    experimentHandoffs: (objectsByType.get("experiment_handoff") ??
      []) as FinalRenderModel["experimentHandoffs"],
  };
}
