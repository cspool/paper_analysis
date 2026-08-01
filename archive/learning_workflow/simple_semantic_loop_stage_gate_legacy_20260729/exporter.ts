import { randomUUID } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { WorkflowStore } from "./db/workflow_store.ts";
import { canonicalJson } from "./contracts/index.ts";
import { loadCurrentPlan } from "./workflow/plan_store.ts";

export function exportRun(store: WorkflowStore, runId: string, workDir: string): void {
  const exportDir = resolve(workDir, "exports");
  mkdirSync(exportDir, { recursive: true });
  atomicWriteJson(
    resolve(exportDir, "workflow_state.json"),
    store.readWorkflowState(runId),
  );
  atomicWriteJson(
    resolve(exportDir, "workflow_plan.json"),
    loadCurrentPlan(store, runId),
  );
  exportCanonicalType(store, runId, "topic", resolve(exportDir, "topic.json"), false);
  exportCanonicalType(store, runId, "anchor", resolve(exportDir, "anchors.jsonl"), true);
  exportCanonicalType(store, runId, "direction", resolve(exportDir, "directions.jsonl"), true);
  exportCanonicalType(store, runId, "search_need", resolve(exportDir, "search_needs.jsonl"), true);
  exportCanonicalType(store, runId, "stop_candidate", resolve(exportDir, "stop_candidates.jsonl"), true);
  exportRows(
    store,
    `SELECT payload_json FROM turn_results WHERE run_id = ? AND message_type = 'EVIDENCE_PACKET' AND status = 'committed' ORDER BY result_id`,
    [runId],
    resolve(exportDir, "evidence_packets.jsonl"),
    "payload_json",
  );
  exportRows(
    store,
    `SELECT payload_json FROM turn_results WHERE run_id = ? AND message_type = 'REVIEW_DELTA' AND status = 'committed' ORDER BY result_id`,
    [runId],
    resolve(exportDir, "direction_reviews.jsonl"),
    "payload_json",
  );
  exportRows(
    store,
    `SELECT payload_json FROM turn_results WHERE run_id = ? AND message_type = 'CLOSURE_REVIEW' AND status = 'committed' ORDER BY result_id`,
    [runId],
    resolve(exportDir, "closure_reviews.jsonl"),
    "payload_json",
  );
  for (const [table, filename] of [
    ["tasks", "tasks.jsonl"],
    ["attempts", "attempts.jsonl"],
    ["events", "events.jsonl"],
  ] as const) {
    const rows = store.query(
      `SELECT * FROM ${table} WHERE run_id = ? ORDER BY rowid`,
      runId,
    );
    atomicWriteText(
      resolve(exportDir, filename),
      rows.map(normalizeDatabaseRow).map(canonicalJson).join("\n") +
        (rows.length ? "\n" : ""),
    );
  }
  atomicWriteJson(
    resolve(exportDir, "validation.json"),
    store
      .query(
        "SELECT * FROM validation_reports WHERE run_id = ? ORDER BY created_at",
        runId,
      )
      .map(normalizeDatabaseRow),
  );
  atomicWriteJson(
    resolve(exportDir, "usage.json"),
    store
      .query(
        "SELECT * FROM usage_records WHERE run_id = ? ORDER BY created_at",
        runId,
      )
      .map(normalizeDatabaseRow),
  );
  const artifactsDir = resolve(workDir, "artifacts");
  mkdirSync(artifactsDir, { recursive: true });
  const artifacts = store
    .query(
      "SELECT * FROM artifact_manifests WHERE run_id = ? ORDER BY artifact_id",
      runId,
    )
    .map(normalizeDatabaseRow);
  atomicWriteText(
    resolve(artifactsDir, "manifest.jsonl"),
    artifacts.map(canonicalJson).join("\n") + (artifacts.length ? "\n" : ""),
  );
}

export function atomicWriteJson(path: string, value: unknown): void {
  atomicWriteText(path, `${JSON.stringify(value, null, 2)}\n`);
}

export function atomicWriteText(path: string, value: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${randomUUID()}`;
  writeFileSync(temporary, value, "utf8");
  renameSync(temporary, path);
}

function exportCanonicalType(
  store: WorkflowStore,
  runId: string,
  objectType: string,
  path: string,
  jsonLines: boolean,
): void {
  const rows = store.query(
    `SELECT object_json FROM canonical_objects
     WHERE run_id = ? AND object_type = ? AND active = 1
     ORDER BY object_id`,
    runId,
    objectType,
  );
  const values = rows.map((row) => JSON.parse(String(row.object_json)));
  if (jsonLines) {
    atomicWriteText(
      path,
      values.map(canonicalJson).join("\n") + (values.length ? "\n" : ""),
    );
  } else {
    atomicWriteJson(path, values[0] ?? null);
  }
}

function exportRows(
  store: WorkflowStore,
  sql: string,
  params: unknown[],
  path: string,
  jsonColumn: string,
): void {
  const values = store
    .query(sql, ...params)
    .map((row) => JSON.parse(String(row[jsonColumn])));
  atomicWriteText(
    path,
    values.map(canonicalJson).join("\n") + (values.length ? "\n" : ""),
  );
}

function normalizeDatabaseRow(
  row: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(row).map(([key, value]) => {
      if (
        key.endsWith("_json") &&
        typeof value === "string"
      ) {
        try {
          return [key.slice(0, -5), JSON.parse(value)];
        } catch {
          return [key, value];
        }
      }
      return [key, value];
    }),
  );
}

