import type { DatabaseSync } from "node:sqlite";

export const DATABASE_SCHEMA_VERSION = 1;

export function migrate(db: DatabaseSync): void {
  db.exec("PRAGMA foreign_keys = ON");
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA synchronous = FULL");
  db.exec("PRAGMA busy_timeout = 5000");
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    ) STRICT;

    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY,
      workflow_id TEXT NOT NULL UNIQUE,
      objective TEXT NOT NULL,
      objective_hash TEXT NOT NULL,
      acceptance_criteria_json TEXT NOT NULL,
      acceptance_criteria_hash TEXT NOT NULL,
      lifecycle TEXT NOT NULL,
      snapshot_version INTEGER NOT NULL,
      canonical_revision INTEGER NOT NULL,
      event_cursor INTEGER NOT NULL,
      workflow_plan_revision INTEGER NOT NULL,
      current_stage_id TEXT,
      active_focus_ref_json TEXT,
      config_json TEXT NOT NULL,
      pause_or_block_reason TEXT,
      completed_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      CHECK (snapshot_version >= 0),
      CHECK (canonical_revision >= 0),
      CHECK (event_cursor >= 0),
      CHECK (workflow_plan_revision >= 1)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS snapshots (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      snapshot_version INTEGER NOT NULL,
      state_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY (run_id, snapshot_version)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS events (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      event_cursor INTEGER NOT NULL,
      snapshot_version INTEGER NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY (run_id, event_cursor)
    ) STRICT;

    CREATE TRIGGER IF NOT EXISTS events_no_update
    BEFORE UPDATE ON events BEGIN
      SELECT RAISE(ABORT, 'events are append-only');
    END;
    CREATE TRIGGER IF NOT EXISTS events_no_delete
    BEFORE DELETE ON events BEGIN
      SELECT RAISE(ABORT, 'events are append-only');
    END;

    CREATE TABLE IF NOT EXISTS workflow_plans (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      revision INTEGER NOT NULL,
      objective_hash TEXT NOT NULL,
      acceptance_criteria_hash TEXT NOT NULL,
      status TEXT NOT NULL,
      plan_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      PRIMARY KEY (run_id, revision)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS workflow_plan_nodes (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      plan_revision INTEGER NOT NULL,
      stage_id TEXT NOT NULL,
      stage_type TEXT NOT NULL,
      execution_kind TEXT NOT NULL,
      role TEXT,
      contract_id TEXT NOT NULL,
      gate_id TEXT NOT NULL,
      lifecycle TEXT NOT NULL,
      created_at_snapshot_version INTEGER NOT NULL,
      superseded_reason TEXT,
      PRIMARY KEY (run_id, plan_revision, stage_id),
      FOREIGN KEY (run_id, plan_revision)
        REFERENCES workflow_plans(run_id, revision)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS workflow_plan_edges (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      plan_revision INTEGER NOT NULL,
      dependency_id TEXT NOT NULL,
      predecessor_stage_id TEXT NOT NULL,
      successor_stage_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      PRIMARY KEY (run_id, plan_revision, dependency_id),
      FOREIGN KEY (run_id, plan_revision)
        REFERENCES workflow_plans(run_id, revision)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS stage_contracts (
      contract_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      stage_id TEXT NOT NULL,
      revision INTEGER NOT NULL,
      stage_type TEXT NOT NULL,
      role TEXT,
      defined_at_snapshot_version INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      contract_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE (run_id, stage_id, revision),
      UNIQUE (run_id, sha256)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS gate_definitions (
      gate_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      stage_id TEXT NOT NULL,
      revision INTEGER NOT NULL,
      defined_at_snapshot_version INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      gate_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE (run_id, stage_id, revision),
      UNIQUE (run_id, sha256)
    ) STRICT;

    CREATE TRIGGER IF NOT EXISTS stage_contracts_immutable_update
    BEFORE UPDATE ON stage_contracts BEGIN
      SELECT RAISE(ABORT, 'frozen StageContract is immutable');
    END;
    CREATE TRIGGER IF NOT EXISTS stage_contracts_immutable_delete
    BEFORE DELETE ON stage_contracts BEGIN
      SELECT RAISE(ABORT, 'frozen StageContract is immutable');
    END;
    CREATE TRIGGER IF NOT EXISTS gate_definitions_immutable_update
    BEFORE UPDATE ON gate_definitions BEGIN
      SELECT RAISE(ABORT, 'frozen GateDefinition is immutable');
    END;
    CREATE TRIGGER IF NOT EXISTS gate_definitions_immutable_delete
    BEFORE DELETE ON gate_definitions BEGIN
      SELECT RAISE(ABORT, 'frozen GateDefinition is immutable');
    END;

    CREATE TABLE IF NOT EXISTS gate_results (
      gate_result_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      stage_id TEXT NOT NULL,
      gate_id TEXT NOT NULL REFERENCES gate_definitions(gate_id),
      result_id TEXT,
      passed INTEGER NOT NULL,
      report_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      CHECK (passed IN (0, 1))
    ) STRICT;

    CREATE TABLE IF NOT EXISTS tasks (
      task_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      stage_id TEXT NOT NULL,
      role TEXT NOT NULL,
      input_message_type TEXT NOT NULL,
      expected_output_message_type TEXT NOT NULL,
      state_binding_json TEXT NOT NULL,
      input_hash TEXT NOT NULL,
      stage_contract_hash TEXT NOT NULL,
      skill_hash TEXT NOT NULL,
      schema_manifest_hash TEXT NOT NULL,
      task_json TEXT NOT NULL,
      status TEXT NOT NULL,
      output_attempt_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE (run_id, stage_id, task_id)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS attempts (
      attempt_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      task_id TEXT NOT NULL REFERENCES tasks(task_id),
      attempt_no INTEGER NOT NULL,
      role TEXT NOT NULL,
      logical_effort TEXT NOT NULL,
      provider_wire_effort TEXT NOT NULL,
      provider_thread_id TEXT,
      provider_turn_id TEXT,
      status TEXT NOT NULL,
      raw_response_artifact_id TEXT,
      error_code TEXT,
      started_at TEXT NOT NULL,
      finished_at TEXT,
      UNIQUE (task_id, attempt_no)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS decision_proposals (
      proposal_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      task_id TEXT NOT NULL,
      attempt_id TEXT NOT NULL,
      expected_state_json TEXT NOT NULL,
      decision_input_hash TEXT NOT NULL,
      decision TEXT NOT NULL,
      proposal_json TEXT NOT NULL,
      validation_report_id TEXT,
      status TEXT NOT NULL,
      rejection_code TEXT,
      created_at TEXT NOT NULL
    ) STRICT;

    CREATE TABLE IF NOT EXISTS turn_results (
      result_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      task_id TEXT NOT NULL REFERENCES tasks(task_id),
      attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
      stage_id TEXT NOT NULL,
      role TEXT NOT NULL,
      message_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      payload_hash TEXT NOT NULL,
      status TEXT NOT NULL,
      committed_snapshot_version INTEGER,
      committed_at TEXT,
      UNIQUE (task_id, payload_hash)
    ) STRICT;
    CREATE UNIQUE INDEX IF NOT EXISTS one_committed_result_per_task
      ON turn_results(task_id);

    CREATE TABLE IF NOT EXISTS canonical_objects (
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      object_type TEXT NOT NULL,
      object_id TEXT NOT NULL,
      revision INTEGER NOT NULL,
      object_json TEXT NOT NULL,
      object_hash TEXT NOT NULL,
      active INTEGER NOT NULL,
      source_result_id TEXT,
      created_at TEXT NOT NULL,
      PRIMARY KEY (run_id, object_type, object_id, revision),
      CHECK (active IN (0, 1))
    ) STRICT;
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_object_revision
      ON canonical_objects(run_id, object_type, object_id)
      WHERE active = 1;

    CREATE TABLE IF NOT EXISTS result_consumptions (
      result_id TEXT PRIMARY KEY REFERENCES turn_results(result_id),
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      commit_id TEXT NOT NULL UNIQUE,
      delta_id TEXT,
      consumed_snapshot_version INTEGER NOT NULL,
      consumed_at TEXT NOT NULL
    ) STRICT;

    CREATE TABLE IF NOT EXISTS artifact_manifests (
      artifact_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      kind TEXT NOT NULL,
      relative_path TEXT NOT NULL,
      sha256 TEXT NOT NULL,
      size_bytes INTEGER NOT NULL,
      trust_class TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE (run_id, relative_path, sha256, trust_class)
    ) STRICT;

    CREATE TABLE IF NOT EXISTS validation_reports (
      validation_report_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      task_id TEXT,
      attempt_id TEXT,
      validator_version TEXT NOT NULL,
      valid INTEGER NOT NULL,
      report_json TEXT NOT NULL,
      resolved_by_id TEXT,
      created_at TEXT NOT NULL,
      CHECK (valid IN (0, 1))
    ) STRICT;

    CREATE TABLE IF NOT EXISTS usage_records (
      usage_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
      input_tokens INTEGER NOT NULL,
      cached_input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      reasoning_output_tokens INTEGER NOT NULL,
      tool_calls INTEGER NOT NULL,
      elapsed_ms INTEGER NOT NULL,
      created_at TEXT NOT NULL
    ) STRICT;

    CREATE TABLE IF NOT EXISTS operator_requests (
      request_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL REFERENCES runs(run_id),
      request_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      applied_at TEXT
    ) STRICT;

    CREATE TABLE IF NOT EXISTS run_locks (
      run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
      owner_id TEXT NOT NULL,
      acquired_at TEXT NOT NULL,
      heartbeat_at TEXT NOT NULL
    ) STRICT;
  `);

  const row = db
    .prepare("SELECT value FROM schema_meta WHERE key = 'schema_version'")
    .get() as { value?: string } | undefined;
  if (!row) {
    db.prepare(
      "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
    ).run(String(DATABASE_SCHEMA_VERSION));
  } else if (Number(row.value) !== DATABASE_SCHEMA_VERSION) {
    throw new Error(
      `unsupported database schema ${row.value}; expected ${DATABASE_SCHEMA_VERSION}`,
    );
  }
}
