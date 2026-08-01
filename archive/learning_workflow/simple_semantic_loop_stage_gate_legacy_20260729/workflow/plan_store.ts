import { randomUUID } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import type {
  GateDefinition,
  GateDefinitionDraft,
  StageContract,
  StageContractDraft,
  StageNode,
  StateBinding,
  WorkflowPlan,
  WorkflowPlanPatch,
} from "../contracts/index.ts";
import {
  MAX_WORKFLOW_PLAN_REVISIONS,
  STAGE_REGISTRY,
  canonicalJson,
  canonicalSha256,
} from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";
import {
  validateStageContractDraft,
} from "../validators/domain_validator.ts";
import {
  compileGateDraft,
  GATE_COMPILER_POLICY_VERSION,
} from "../stages/gate_compiler.ts";
import { GATE_EVALUATOR_VERSION } from "../stages/gate_engine.ts";

export interface FrozenStagePair {
  node: StageNode;
  contract: StageContract;
  gate: GateDefinition;
}

export function freezeStageDraft(
  draft: StageContractDraft,
  gateDraft: GateDefinitionDraft,
  snapshotVersion: number,
  stageId = `stage-${randomUUID()}`,
): FrozenStagePair {
  const validation = validateStageContractDraft(draft);
  if (!validation.valid) {
    throw new Error(
      `invalid StageContractDraft: ${canonicalJson(validation.errors)}`,
    );
  }
  if (draft.proposalLocalStageKey !== gateDraft.proposalLocalStageKey) {
    throw new Error("Stage and Gate proposal-local keys differ");
  }
  const gateCompilation = compileGateDraft(draft, gateDraft);
  if (!gateCompilation.report.valid) {
    throw new Error(
      `invalid GateDefinitionDraft: ${canonicalJson(
        gateCompilation.report.errors,
      )}`,
    );
  }
  const contractWithoutHash = {
    ...structuredClone(draft),
    contractId: `contract-${randomUUID()}`,
    stageId,
    revision: 1,
    definedAtSnapshotVersion: snapshotVersion,
  };
  const contract: StageContract = {
    ...contractWithoutHash,
    sha256: canonicalSha256(contractWithoutHash),
  };
  const gateWithoutHash = {
    ...gateCompilation.compiled,
    gateId: `gate-${randomUUID()}`,
    stageId,
    stageContractHash: contract.sha256,
    proposedCriteriaSha256: canonicalSha256(
      gateDraft.mechanicalChecks,
    ),
    compilerPolicyVersion: GATE_COMPILER_POLICY_VERSION,
    evaluatorVersion: GATE_EVALUATOR_VERSION,
    revision: 1,
    definedAtSnapshotVersion: snapshotVersion,
  };
  const gate: GateDefinition = {
    ...gateWithoutHash,
    sha256: canonicalSha256(gateWithoutHash),
  };
  const node: StageNode = {
    stageId,
    stageType: draft.stageType,
    executionKind: draft.executionKind,
    role: draft.role,
    contractId: contract.contractId,
    gateId: gate.gateId,
    lifecycle: "runnable",
    createdAtSnapshotVersion: snapshotVersion,
    supersededReason: null,
  };
  return { node, contract, gate };
}

export function validateWorkflowPlan(plan: WorkflowPlan): string[] {
  const errors: string[] = [];
  const ids = new Set<string>();
  for (const node of plan.stageNodes) {
    if (ids.has(node.stageId)) errors.push(`duplicate stage ${node.stageId}`);
    ids.add(node.stageId);
    const registered = STAGE_REGISTRY[node.stageType];
    if (
      !registered ||
      registered.executionKind !== node.executionKind ||
      registered.role !== node.role
    ) {
      errors.push(`Stage registry mismatch for ${node.stageId}`);
    }
  }
  const dependencyIds = new Set<string>();
  for (const edge of plan.dependencies) {
    if (dependencyIds.has(edge.dependencyId)) {
      errors.push(`duplicate dependency ${edge.dependencyId}`);
    }
    dependencyIds.add(edge.dependencyId);
    if (
      !ids.has(edge.predecessorStageId) ||
      !ids.has(edge.successorStageId)
    ) {
      errors.push(`dependency ${edge.dependencyId} references unknown Stage`);
    }
    if (edge.predecessorStageId === edge.successorStageId) {
      errors.push(`dependency ${edge.dependencyId} is a self edge`);
    }
  }
  if (hasCycle(plan)) errors.push("WorkflowPlan DAG contains a cycle");
  return errors;
}

export function applyPlanPatch(
  current: WorkflowPlan,
  patch: WorkflowPlanPatch,
  snapshotVersion: number,
): {
  plan: WorkflowPlan;
  frozenStages: FrozenStagePair[];
} {
  if (current.revision >= MAX_WORKFLOW_PLAN_REVISIONS) {
    throw new Error(
      `maximum WorkflowPlan revision ${MAX_WORKFLOW_PLAN_REVISIONS} reached`,
    );
  }
  if (
    patch.expectedPlanRevision !== current.revision ||
    patch.objectiveHash !== current.objectiveHash ||
    patch.acceptanceCriteriaHash !== current.acceptanceCriteriaHash
  ) {
    throw new Error("plan patch is stale or changes immutable hashes");
  }
  const nodes = structuredClone(current.stageNodes);
  const edges = structuredClone(current.dependencies);
  const frozenStages: FrozenStagePair[] = [];
  for (const operation of patch.operations) {
    switch (operation.op) {
      case "add_stage": {
        const registered = STAGE_REGISTRY[operation.stage.stageType];
        if (
          ["controller_trigger", "controller_closure", "controller_finalization"].includes(
            registered.creationAuthority,
          )
        ) {
          throw new Error(
            `Workflow patch cannot create ${operation.stage.stageType}`,
          );
        }
        const frozen = freezeStageDraft(
          operation.stage.contract,
          operation.stage.gate,
          snapshotVersion,
        );
        if (
          frozen.node.stageType !== operation.stage.stageType ||
          frozen.node.executionKind !== operation.stage.executionKind ||
          frozen.node.role !== operation.stage.role
        ) {
          throw new Error("StageNodeDraft and contract registry binding differ");
        }
        frozen.node.lifecycle =
          operation.stage.dependsOnStageIds.length === 0 ? "runnable" : "frozen";
        nodes.push(frozen.node);
        frozenStages.push(frozen);
        for (const predecessorStageId of operation.stage.dependsOnStageIds) {
          edges.push({
            dependencyId: `dependency-${randomUUID()}`,
            predecessorStageId,
            successorStageId: frozen.node.stageId,
            kind: "requires_committed",
          });
        }
        break;
      }
      case "supersede_stage": {
        const node = nodes.find((candidate) => candidate.stageId === operation.stageId);
        if (!node) throw new Error(`unknown Stage ${operation.stageId}`);
        if (
          ![
            "draft_proposed",
            "validated",
            "frozen",
            "runnable",
            "failed",
            "blocked",
          ].includes(node.lifecycle)
        ) {
          throw new Error(`cannot supersede executed Stage ${node.stageId}`);
        }
        node.lifecycle = "superseded";
        node.supersededReason = operation.reason;
        break;
      }
      case "add_dependency":
        edges.push(structuredClone(operation.dependency));
        break;
      case "remove_dependency": {
        const index = edges.findIndex(
          (edge) => edge.dependencyId === operation.dependencyId,
        );
        if (index < 0) throw new Error(`unknown dependency ${operation.dependencyId}`);
        edges.splice(index, 1);
        break;
      }
    }
  }
  const plan: WorkflowPlan = {
    ...structuredClone(current),
    revision: current.revision + 1,
    stageNodes: nodes,
    dependencies: edges,
  };
  const errors = validateWorkflowPlan(plan);
  if (errors.length) throw new Error(errors.join("; "));
  return { plan, frozenStages };
}

export function persistPlanRevision(
  store: WorkflowStore,
  runId: string,
  expected: StateBinding,
  plan: WorkflowPlan,
  frozenStages: readonly FrozenStagePair[],
  eventType = "workflow_plan_revised",
  extraApply?: (db: DatabaseSync, nextSnapshotVersion: number) => void,
): StateBinding {
  if (plan.revision !== expected.workflowPlanRevision + 1) {
    throw new Error("new plan revision must be exactly current+1");
  }
  return store.casTransition(
    runId,
    expected,
    {
      workflowPlanRevisionDelta: 1,
      lifecycle: "running",
      eventType,
      eventPayload: {
        planRevision: plan.revision,
        stageIds: frozenStages.map((stage) => stage.node.stageId),
      },
    },
    (db, nextSnapshotVersion) => {
      const now = new Date().toISOString();
      db.prepare(
        `INSERT INTO workflow_plans(
           run_id, revision, objective_hash, acceptance_criteria_hash,
           status, plan_json, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        runId,
        plan.revision,
        plan.objectiveHash,
        plan.acceptanceCriteriaHash,
        plan.planStatus,
        canonicalJson(plan),
        now,
      );
      for (const node of plan.stageNodes) {
        db.prepare(
          `INSERT INTO workflow_plan_nodes(
             run_id, plan_revision, stage_id, stage_type, execution_kind, role,
             contract_id, gate_id, lifecycle, created_at_snapshot_version,
             superseded_reason
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          runId,
          plan.revision,
          node.stageId,
          node.stageType,
          node.executionKind,
          node.role,
          node.contractId,
          node.gateId,
          node.lifecycle,
          node.createdAtSnapshotVersion,
          node.supersededReason,
        );
      }
      for (const edge of plan.dependencies) {
        db.prepare(
          `INSERT INTO workflow_plan_edges(
             run_id, plan_revision, dependency_id, predecessor_stage_id,
             successor_stage_id, kind
           ) VALUES (?, ?, ?, ?, ?, ?)`,
        ).run(
          runId,
          plan.revision,
          edge.dependencyId,
          edge.predecessorStageId,
          edge.successorStageId,
          edge.kind,
        );
      }
      for (const frozen of frozenStages) {
        db.prepare(
          `INSERT INTO stage_contracts(
             contract_id, run_id, stage_id, revision, stage_type, role,
             defined_at_snapshot_version, sha256, contract_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          frozen.contract.contractId,
          runId,
          frozen.contract.stageId,
          frozen.contract.revision,
          frozen.contract.stageType,
          frozen.contract.role,
          frozen.contract.definedAtSnapshotVersion,
          frozen.contract.sha256,
          canonicalJson(frozen.contract),
          now,
        );
        db.prepare(
          `INSERT INTO gate_definitions(
             gate_id, run_id, stage_id, revision, defined_at_snapshot_version,
             sha256, gate_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          frozen.gate.gateId,
          runId,
          frozen.gate.stageId,
          frozen.gate.revision,
          frozen.gate.definedAtSnapshotVersion,
          frozen.gate.sha256,
          canonicalJson(frozen.gate),
          now,
        );
      }
      for (const node of plan.stageNodes.filter(
        (candidate) => candidate.lifecycle === "superseded",
      )) {
        db.prepare(
          `UPDATE tasks SET status = 'superseded', updated_at = ?
           WHERE run_id = ? AND stage_id = ?
           AND status NOT IN ('committed', 'resolved', 'cancelled')`,
        ).run(now, runId, node.stageId);
        db.prepare(
          `UPDATE validation_reports SET resolved_by_id = ?
           WHERE run_id = ? AND task_id IN (
             SELECT task_id FROM tasks WHERE run_id = ? AND stage_id = ?
           ) AND valid = 0 AND resolved_by_id IS NULL`,
        ).run(
          `plan-revision-${plan.revision}`,
          runId,
          runId,
          node.stageId,
        );
      }
      extraApply?.(db, nextSnapshotVersion);
    },
  );
}

export function loadCurrentPlan(
  store: WorkflowStore,
  runId: string,
): WorkflowPlan {
  const run = store.getRun(runId);
  const row = store.db
    .prepare(
      "SELECT plan_json FROM workflow_plans WHERE run_id = ? AND revision = ?",
    )
    .get(runId, run.workflowPlanRevision) as { plan_json: string } | undefined;
  if (!row) throw new Error("current WorkflowPlan row is missing");
  const plan = JSON.parse(row.plan_json) as WorkflowPlan;
  const nodeRows = store.db
    .prepare(
      `SELECT * FROM workflow_plan_nodes
       WHERE run_id = ? AND plan_revision = ? ORDER BY stage_id`,
    )
    .all(runId, run.workflowPlanRevision) as Array<Record<string, unknown>>;
  if (nodeRows.length) {
    plan.stageNodes = nodeRows.map((row) => ({
      stageId: String(row.stage_id),
      stageType: row.stage_type as StageNode["stageType"],
      executionKind: row.execution_kind as StageNode["executionKind"],
      role: (row.role ?? null) as StageNode["role"],
      contractId: String(row.contract_id),
      gateId: String(row.gate_id),
      lifecycle: row.lifecycle as StageNode["lifecycle"],
      createdAtSnapshotVersion: Number(row.created_at_snapshot_version),
      supersededReason:
        row.superseded_reason === null ? null : String(row.superseded_reason),
    }));
  }
  return plan;
}

function hasCycle(plan: WorkflowPlan): boolean {
  const outgoing = new Map<string, string[]>();
  const indegree = new Map(plan.stageNodes.map((node) => [node.stageId, 0]));
  for (const edge of plan.dependencies) {
    outgoing.set(edge.predecessorStageId, [
      ...(outgoing.get(edge.predecessorStageId) ?? []),
      edge.successorStageId,
    ]);
    indegree.set(
      edge.successorStageId,
      (indegree.get(edge.successorStageId) ?? 0) + 1,
    );
  }
  const queue = [...indegree.entries()]
    .filter(([, degree]) => degree === 0)
    .map(([stageId]) => stageId);
  let visited = 0;
  while (queue.length) {
    const current = queue.shift()!;
    visited += 1;
    for (const next of outgoing.get(current) ?? []) {
      const degree = (indegree.get(next) ?? 0) - 1;
      indegree.set(next, degree);
      if (degree === 0) queue.push(next);
    }
  }
  return visited !== plan.stageNodes.length;
}
