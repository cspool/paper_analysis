import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { test } from "node:test";
import {
  canonicalSha256,
  type WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  CasConflictError,
  WorkflowStore,
} from "../db/workflow_store.ts";
import { SemanticLoopController } from "../controller.ts";
import { ScriptedFreshTurnRuntime } from "../turns/runtime.ts";
import {
  EVIDENCE_BUDGET,
  ZERO_BUDGET,
  makeTopic,
} from "./fixtures/factory.ts";
import {
  applyPlanPatch,
  loadCurrentPlan,
} from "../workflow/plan_store.ts";
import { normalizeAgentOutput } from "../failure_handling/output_normalizer.ts";

function temporaryWorkDir(): string {
  return mkdtempSync(resolve(tmpdir(), "simple-semantic-loop-"));
}

test("SQLite store CAS rejects stale state and preserves append-only events", () => {
  const workDir = temporaryWorkDir();
  const store = new WorkflowStore(resolve(workDir, "workflow.db"));
  try {
    store.createRun({
      runId: "run-1",
      workflowId: "workflow-1",
      objective: "objective",
      acceptanceCriteria: ["criterion"],
      config: {},
    });
    const initial = store.stateBinding("run-1");
    const next = store.casTransition("run-1", initial, {
      lifecycle: "running",
      eventType: "test_transition",
      eventPayload: { value: 1 },
    });
    assert.equal(next.snapshotVersion, 1);
    assert.equal(next.eventCursor, 1);
    assert.throws(
      () =>
        store.casTransition("run-1", initial, {
          lifecycle: "running",
          eventType: "stale",
          eventPayload: {},
        }),
      CasConflictError,
    );
    assert.throws(() =>
      store.db.exec(
        "UPDATE events SET event_type = 'tampered' WHERE run_id = 'run-1'",
      ),
    );
  } finally {
    store.close();
  }
});

test("frozen Stage/Gate definitions reject UPDATE and DELETE", () => {
  const workDir = temporaryWorkDir();
  const store = new WorkflowStore(resolve(workDir, "workflow.db"));
  try {
    store.createRun({
      runId: "run-1",
      workflowId: "workflow-1",
      objective: "objective",
      acceptanceCriteria: ["criterion"],
      config: {},
    });
    const plan = loadCurrentPlan(store, "run-1");
    const patch = {
      expectedPlanRevision: 1,
      objectiveHash: plan.objectiveHash,
      acceptanceCriteriaHash: plan.acceptanceCriteriaHash,
      rationale: "add a bounded script stage",
      operations: [
        {
          op: "add_stage" as const,
          stage: {
            proposalLocalStageKey: "local-1",
            stageType: "SCRIPT_APPLY_TOPIC_FRAME" as const,
            executionKind: "SCRIPT_TRANSITION" as const,
            role: null,
            objective: "apply TopicFrame",
            dependsOnStageIds: [],
            contract: {
              proposalLocalStageKey: "local-1",
              stageType: "SCRIPT_APPLY_TOPIC_FRAME" as const,
              objective: "apply TopicFrame",
              scope: [],
              executionKind: "SCRIPT_TRANSITION" as const,
              role: null,
              requiredInputs: [],
              expectedOutputMessageType: null,
              requestedTools: [],
              requestedPaths: [],
              prohibitedActions: ["experiment execution"],
              budget: structuredClone(ZERO_BUDGET),
            },
            gate: {
              proposalLocalStageKey: "local-1",
              mechanicalChecks: [],
              semanticEvaluation: {
                required: false,
                evaluatorRole: null,
                rubricId: null,
                inputProjection: [],
                expectedOutputMessageType: null,
              },
            },
          },
        },
      ],
    };
    const applied = applyPlanPatch(plan, patch, 0);
    const stage = applied.frozenStages[0]!;
    const now = new Date().toISOString();
    store.db.prepare(
      `INSERT INTO stage_contracts(
         contract_id, run_id, stage_id, revision, stage_type, role,
         defined_at_snapshot_version, sha256, contract_json, created_at
       ) VALUES (?, 'run-1', ?, 1, ?, NULL, 0, ?, ?, ?)`,
    ).run(
      stage.contract.contractId,
      stage.contract.stageId,
      stage.contract.stageType,
      stage.contract.sha256,
      JSON.stringify(stage.contract),
      now,
    );
    store.db.prepare(
      `INSERT INTO gate_definitions(
         gate_id, run_id, stage_id, revision, defined_at_snapshot_version,
         sha256, gate_json, created_at
       ) VALUES (?, 'run-1', ?, 1, 0, ?, ?, ?)`,
    ).run(
      stage.gate.gateId,
      stage.gate.stageId,
      stage.gate.sha256,
      JSON.stringify(stage.gate),
      now,
    );
    assert.throws(() =>
      store.db
        .prepare("UPDATE stage_contracts SET role = 'x' WHERE contract_id = ?")
        .run(stage.contract.contractId),
    );
    assert.throws(() =>
      store.db
        .prepare("DELETE FROM gate_definitions WHERE gate_id = ?")
        .run(stage.gate.gateId),
    );
  } finally {
    store.close();
  }
});

test("output normalizer removes only permitted wrappers and rejects two JSON values", () => {
  const one = normalizeAgentOutput("\uFEFF```json\r\n{\"a\":1}\r\n```\r\n");
  assert.deepEqual(one.parsed, { a: 1 });
  assert.ok(one.transformations.includes("removed_bom"));
  assert.ok(one.transformations.includes("removed_single_fence"));

  const unique = normalizeAgentOutput("result follows: {\"a\":1} done");
  assert.deepEqual(unique.parsed, { a: 1 });
  assert.ok(unique.transformations.includes("extracted_unique_json"));

  const two = normalizeAgentOutput("{\"a\":1}\n{\"b\":2}");
  assert.equal(two.parsed, null);
  assert.equal(two.errorCode, "multiple_json_values");
});

test("Controller uses Workflow Turn only at registered semantic points and applies Topic deterministically", async () => {
  const workDir = temporaryWorkDir();
  const store = new WorkflowStore(resolve(workDir, "workflow.db"));
  store.createRun({
    runId: "run-1",
    workflowId: "workflow-1",
    objective: "Build a bounded semantic exploration.",
    acceptanceCriteria: ["Preserve scope and produce traceable decisions."],
    config: { maxTurns: 10 },
  });
  const runtime = new ScriptedFreshTurnRuntime(async (dispatch) => {
    const task = parseTaskPacket(dispatch.prompt) as WorkflowTurnTask;
    if (task.trigger === "INITIALIZE_TOPIC") {
      return JSON.stringify(initialTopicProposal(task));
    }
    return JSON.stringify(blockedProposal(task));
  });
  const controller = new SemanticLoopController(store, runtime, {
    projectRoot: resolve(import.meta.dirname, "../../.."),
    workDir,
    model: "scripted-model",
    skillRoot: resolve(import.meta.dirname, "../../../.codex/skills"),
    schemaManifestPath: resolve(
      import.meta.dirname,
      "../schemas/schema_manifest.json",
    ),
    capabilityManifest: {
      provider: "scripted",
      model: "scripted-model",
      wireEffortByLogicalEffort: { high: "high", max: "max" },
      highestWireEffort: "max",
    },
    budgets: {
      workflow: structuredClone(ZERO_BUDGET),
      evidence: structuredClone(EVIDENCE_BUDGET),
      direction: structuredClone(ZERO_BUDGET),
      closure: structuredClone(ZERO_BUDGET),
    },
    maxTransitionsPerRun: 12,
    noProgressThreshold: 3,
  });
  try {
    const result = await controller.run("run-1");
    assert.equal(result.lifecycle, "blocked_semantic");
    const topics = store.query(
      `SELECT object_json FROM canonical_objects
       WHERE run_id = 'run-1' AND object_type = 'topic' AND active = 1`,
    );
    assert.equal(topics.length, 1);
    assert.equal(JSON.parse(String(topics[0]!.object_json)).topicId, "topic-1");
    const attempts = store.query(
      "SELECT role, logical_effort, provider_thread_id FROM attempts ORDER BY started_at",
    );
    assert.ok(attempts.length >= 2);
    assert.ok(
      attempts.every(
        (attempt) =>
          attempt.role === "workflow_decision" &&
          attempt.logical_effort === "max",
      ),
    );
    assert.equal(
      new Set(attempts.map((attempt) => attempt.provider_thread_id)).size,
      attempts.length,
    );
  } finally {
    await runtime.close?.();
    store.close();
  }
});

function parseTaskPacket(prompt: string): unknown {
  const match = prompt.match(
    /\[TASK_PACKET\]\n([\s\S]*?)\n\n\[EXPECTED_OUTPUT_SCHEMA\]/,
  );
  assert.ok(match, "prompt has one structured TASK_PACKET section");
  return JSON.parse(match[1]!);
}

function baseProposal(task: WorkflowTurnTask) {
  return {
    protocolVersion: 1 as const,
    messageType: "WORKFLOW_DECISION_PROPOSAL" as const,
    workflowId: task.workflowId,
    runId: task.runId,
    taskId: task.taskId,
    attemptId: task.attemptId,
    stageId: task.stageId,
    stageContractHash: task.stageContractHash,
    expectedState: task.stateSnapshot,
    decisionInputHash: task.decisionInputHash,
    proposalId: `proposal-${task.taskId}`,
    reason: `Respond to ${task.trigger}.`,
    assumptions: [],
    proposedStageContract: null,
    proposedGateDefinition: null,
    proposedPlanPatch: null,
    targetStageId: null,
    domainProposal: null,
    basisArtifactRefs: [],
    basisResultRefs: [],
    requestedUserInput: null,
    blockedReport: null,
    pauseProposal: null,
    confidence: null,
  };
}

function initialTopicProposal(task: WorkflowTurnTask) {
  const base = baseProposal(task);
  const local = "topic-stage-local";
  return {
    ...base,
    decision: "RUN_STAGE" as const,
    proposedStageContract: {
      proposalLocalStageKey: local,
      stageType: "SCRIPT_APPLY_TOPIC_FRAME",
      objective: "Commit the initial TopicFrame.",
      scope: [],
      executionKind: "SCRIPT_TRANSITION",
      role: null,
      requiredInputs: [],
      expectedOutputMessageType: null,
      requestedTools: [],
      requestedPaths: [],
      prohibitedActions: [
        "state write outside Controller",
        "experiment execution",
      ],
      budget: structuredClone(ZERO_BUDGET),
    },
    proposedGateDefinition: {
      proposalLocalStageKey: local,
      mechanicalChecks: [],
      semanticEvaluation: {
        required: false,
        evaluatorRole: null,
        rubricId: null,
        inputProjection: [],
        expectedOutputMessageType: null,
      },
    },
    domainProposal: { kind: "topic_frame", value: makeTopic() },
  };
}

function blockedProposal(task: WorkflowTurnTask) {
  return {
    ...baseProposal(task),
    decision: "REPORT_BLOCKED" as const,
    blockedReport: {
      blockedReportId: `blocked-${task.taskId}`,
      kind: "semantic",
      summary: "The scripted fixture intentionally stops after Topic setup.",
      blockingFacts: ["No Anchor proposal is supplied by this test runtime."],
      exhaustedAlternatives: [],
      relatedRefs: [],
      userActionNeeded: null,
    },
  };
}
