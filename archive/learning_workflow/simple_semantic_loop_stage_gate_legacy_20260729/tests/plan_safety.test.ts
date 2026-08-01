import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MAX_WORKFLOW_PLAN_REVISIONS,
  canonicalSha256,
  type StageContractDraft,
  type WorkflowPlan,
  type WorkflowPlanPatch,
} from "../contracts/index.ts";
import {
  applyPlanPatch,
  freezeStageDraft,
  validateWorkflowPlan,
} from "../workflow/plan_store.ts";
import {
  compileGateDraft,
  GATE_COMPILER_POLICY_VERSION,
} from "../stages/gate_compiler.ts";
import {
  evaluateGate,
  GATE_EVALUATOR_VERSION,
} from "../stages/gate_engine.ts";
import { ZERO_BUDGET } from "./fixtures/factory.ts";

test("dynamic plan patches reject dependency cycles", () => {
  const initial = plan();
  const first = applyPlanPatch(
    initial,
    addStagePatch(initial, stageDraft("stage-a")),
    1,
  ).plan;
  const stageA = first.stageNodes[0]!.stageId;
  const secondDraft = stageDraft("stage-b");
  const secondPatch = addStagePatch(first, secondDraft);
  if (secondPatch.operations[0]!.op !== "add_stage") {
    throw new Error("fixture lost add_stage operation");
  }
  secondPatch.operations[0]!.stage.dependsOnStageIds = [stageA];
  const second = applyPlanPatch(first, secondPatch, 2).plan;
  const stageB = second.stageNodes.find(
    (node) => node.stageId !== stageA,
  )!.stageId;
  const cyclic: WorkflowPlanPatch = {
    expectedPlanRevision: second.revision,
    objectiveHash: second.objectiveHash,
    acceptanceCriteriaHash: second.acceptanceCriteriaHash,
    rationale: "Attempt a reverse edge.",
    operations: [
      {
        op: "add_dependency",
        dependency: {
          dependencyId: "dependency-cycle",
          predecessorStageId: stageB,
          successorStageId: stageA,
          kind: "requires_committed",
        },
      },
    ],
  };
  assert.throws(
    () => applyPlanPatch(second, cyclic, 3),
    /contains a cycle/,
  );
});

test("executed Stages cannot be superseded and unknown Stage bindings fail closed", () => {
  const initial = plan();
  const applied = applyPlanPatch(
    initial,
    addStagePatch(initial, stageDraft("stage-a")),
    1,
  ).plan;
  applied.stageNodes[0]!.lifecycle = "committed";
  const supersede: WorkflowPlanPatch = {
    expectedPlanRevision: applied.revision,
    objectiveHash: applied.objectiveHash,
    acceptanceCriteriaHash: applied.acceptanceCriteriaHash,
    rationale: "Illegal post-execution supersede.",
    operations: [
      {
        op: "supersede_stage",
        stageId: applied.stageNodes[0]!.stageId,
        reason: "too late",
      },
    ],
  };
  assert.throws(
    () => applyPlanPatch(applied, supersede, 2),
    /cannot supersede executed Stage/,
  );

  const unknown = structuredClone(applied);
  (unknown.stageNodes[0] as { stageType: string }).stageType =
    "EXPERIMENT_RUN";
  assert.ok(
    validateWorkflowPlan(unknown).some((error) =>
      error.includes("Stage registry mismatch"),
    ),
  );
});

test("plan revision growth has a fixed non-agent-overridable ceiling", () => {
  const current = plan();
  current.revision = MAX_WORKFLOW_PLAN_REVISIONS;
  const patch = addStagePatch(current, stageDraft("too-many"));
  assert.throws(
    () => applyPlanPatch(current, patch, 10),
    /maximum WorkflowPlan revision/,
  );
});

test("Stage and Gate are frozen together before execution with immutable hashes", () => {
  const current = plan();
  const applied = applyPlanPatch(
    current,
    addStagePatch(current, stageDraft("freeze-pair")),
    17,
  );
  const frozen = applied.frozenStages[0]!;
  assert.equal(frozen.contract.definedAtSnapshotVersion, 17);
  assert.equal(frozen.gate.definedAtSnapshotVersion, 17);
  assert.equal(frozen.contract.stageId, frozen.gate.stageId);
  const { sha256: _contractHash, ...contractPreimage } = frozen.contract;
  const { sha256: _gateHash, ...gatePreimage } = frozen.gate;
  assert.equal(frozen.contract.sha256, canonicalSha256(contractPreimage));
  assert.equal(frozen.gate.sha256, canonicalSha256(gatePreimage));
  assert.equal(
    frozen.gate.stageContractHash,
    frozen.contract.sha256,
  );
  assert.equal(
    frozen.gate.compilerPolicyVersion,
    GATE_COMPILER_POLICY_VERSION,
  );
  assert.equal(
    frozen.gate.evaluatorVersion,
    GATE_EVALUATOR_VERSION,
  );
  assert.ok(
    frozen.gate.mechanicalChecks.some(
      (check) =>
        check.checkId === "controller.script_transition_valid",
    ),
  );
});

test("Agent Gate criteria cannot contradict a Controller mandatory check", () => {
  const stage = stageDraft("contradict-controller");
  const compiled = compileGateDraft(stage, {
    proposalLocalStageKey: stage.proposalLocalStageKey,
    mechanicalChecks: [
      {
        checkId: "agent-negates-script-validator",
        predicate: "equals",
        actual: {
          source: "validator",
          fact: "script_transition_valid",
          pointer: null,
          valueType: "boolean",
        },
        expected: false,
      },
    ],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  });
  assert.equal(compiled.report.valid, false);
  assert.ok(
    compiled.report.errors.some(
      (error) =>
        error.code === "gate.contradicts_controller_mandatory_check",
    ),
  );
});

test("Gate compiler admits the domain payload root and enforces validator pointer modes", () => {
  const stage: StageContractDraft = {
    proposalLocalStageKey: "evidence-payload-root",
    stageType: "EVIDENCE_READ",
    objective: "Read one bounded evidence packet.",
    scope: [],
    executionKind: "WORKER_TURN",
    role: "evidence_reader",
    requiredInputs: [],
    expectedOutputMessageType: "EVIDENCE_PACKET",
    requestedTools: [],
    requestedPaths: [],
    prohibitedActions: ["experiment execution"],
    budget: structuredClone(ZERO_BUDGET),
  };
  const valid = compileGateDraft(stage, {
    proposalLocalStageKey: stage.proposalLocalStageKey,
    mechanicalChecks: [
      {
        checkId: "payload-core-fields",
        predicate: "contains_fields",
        actual: {
          source: "result",
          pointer: "/payload",
          valueType: "object",
        },
        expected: ["packetId", "needId", "findings"],
      },
      {
        checkId: "domain-validator",
        predicate: "equals",
        actual: {
          source: "validator",
          fact: "registered_validator_passes",
          pointer: null,
          valueType: "boolean",
        },
        expected: true,
      },
      {
        checkId: "finding-references",
        predicate: "equals",
        actual: {
          source: "validator",
          fact: "references_resolve",
          pointer: "/payload/findings",
          valueType: "boolean",
        },
        expected: true,
      },
    ],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  });
  assert.equal(valid.report.valid, true);

  const invalid = compileGateDraft(stage, {
    proposalLocalStageKey: stage.proposalLocalStageKey,
    mechanicalChecks: [
      {
        checkId: "invalid-domain-validator-pointer",
        predicate: "equals",
        actual: {
          source: "validator",
          fact: "registered_validator_passes",
          pointer: "/payload",
          valueType: "boolean",
        },
        expected: true,
      },
    ],
    semanticEvaluation: {
      required: false,
      evaluatorRole: null,
      rubricId: null,
      inputProjection: [],
      expectedOutputMessageType: null,
    },
  });
  const pointerError = invalid.report.errors.find(
    (error) => error.code === "gate.validator_pointer_contract",
  );
  assert.ok(pointerError);
  assert.match(pointerError.message, /requires pointer:null/);
});

test("Gate evaluation converts resolver faults and version drift into failed checks", () => {
  const stage = stageDraft("total-gate");
  const frozen = freezeStageDraft(
    stage,
    {
      proposalLocalStageKey: stage.proposalLocalStageKey,
      mechanicalChecks: [],
      semanticEvaluation: {
        required: false,
        evaluatorRole: null,
        rubricId: null,
        inputProjection: [],
        expectedOutputMessageType: null,
      },
    },
    1,
  );
  const resolverFailure = evaluateGate(frozen.gate, {
    resolve() {
      throw new Error("untrusted resolver value");
    },
  });
  assert.equal(resolverFailure.passed, false);
  assert.equal(
    resolverFailure.checks[0]!.errorCode,
    "gate.resolver_exception",
  );

  const drifted = structuredClone(frozen.gate);
  drifted.evaluatorVersion = "future-evaluator/99";
  const versionFailure = evaluateGate(drifted, {
    resolve() {
      return { resolved: true, value: true, detail: "unused" };
    },
  });
  assert.equal(versionFailure.passed, false);
  assert.equal(
    versionFailure.checks[0]!.errorCode,
    "gate.evaluator_version_unsupported",
  );
});

function plan(): WorkflowPlan {
  return {
    workflowId: "workflow-1",
    revision: 1,
    objectiveHash: "a".repeat(64),
    acceptanceCriteriaHash: "b".repeat(64),
    stageNodes: [],
    dependencies: [],
    planStatus: "active",
  };
}

function stageDraft(key: string): StageContractDraft {
  return {
    proposalLocalStageKey: key,
    stageType: "SCRIPT_APPLY_TOPIC_FRAME",
    objective: "Apply one validated TopicFrame.",
    scope: [],
    executionKind: "SCRIPT_TRANSITION",
    role: null,
    requiredInputs: [],
    expectedOutputMessageType: null,
    requestedTools: [],
    requestedPaths: [],
    prohibitedActions: ["experiment execution"],
    budget: structuredClone(ZERO_BUDGET),
  };
}

function addStagePatch(
  current: WorkflowPlan,
  contract: StageContractDraft,
): WorkflowPlanPatch {
  return {
    expectedPlanRevision: current.revision,
    objectiveHash: current.objectiveHash,
    acceptanceCriteriaHash: current.acceptanceCriteriaHash,
    rationale: "Add one registered bounded Stage.",
    operations: [
      {
        op: "add_stage",
        stage: {
          proposalLocalStageKey: contract.proposalLocalStageKey,
          stageType: contract.stageType,
          executionKind: contract.executionKind,
          role: contract.role,
          objective: contract.objective,
          dependsOnStageIds: [],
          contract,
          gate: {
            proposalLocalStageKey: contract.proposalLocalStageKey,
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
}
