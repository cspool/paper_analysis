import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ROLE_MESSAGE_TYPES,
  ROLE_REASONING_EFFORT,
  STAGE_REGISTRY,
  LAYER_DEFINITIONS,
  canonicalJson,
  canonicalSha256,
  normalizeContainedRelativePath,
} from "../contracts/index.ts";
import {
  validateClosureReview,
  validateEvidencePacket,
  validateDirection,
  validateReviewDelta,
  validateSchema,
  validateWorkflowDecisionProposal,
} from "../validators/index.ts";
import {
  makeClosureResult,
  makeClosureTask,
  makeDirectionTask,
  makeDirection,
  makeEvidenceResult,
  makeEvidenceTask,
  makeReviewResult,
  makeWorkflowProposal,
  makeWorkflowTask,
  makeTopic,
} from "./fixtures/factory.ts";

test("canonical JSON and SHA-256 are independent of object key order", () => {
  const left = { z: [3, { b: 2, a: 1 }], a: true };
  const right = { a: true, z: [3, { a: 1, b: 2 }] };
  assert.equal(canonicalJson(left), canonicalJson(right));
  assert.equal(canonicalSha256(left), canonicalSha256(right));
});

test("contained relative paths reject traversal and absolute targets", () => {
  assert.equal(
    normalizeContainedRelativePath("/tmp/approved", "nested/file.json"),
    "nested/file.json",
  );
  assert.throws(() =>
    normalizeContainedRelativePath("/tmp/approved", "../escape.json"),
  );
  assert.throws(() =>
    normalizeContainedRelativePath("/tmp/approved", "/etc/passwd"),
  );
});

test("role/message/effort registry has exactly four immutable Turn roles", () => {
  assert.deepEqual(Object.keys(ROLE_MESSAGE_TYPES).sort(), [
    "closure_reviewer",
    "direction_reviewer",
    "evidence_reader",
    "workflow_decision",
  ]);
  assert.deepEqual(ROLE_REASONING_EFFORT, {
    workflow_decision: "max",
    evidence_reader: "high",
    direction_reviewer: "high",
    closure_reviewer: "high",
  });
  assert.equal(Object.isFrozen(ROLE_REASONING_EFFORT), true);
});

test("Stage registry binds type, execution kind, role, output and authority", () => {
  assert.equal(Object.keys(STAGE_REGISTRY).length, 7);
  assert.deepEqual(STAGE_REGISTRY.WORKFLOW_DECISION, {
    executionKind: "DECISION_TURN",
    role: "workflow_decision",
    output: "WORKFLOW_DECISION_PROPOSAL",
    creationAuthority: "controller_trigger",
  });
  assert.deepEqual(STAGE_REGISTRY.CLOSURE_REVIEW, {
    executionKind: "EVALUATOR_TURN",
    role: "closure_reviewer",
    output: "CLOSURE_REVIEW",
    creationAuthority: "controller_closure",
  });
});

test("L1-L6 are fixed modification coordinates and Direction layers stay inside Topic scope", () => {
  assert.deepEqual(Object.keys(LAYER_DEFINITIONS), [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
  ]);
  const topic = makeTopic();
  const direction = makeDirection();
  direction.changes[0]!.layer = "L6";
  const report = validateDirection(direction, topic);
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some(
      (error) => error.code === "domain.direction_layer_outside_topic",
    ),
  );
});

test("all four canonical task/result fixtures satisfy strict schemas", () => {
  const fixtures = [
    ["EVIDENCE_READER_TASK", makeEvidenceTask()],
    ["EVIDENCE_PACKET", makeEvidenceResult()],
    ["DIRECTION_REVIEW_TASK", makeDirectionTask()],
    ["REVIEW_DELTA", makeReviewResult()],
    ["CLOSURE_REVIEW_TASK", makeClosureTask()],
    ["CLOSURE_REVIEW", makeClosureResult()],
    ["WORKFLOW_TURN_TASK", makeWorkflowTask()],
    ["WORKFLOW_DECISION_PROPOSAL", makeWorkflowProposal()],
  ] as const;
  for (const [name, fixture] of fixtures) {
    const report = validateSchema(name, fixture);
    assert.equal(
      report.valid,
      true,
      `${name}: ${JSON.stringify(report.errors, null, 2)}`,
    );
  }
});

test("schemas fail closed on old message names and unknown fields", () => {
  const old = structuredClone(makeEvidenceTask()) as Record<string, unknown>;
  old.messageType = "EVIDENCE_TASK";
  assert.equal(validateSchema("EVIDENCE_READER_TASK", old).valid, false);

  const extra = structuredClone(makeClosureResult()) as Record<string, unknown>;
  extra.session = { allAnchorsSaturated: true };
  const report = validateSchema("CLOSURE_REVIEW", extra);
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some((error) => error.code === "schema.additionalProperties"),
  );
});

test("valid EvidencePacket passes provenance, budget and conclusion gates", () => {
  const report = validateEvidencePacket(makeEvidenceResult(), makeEvidenceTask());
  assert.equal(report.valid, true, JSON.stringify(report.errors, null, 2));
});

test("Evidence not_found cannot hide a finding", () => {
  const result = makeEvidenceResult();
  result.payload.conclusion = "not_found";
  result.payload.unanswered = [
    {
      unansweredId: "u-1",
      successCriterion:
        "A source states the mechanism under the decode scenario.",
      reason: "no_matching_source",
      attemptedSearchIds: ["search-1"],
    },
  ];
  const report = validateEvidencePacket(result, makeEvidenceTask());
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some((error) => error.code === "evidence.not_found_matrix"),
  );
});

test("valid Direction testable result passes eleven-check matrix", () => {
  const report = validateReviewDelta(
    makeReviewResult(),
    makeDirectionTask(),
  );
  assert.equal(report.valid, true, JSON.stringify(report.errors, null, 2));
});

test("Direction cannot claim testable with a knowledge-answerable gap", () => {
  const result = makeReviewResult();
  result.payload.readinessChecks.knowledgeAnswerableCriticalGapRemaining = true;
  const report = validateReviewDelta(result, makeDirectionTask());
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some((error) => error.code === "review.testable_matrix"),
  );
});

test("valid Closure accept requires all thirteen checks and five finalizers", () => {
  const report = validateClosureReview(
    makeClosureResult(),
    makeClosureTask(),
  );
  assert.equal(report.valid, true, JSON.stringify(report.errors, null, 2));
});

test("Closure accept cannot mask an open critical Need", () => {
  const task = makeClosureTask();
  task.payload.needs.push({
    needId: "need-open",
    revision: 1,
    owner: { topicId: "topic-1", anchorId: null, directionId: null },
    critical: true,
    answerability: "knowledge_base",
    status: "pending",
  });
  task.payload.stopCandidateBundle.proof.openNeedIds.push("need-open");
  const report = validateClosureReview(makeClosureResult(), task);
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some(
      (error) => error.code === "closure.check_fact_mismatch",
    ),
  );
});

test("valid Workflow RUN_STAGE proposal passes action and authority gates", () => {
  const report = validateWorkflowDecisionProposal(
    makeWorkflowProposal(),
    makeWorkflowTask(),
  );
  assert.equal(report.valid, true, JSON.stringify(report.errors, null, 2));
});

test("Workflow Agent cannot self-schedule WORKFLOW_DECISION", () => {
  const task = makeWorkflowTask();
  task.permission.allowedStageTypes.push("WORKFLOW_DECISION");
  task.permission.allowedRoles.push("workflow_decision");
  const proposal = makeWorkflowProposal();
  proposal.proposedStageContract!.stageType = "WORKFLOW_DECISION";
  proposal.proposedStageContract!.executionKind = "DECISION_TURN";
  proposal.proposedStageContract!.role = "workflow_decision";
  proposal.proposedStageContract!.expectedOutputMessageType =
    "WORKFLOW_DECISION_PROPOSAL";
  const report = validateWorkflowDecisionProposal(proposal, task);
  assert.equal(report.valid, false);
  assert.ok(
    report.errors.some(
      (error) => error.code === "workflow.stage_creation_authority",
    ),
  );
});
