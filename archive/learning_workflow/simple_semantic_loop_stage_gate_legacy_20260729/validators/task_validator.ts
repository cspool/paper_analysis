import type {
  AnyTurnTask,
  ClosureReviewTaskEnvelope,
  DirectionReviewTaskEnvelope,
  EvidenceReaderTaskEnvelope,
  RegisteredRole,
  TurnBudget,
  ValidationReport,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  DIMENSION_PATHS,
  validateAnchor,
  validateBudgetEquality,
  validateDirection,
  validateSearchNeed,
  validateTopicFrame,
  validateTurnBudget,
} from "./domain_validator.ts";
import {
  ALLOWED_EVIDENCE_TOOLS,
} from "../security/no_experiment_guard.ts";
import {
  ROLE_MESSAGE_TYPES,
  ROLE_SKILLS,
  TRIGGER_ALLOWED_ACTIONS,
  canonicalSha256,
} from "../contracts/index.ts";
import {
  addError,
  emptyReport,
  mergeReports,
  validateSchema,
} from "./schema_validator.ts";
import { canonicalEqual } from "../contracts/index.ts";

export interface TaskValidationContext {
  role: RegisteredRole;
  frozenBudget: TurnBudget;
  currentState: {
    snapshotVersion: number;
    canonicalRevision: number;
    eventCursor: number;
    workflowPlanRevision: number;
  };
  stageContractHash: string;
  schemaManifestSha256: string;
  skillSha256: string;
  rubricSha256?: string;
  expectedInputHash: string;
  expectedOutputSchemaSha256: string;
}

export function validateTaskForDispatch(
  task: AnyTurnTask,
  context: TaskValidationContext,
): ValidationReport {
  switch (context.role) {
    case "workflow_decision":
      return validateWorkflowTask(task as WorkflowTurnTask, context);
    case "evidence_reader":
      return validateEvidenceTask(task as EvidenceReaderTaskEnvelope, context);
    case "direction_reviewer":
      return validateDirectionTask(task as DirectionReviewTaskEnvelope, context);
    case "closure_reviewer":
      return validateClosureTask(task as ClosureReviewTaskEnvelope, context);
  }
}

function commonTaskChecks(
  task: {
    messageType: string;
    stageContractHash: string;
    stateBinding?: TaskValidationContext["currentState"];
    stateSnapshot?: TaskValidationContext["currentState"];
  },
  payload: {
    budget: TurnBudget;
    skill: { name: string; sha256: string };
    schema: {
      manifestSha256: string;
      expectedOutputMessageType?: string;
      expectedOutputSchemaSha256?: string;
    };
    permission?: { role: RegisteredRole; maxBudget: TurnBudget };
  },
  context: TaskValidationContext,
): ValidationReport {
  const report = emptyReport();
  if (task.messageType !== ROLE_MESSAGE_TYPES[context.role].input) {
    addError(
      report,
      "registry.input_message_type",
      "/messageType",
      `${context.role} requires ${ROLE_MESSAGE_TYPES[context.role].input}`,
    );
  }
  if (task.stageContractHash !== context.stageContractHash) {
    addError(
      report,
      "binding.stage_contract_hash",
      "/stageContractHash",
      "task does not bind the frozen StageContract",
    );
  }
  const binding = task.stateBinding ?? task.stateSnapshot;
  if (!canonicalEqual(binding, context.currentState)) {
    addError(
      report,
      "binding.stale_task",
      task.stateBinding ? "/stateBinding" : "/stateSnapshot",
      "task StateBinding is stale",
    );
  }
  if (
    payload.skill.name !== ROLE_SKILLS[context.role] ||
    payload.skill.sha256 !== context.skillSha256
  ) {
    addError(
      report,
      "binding.skill",
      "/payload/skill",
      "Skill name/hash does not match the registered package",
    );
  }
  if (payload.schema.manifestSha256 !== context.schemaManifestSha256) {
    addError(
      report,
      "binding.schema_manifest",
      "/payload/schema/manifestSha256",
      "schema manifest hash does not match the runtime manifest",
    );
  }
  if (
    "expectedOutputMessageType" in payload.schema &&
    (payload.schema.expectedOutputMessageType !==
      ROLE_MESSAGE_TYPES[context.role].output ||
      payload.schema.expectedOutputSchemaSha256 !==
        context.expectedOutputSchemaSha256)
  ) {
    addError(
      report,
      "binding.output_schema",
      "/payload/schema",
      "expected output message/schema does not match the role registry and manifest",
    );
  }
  if (payload.permission && payload.permission.role !== context.role) {
    addError(
      report,
      "permission.role",
      "/payload/permission/role",
      "permission role does not match dispatch role",
    );
  }
  return mergeReports(
    report,
    validateTurnBudget(context.role, payload.budget),
    validateBudgetEquality(payload.budget, context.frozenBudget),
    ...(payload.permission
      ? [validateBudgetEquality(payload.budget, payload.permission.maxBudget)]
      : []),
  );
}

function validateEvidenceTask(
  task: EvidenceReaderTaskEnvelope,
  context: TaskValidationContext,
): ValidationReport {
  const schema = validateSchema("EVIDENCE_READER_TASK", task);
  if (!schema.valid) return schema;
  const payload = task.payload;
  const report = mergeReports(
    schema,
    commonTaskChecks(task, payload, context),
    validateSearchNeed(payload.searchNeed),
    validateTopicFrame(payload.focus.topic),
    ...(payload.focus.anchor ? [validateAnchor(payload.focus.anchor)] : []),
    ...(payload.focus.direction
      ? [validateDirection(payload.focus.direction, payload.focus.topic)]
      : []),
  );
  if (payload.searchNeed.status !== "pending") {
    addError(
      report,
      "task.search_need_not_pending",
      "/payload/searchNeed/status",
      "Evidence Reader accepts only pending SearchNeed",
    );
  }
  const expectedRoots = payload.searchNeed.targetDimensions.map((dimension) => ({
    dimension,
    relativePathPrefix: DIMENSION_PATHS[dimension],
  }));
  if (!canonicalEqual(payload.allowedVaultRoots, expectedRoots)) {
    addError(
      report,
      "permission.vault_roots",
      "/payload/allowedVaultRoots",
      "vault roots must exactly match the frozen primary/auxiliary dimensions",
    );
  }
  const expectedPaths = expectedRoots.map((root) => root.relativePathPrefix);
  if (
    !canonicalEqual(
      payload.permission.tools,
      [...ALLOWED_EVIDENCE_TOOLS],
    ) ||
    !sameStringSet(payload.permission.allowedPathPrefixes, expectedPaths) ||
    payload.permission.filesystem !== "vault_read_only"
  ) {
    addError(
      report,
      "permission.evidence_profile",
      "/payload/permission",
      "Evidence task must expose exactly the two read-only Obsidian tools and frozen dimension paths",
    );
  }
  validatePayloadInputHash(task, context, report);
  return report;
}

function validateDirectionTask(
  task: DirectionReviewTaskEnvelope,
  context: TaskValidationContext,
): ValidationReport {
  const schema = validateSchema("DIRECTION_REVIEW_TASK", task);
  if (!schema.valid) return schema;
  const payload = task.payload;
  const report = mergeReports(
    schema,
    commonTaskChecks(task, payload, context),
    validateTopicFrame(payload.topic),
    validateAnchor(payload.anchor),
    validateDirection(payload.direction, payload.topic),
  );
  if (
    payload.anchor.topicId !== payload.topic.topicId ||
    payload.direction.anchorId !== payload.anchor.anchorId
  ) {
    addError(
      report,
      "task.direction_scope_binding",
      "/payload",
      "Direction, Anchor, and Topic IDs are inconsistent",
    );
  }
  const committedEvidence = new Set(
    payload.evidenceFindings.map((finding) => finding.evidenceId),
  );
  for (const evidenceId of [
    ...payload.direction.supportingEvidenceRefs,
    ...payload.direction.contradictingEvidenceRefs,
  ]) {
    if (!committedEvidence.has(evidenceId)) {
      addError(
        report,
        "task.uncommitted_evidence_ref",
        "/payload/direction",
        `Direction cites Evidence not supplied as committed: ${evidenceId}`,
      );
    }
  }
  if (
    context.rubricSha256 &&
    payload.rubric.sha256 !== context.rubricSha256
  ) {
    addError(
      report,
      "binding.rubric",
      "/payload/rubric/sha256",
      "Direction rubric hash is not registered",
    );
  }
  validateZeroToolPermission(payload.permission, report);
  validatePayloadInputHash(task, context, report);
  return report;
}

function validateClosureTask(
  task: ClosureReviewTaskEnvelope,
  context: TaskValidationContext,
): ValidationReport {
  const schema = validateSchema("CLOSURE_REVIEW_TASK", task);
  if (!schema.valid) return schema;
  const payload = task.payload;
  const report = mergeReports(
    schema,
    commonTaskChecks(task, payload, context),
    validateTopicFrame(payload.topic),
  );
  const bundle = payload.stopCandidateBundle;
  const revisions = [
    bundle.candidate.canonicalRevision,
    bundle.proof.canonicalRevision,
    payload.currentCanonicalRevision,
    context.currentState.canonicalRevision,
  ];
  if (new Set(revisions).size !== 1) {
    addError(
      report,
      "closure.revision_binding",
      "/payload/stopCandidateBundle",
      "candidate, proof, task, and canonical revision must match",
    );
  }
  if (
    bundle.candidate.stopCandidateId !== bundle.proof.stopCandidateId ||
    bundle.candidate.stopProofId !== bundle.proof.proofId
  ) {
    addError(
      report,
      "closure.candidate_proof_binding",
      "/payload/stopCandidateBundle",
      "StopCandidate and StopProof IDs do not bind",
    );
  }
  if (
    !payload.mechanicalPreflight.passed ||
    payload.mechanicalPreflight.checks.some((check) => !check.passed)
  ) {
    addError(
      report,
      "closure.preflight_required",
      "/payload/mechanicalPreflight",
      "Controller may dispatch Closure Reviewer only after a passing preflight",
    );
  }
  if (
    payload.freshTurn !== true ||
    payload.providerHistoryIncluded !== false ||
    payload.canonicalOnly !== true
  ) {
    addError(
      report,
      "closure.independence",
      "/payload",
      "Closure Reviewer requires fresh, history-free, canonical-only input",
    );
  }
  if (
    context.rubricSha256 &&
    payload.rubric.sha256 !== context.rubricSha256
  ) {
    addError(
      report,
      "binding.rubric",
      "/payload/rubric/sha256",
      "Closure rubric hash is not registered",
    );
  }
  validateZeroToolPermission(payload.permission, report);
  validatePayloadInputHash(task, context, report);
  return report;
}

function validateWorkflowTask(
  task: WorkflowTurnTask,
  context: TaskValidationContext,
): ValidationReport {
  const schema = validateSchema("WORKFLOW_TURN_TASK", task);
  if (!schema.valid) return schema;
  const pseudoPayload = {
    budget:
      task.permission.maxBudgetByRole.workflow_decision ?? context.frozenBudget,
    skill: task.skill,
    schema: task.schema,
  };
  const report = mergeReports(
    schema,
    commonTaskChecks(task, pseudoPayload, context),
  );
  const registered = new Set(TRIGGER_ALLOWED_ACTIONS[task.trigger]);
  if (
    task.permission.allowedActions.some((action) => !registered.has(action as never))
  ) {
    addError(
      report,
      "permission.trigger_action",
      "/permission/allowedActions",
      "task grants an action not registered for its trigger",
    );
  }
  if (task.triggerReport.trigger !== task.trigger) {
    addError(
      report,
      "binding.trigger_report",
      "/triggerReport/trigger",
      "trigger report does not bind the Workflow trigger",
    );
  }
  if (
    canonicalSha256(task.immutableObjective) !== task.objectiveHash ||
    canonicalSha256(task.immutableAcceptanceCriteria) !==
      task.acceptanceCriteriaHash
  ) {
    addError(
      report,
      "binding.immutable_objective_hash",
      "/objectiveHash",
      "Workflow objective and acceptance criteria must match their immutable hashes",
    );
  }
  if (
    task.relevantPlan.revision !==
    task.stateSnapshot.workflowPlanRevision
  ) {
    addError(
      report,
      "binding.plan_revision",
      "/relevantPlan/revision",
      "relevant plan must bind the current StateBinding revision",
    );
  }
  if (
    task.permission.allowedRoles.some((role) =>
      ["workflow_decision", "closure_reviewer"].includes(role),
    ) ||
    task.permission.allowedStageTypes.some((stageType) =>
      ["WORKFLOW_DECISION", "CLOSURE_REVIEW", "RENDER_FINAL"].includes(
        stageType,
      ),
    ) ||
    task.permission.allowedTools.some(
      (tool) => !ALLOWED_EVIDENCE_TOOLS.includes(tool as never),
    ) ||
    task.permission.allowedPathPrefixes.some(
      (path) => !Object.values(DIMENSION_PATHS).includes(path as never),
    )
  ) {
    addError(
      report,
      "permission.workflow_profile",
      "/permission",
      "Workflow task permission may expose only registered Worker/Evaluator capabilities",
    );
  }
  const workflowPreimage = structuredClone(task);
  workflowPreimage.decisionInputHash = "";
  const expectedHash = canonicalSha256({
    task: workflowPreimage,
    skillSha256: task.skill.sha256,
    expectedSchemaSha256: task.schema.expectedOutputSchemaSha256,
    inlineArtifactHashes: [],
  });
  if (
    task.decisionInputHash !== context.expectedInputHash ||
    task.decisionInputHash !== expectedHash
  ) {
    addError(
      report,
      "binding.decision_input_hash",
      "/decisionInputHash",
      "Workflow task input hash does not bind the complete normalized decision input",
    );
  }
  return report;
}

function validatePayloadInputHash(
  task:
    | EvidenceReaderTaskEnvelope
    | DirectionReviewTaskEnvelope
    | ClosureReviewTaskEnvelope,
  context: TaskValidationContext,
  report: ValidationReport,
): void {
  const preimage = structuredClone(task);
  preimage.inputHash = "";
  const expected = canonicalSha256(preimage);
  if (
    task.inputHash !== expected ||
    task.inputHash !== context.expectedInputHash
  ) {
    addError(
      report,
      "binding.input_hash",
      "/inputHash",
      "task input hash does not bind the complete payload task",
    );
  }
}

function validateZeroToolPermission(
  permission: DirectionReviewTaskEnvelope["payload"]["permission"],
  report: ValidationReport,
): void {
  if (
    permission.tools.length !== 0 ||
    permission.allowedPathPrefixes.length !== 0 ||
    permission.filesystem !== "none"
  ) {
    addError(
      report,
      "permission.zero_tool_profile",
      "/payload/permission",
      "zero-tool Reviewer task cannot expose tools, paths, or filesystem access",
    );
  }
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  const a = new Set(left);
  const b = new Set(right);
  return a.size === b.size && [...a].every((value) => b.has(value));
}
