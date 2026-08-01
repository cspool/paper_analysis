import type {
  RegisteredStageType,
  StageNodeDraft,
  ValidationReport,
  WorkflowDecisionProposal,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  ROLE_MESSAGE_TYPES,
  RUBRIC_REGISTRY,
  STAGE_REGISTRY,
  TRIGGER_ALLOWED_ACTIONS,
  canonicalEqual,
} from "../contracts/index.ts";
import {
  validateAnchor,
  validateBudgetEquality,
  validateDirection,
  validateSearchNeed,
  validateStageContractDraft,
  validateTopicFrame,
} from "./domain_validator.ts";
import {
  addError,
  emptyReport,
  mergeReports,
  validateSchema,
} from "./schema_validator.ts";
import { validateTurnIdentityEcho } from "./envelope_validator.ts";
import { compileGateDraft } from "../stages/gate_compiler.ts";

export function validateWorkflowDecisionProposal(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
): ValidationReport {
  const schema = validateSchema("WORKFLOW_DECISION_PROPOSAL", proposal);
  if (!schema.valid) return schema;
  const report = mergeReports(
    schema,
    validateTurnIdentityEcho(proposal, {
      role: "workflow_decision",
      task,
      stateBinding: task.stateSnapshot,
      decisionInputHash: task.decisionInputHash,
    }),
  );
  if (!canonicalEqual(proposal.expectedState, task.stateSnapshot)) {
    addError(
      report,
      "workflow.stale_expected_state",
      "/expectedState",
      "proposal expectedState must exactly echo the authoritative snapshot",
    );
  }
  if (proposal.decisionInputHash !== task.decisionInputHash) {
    addError(
      report,
      "workflow.decision_input_hash",
      "/decisionInputHash",
      "proposal does not bind the normalized Workflow input",
    );
  }
  const triggerActions = new Set(TRIGGER_ALLOWED_ACTIONS[task.trigger]);
  const taskActions = new Set(task.permission.allowedActions);
  if (
    !triggerActions.has(proposal.decision as never) ||
    !taskActions.has(proposal.decision)
  ) {
    addError(
      report,
      "workflow.action_not_allowed",
      "/decision",
      "decision is not allowed by both trigger registry and task permission envelope",
    );
  }
  validateActionPayloadMatrix(proposal, report);
  validateProposalRefs(proposal, task, report);
  validateStageAndGate(proposal, task, report);
  validatePlanPatch(proposal, task, report);
  validateDomainProposal(proposal, task, report);
  return report;
}

function validateActionPayloadMatrix(
  proposal: WorkflowDecisionProposal,
  report: ValidationReport,
): void {
  const present = {
    stage: proposal.proposedStageContract !== null,
    gate: proposal.proposedGateDefinition !== null,
    patch: proposal.proposedPlanPatch !== null,
    target: proposal.targetStageId !== null,
    domain: proposal.domainProposal !== null,
    user: proposal.requestedUserInput !== null,
    blocked: proposal.blockedReport !== null,
    pause: proposal.pauseProposal !== null,
  };
  const invalid = (message: string) =>
    addError(
      report,
      "workflow.action_payload_matrix",
      "/",
      `${proposal.decision}: ${message}`,
    );
  switch (proposal.decision) {
    case "RUN_STAGE":
      if (
        !present.stage ||
        !present.gate ||
        present.target ||
        present.patch ||
        present.blocked ||
        present.pause ||
        present.user
      ) {
        invalid("requires Stage/Gate and forbids patch/target/blocked/pause/user");
      }
      break;
    case "RETRY_STAGE":
      if (
        !present.target ||
        present.stage ||
        present.gate ||
        present.patch ||
        present.domain ||
        present.user ||
        present.blocked ||
        present.pause
      ) {
        invalid("requires only targetStageId");
      }
      break;
    case "REPLAN":
      if (
        !present.patch ||
        present.stage ||
        present.gate ||
        present.user ||
        present.blocked ||
        present.pause ||
        present.target ||
        present.domain
      ) {
        invalid("requires only a plan patch");
      }
      break;
    case "REQUEST_EVALUATION":
      if (
        !present.stage ||
        !present.gate ||
        !present.domain ||
        present.target ||
        present.patch ||
        present.user ||
        present.blocked ||
        present.pause
      ) {
        invalid("requires evaluator Stage/Gate/request and no other control payload");
      }
      break;
    case "ASK_USER":
      if (
        !present.user ||
        present.stage ||
        present.gate ||
        present.patch ||
        present.target ||
        present.domain ||
        present.blocked ||
        present.pause
      ) {
        invalid("requires only requestedUserInput");
      }
      break;
    case "REPORT_BLOCKED":
      if (
        !present.blocked ||
        present.stage ||
        present.gate ||
        present.patch ||
        present.target ||
        present.domain ||
        present.user ||
        present.pause
      ) {
        invalid("requires only blockedReport");
      }
      break;
    case "PROPOSE_PAUSE":
      if (
        !present.pause ||
        present.stage ||
        present.gate ||
        present.patch ||
        present.target ||
        present.domain ||
        present.user ||
        present.blocked
      ) {
        invalid("requires only pauseProposal");
      }
      break;
    case "PROPOSE_COMPLETE":
      if (
        !present.domain ||
        proposal.domainProposal?.kind !== "stop_candidate" ||
        present.stage ||
        present.gate ||
        present.patch ||
        present.target ||
        present.user ||
        present.blocked ||
        present.pause
      ) {
        invalid("requires only a stop_candidate DomainProposal");
      }
      break;
  }
}

function validateStageAndGate(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  const stage = proposal.proposedStageContract;
  const gate = proposal.proposedGateDefinition;
  if (!stage && !gate) return;
  if (!stage || !gate) return;
  const stageReport = validateStageContractDraft(stage);
  report.errors.push(...stageReport.errors);
  report.valid = report.errors.length === 0;
  if (stage.proposalLocalStageKey !== gate.proposalLocalStageKey) {
    addError(
      report,
      "workflow.stage_gate_key",
      "/proposedGateDefinition/proposalLocalStageKey",
      "Stage and Gate drafts must share one proposal-local key",
    );
  }
  if (!task.permission.allowedStageTypes.includes(stage.stageType)) {
    addError(
      report,
      "permission.stage_type",
      "/proposedStageContract/stageType",
      "Stage type is outside the permission envelope",
    );
  }
  if (stage.role && !task.permission.allowedRoles.includes(stage.role)) {
    addError(
      report,
      "permission.role",
      "/proposedStageContract/role",
      "role is outside the permission envelope",
    );
  }
  if (
    stage.requestedTools.some(
      (tool) => !task.permission.allowedTools.includes(tool),
    ) ||
    stage.requestedPaths.some(
      (path) =>
        !task.permission.allowedPathPrefixes.some((prefix) =>
          path.startsWith(prefix),
        ),
    )
  ) {
    addError(
      report,
      "permission.tool_or_path",
      "/proposedStageContract",
      "Stage requests a tool or path outside the permission envelope",
    );
  }
  if (stage.role) {
    const maxBudget = task.permission.maxBudgetByRole[stage.role];
    if (!maxBudget) {
      addError(
        report,
        "permission.role_budget_missing",
        "/proposedStageContract/budget",
        "permission envelope has no budget for the proposed role",
      );
    } else {
      validateBudgetCeiling(stage.budget, maxBudget, report);
    }
  }
  const authority = STAGE_REGISTRY[stage.stageType].creationAuthority;
  if (
    (proposal.decision === "RUN_STAGE" &&
      authority !== "workflow_run_stage") ||
    (proposal.decision === "REQUEST_EVALUATION" &&
      authority !== "workflow_request_evaluation")
  ) {
    addError(
      report,
      "workflow.stage_creation_authority",
      "/proposedStageContract/stageType",
      "Workflow proposal cannot self-schedule or bypass controller-only/evaluator creation paths",
    );
  }
  if (
    proposal.decision === "RUN_STAGE" &&
    ![
      "SCRIPT_APPLY_TOPIC_FRAME",
      "SCRIPT_APPLY_SEMANTIC_DELTA",
      "EVIDENCE_READ",
    ].includes(stage.stageType)
  ) {
    addError(
      report,
      "workflow.run_stage_type",
      "/proposedStageContract/stageType",
      "RUN_STAGE accepts only script apply or Evidence stages",
    );
  }
  if (
    proposal.decision === "REQUEST_EVALUATION" &&
    (stage.stageType !== "DIRECTION_REVIEW" ||
      stage.role !== "direction_reviewer")
  ) {
    addError(
      report,
      "workflow.evaluator_stage_type",
      "/proposedStageContract",
      "REQUEST_EVALUATION is limited to Direction Reviewer",
    );
  }
  const gateCompilation = compileGateDraft(stage, gate);
  report.errors.push(...gateCompilation.report.errors);
  report.valid = report.errors.length === 0;
  if (
    gate.semanticEvaluation.required &&
    gate.semanticEvaluation.evaluatorRole === stage.role
  ) {
    addError(
      report,
      "workflow.self_evaluation",
      "/proposedGateDefinition/semanticEvaluation/evaluatorRole",
      "Worker cannot be its own evaluator",
    );
  }
  validateEvidenceStageNeedBinding(proposal, task, report);
}

function validateBudgetCeiling(
  proposed: WorkflowDecisionProposal["proposedStageContract"] extends null
    ? never
    : NonNullable<WorkflowDecisionProposal["proposedStageContract"]>["budget"],
  ceiling: typeof proposed,
  report: ValidationReport,
): void {
  for (const field of [
    "timeoutMs",
    "maxInputTokens",
    "maxOutputTokens",
    "maxToolCalls",
  ] as const) {
    if (proposed[field] > ceiling[field]) {
      addError(
        report,
        "permission.budget_ceiling",
        `/proposedStageContract/budget/${field}`,
        `${field} exceeds the permission envelope`,
      );
    }
  }
  if (proposed.evidenceRead && ceiling.evidenceRead) {
    for (const field of Object.keys(proposed.evidenceRead) as Array<
      keyof typeof proposed.evidenceRead
    >) {
      if (proposed.evidenceRead[field] > ceiling.evidenceRead[field]) {
        addError(
          report,
          "permission.evidence_budget_ceiling",
          `/proposedStageContract/budget/evidenceRead/${field}`,
          `${field} exceeds the permission envelope`,
        );
      }
    }
  } else if (proposed.evidenceRead && !ceiling.evidenceRead) {
    addError(
      report,
      "permission.evidence_budget",
      "/proposedStageContract/budget/evidenceRead",
      "permission envelope does not grant Evidence budget",
    );
  }
}

function validatePlanPatch(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  const patch = proposal.proposedPlanPatch;
  if (!patch) return;
  if (
    patch.expectedPlanRevision !== task.relevantPlan.revision ||
    patch.objectiveHash !== task.objectiveHash ||
    patch.acceptanceCriteriaHash !== task.acceptanceCriteriaHash
  ) {
    addError(
      report,
      "workflow.plan_patch_binding",
      "/proposedPlanPatch",
      "plan patch must preserve immutable hashes and current plan revision",
    );
  }
  const protectedTypes = new Set<RegisteredStageType>([
    "WORKFLOW_DECISION",
    "CLOSURE_REVIEW",
    "RENDER_FINAL",
  ]);
  if (patch.operations.length === 0 || patch.operations.length > 8) {
    addError(
      report,
      "workflow.plan_patch_bounds",
      "/proposedPlanPatch/operations",
      "plan patch must contain between one and eight bounded operations",
    );
  }
  if (
    patch.operations.filter((operation) => operation.op === "add_stage")
      .length > 1
  ) {
    addError(
      report,
      "workflow.plan_patch_stage_count",
      "/proposedPlanPatch/operations",
      "one Workflow proposal may add at most one Stage",
    );
  }
  const stageIds = new Set(task.relevantPlan.stageIds);
  const dependencyIds = new Set(task.relevantPlan.dependencyIds);
  for (const [index, operation] of patch.operations.entries()) {
    if (operation.op === "add_stage") {
      if (protectedTypes.has(operation.stage.stageType)) {
        addError(
          report,
          "workflow.plan_patch_protected_stage",
          `/proposedPlanPatch/operations/${index}/stage/stageType`,
          "Workflow Agent cannot create controller-only stages",
        );
      }
      const contractMatches =
        operation.stage.stageType === operation.stage.contract.stageType &&
        operation.stage.executionKind ===
          operation.stage.contract.executionKind &&
        operation.stage.role === operation.stage.contract.role &&
        operation.stage.proposalLocalStageKey ===
          operation.stage.contract.proposalLocalStageKey &&
        operation.stage.proposalLocalStageKey ===
          operation.stage.gate.proposalLocalStageKey;
      if (!contractMatches) {
        addError(
          report,
          "workflow.plan_stage_draft_binding",
          `/proposedPlanPatch/operations/${index}/stage`,
          "StageNodeDraft, contract, and Gate must be internally bound",
        );
      }
      validatePlanAddedStage(
        operation.stage,
        task,
        report,
        `/proposedPlanPatch/operations/${index}/stage`,
      );
      if (proposal.decision === "REPLAN") {
        validateSelfContainedReplanStage(
          operation.stage,
          task,
          report,
          `/proposedPlanPatch/operations/${index}/stage`,
        );
      }
      for (const dependency of operation.stage.dependsOnStageIds) {
        if (!stageIds.has(dependency)) {
          addError(
            report,
            "workflow.plan_unknown_dependency_stage",
            `/proposedPlanPatch/operations/${index}/stage/dependsOnStageIds`,
            "new Stage dependencies must reference supplied current-plan Stage IDs",
          );
        }
      }
    } else if (operation.op === "supersede_stage") {
      if (!stageIds.has(operation.stageId)) {
        addError(
          report,
          "workflow.plan_unknown_supersede_stage",
          `/proposedPlanPatch/operations/${index}/stageId`,
          "supersede target was not supplied in the current plan",
        );
      }
    } else if (operation.op === "add_dependency") {
      if (
        !stageIds.has(operation.dependency.predecessorStageId) ||
        !stageIds.has(operation.dependency.successorStageId)
      ) {
        addError(
          report,
          "workflow.plan_unknown_dependency_endpoint",
          `/proposedPlanPatch/operations/${index}/dependency`,
          "dependency endpoints must both be supplied current-plan Stage IDs",
        );
      }
    } else if (!dependencyIds.has(operation.dependencyId)) {
      addError(
        report,
        "workflow.plan_unknown_remove_dependency",
        `/proposedPlanPatch/operations/${index}/dependencyId`,
        "removed dependency was not supplied in the current plan",
      );
    }
  }
}

function validateSelfContainedReplanStage(
  stage: StageNodeDraft,
  task: WorkflowTurnTask,
  report: ValidationReport,
  pointer: string,
): void {
  if (stage.stageType !== "EVIDENCE_READ") {
    addError(
      report,
      "workflow.replan_stage_not_self_contained",
      `${pointer}/stageType`,
      "REPLAN may add only an EVIDENCE_READ for an already supplied pending SearchNeed",
    );
    return;
  }
  const needRefs = stage.contract.scope.filter(
    (ref) => ref.objectType === "search_need",
  );
  if (needRefs.length !== 1) {
    addError(
      report,
      "workflow.replan_stage_need_scope",
      `${pointer}/contract/scope`,
      "replanned EVIDENCE_READ must scope exactly one SearchNeed",
    );
    return;
  }
  const needRef = needRefs[0]!;
  const supplied = task.permission.suppliedObjectRefs.some((ref) =>
    canonicalEqual(ref, needRef),
  );
  const pending = task.domainProjection.searchNeeds.some(
    (need) =>
      need.needId === needRef.objectId &&
      need.revision === needRef.revision &&
      need.status === "pending",
  );
  if (!supplied || !pending) {
    addError(
      report,
      "workflow.replan_stage_need_not_pending",
      `${pointer}/contract/scope`,
      "replanned EVIDENCE_READ SearchNeed must be current, pending, and supplied",
    );
  }
}

function validatePlanAddedStage(
  stage: StageNodeDraft,
  task: WorkflowTurnTask,
  report: ValidationReport,
  pointer: string,
): void {
  const nested = validateStageContractDraft(stage.contract);
  append(report, nested);
  if (!task.permission.allowedStageTypes.includes(stage.stageType)) {
    addError(
      report,
      "permission.stage_type",
      `${pointer}/stageType`,
      "replanned Stage type is outside the permission envelope",
    );
  }
  if (
    stage.role !== null &&
    !task.permission.allowedRoles.includes(stage.role)
  ) {
    addError(
      report,
      "permission.role",
      `${pointer}/role`,
      "replanned Stage role is outside the permission envelope",
    );
  }
  if (
    stage.contract.requestedTools.some(
      (tool) => !task.permission.allowedTools.includes(tool),
    ) ||
    stage.contract.requestedPaths.some(
      (path) =>
        !task.permission.allowedPathPrefixes.some((prefix) =>
          path.startsWith(prefix),
        ),
    )
  ) {
    addError(
      report,
      "permission.tool_or_path",
      `${pointer}/contract`,
      "replanned Stage requests a tool or path outside the permission envelope",
    );
  }
  if (stage.role) {
    const ceiling = task.permission.maxBudgetByRole[stage.role];
    if (!ceiling) {
      addError(
        report,
        "permission.role_budget_missing",
        `${pointer}/contract/budget`,
        "permission envelope has no budget for the replanned role",
      );
    } else {
      validateBudgetCeiling(stage.contract.budget, ceiling, report);
    }
  }
  stage.contract.requiredInputs.forEach((artifact, index) => {
    if (
      !task.permission.suppliedArtifactIds.includes(
        artifact.artifactId,
      ) ||
      !task.approvedArtifacts.some((approved) =>
        canonicalEqual(approved, artifact),
      )
    ) {
      addError(
        report,
        "workflow.required_input_not_exactly_supplied",
        `${pointer}/contract/requiredInputs/${index}`,
        "replanned Stage input must exactly match an approved ArtifactRef",
      );
    }
  });
  stage.contract.scope.forEach((ref, index) => {
    if (
      !task.permission.suppliedObjectRefs.some((supplied) =>
        canonicalEqual(supplied, ref),
      )
    ) {
      addError(
        report,
        "workflow.replan_scope_ref_not_supplied",
        `${pointer}/contract/scope/${index}`,
        "replanned Stage scope must exactly match a supplied current ObjectRef",
      );
    }
  });
  const gateCompilation = compileGateDraft(
    stage.contract,
    stage.gate,
  );
  gateCompilation.report.errors.forEach((error) => {
    report.errors.push({
      ...error,
      jsonPointer: error.jsonPointer
        ? `${pointer}${error.jsonPointer}`
        : pointer,
    });
  });
  report.valid = report.errors.length === 0;
  if (
    stage.gate.semanticEvaluation.required &&
    stage.gate.semanticEvaluation.evaluatorRole === stage.role
  ) {
    addError(
      report,
      "workflow.self_evaluation",
      `${pointer}/gate/semanticEvaluation/evaluatorRole`,
      "Worker cannot be its own evaluator",
    );
  }
}

function validateProposalRefs(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  const artifactIds = new Set(task.permission.suppliedArtifactIds);
  proposal.basisArtifactRefs.forEach((artifact, index) => {
    if (
      !artifactIds.has(artifact.artifactId) ||
      !task.approvedArtifacts.some(
        (approved) =>
          approved.artifactId === artifact.artifactId &&
          canonicalEqual(approved, artifact),
      )
    ) {
      addError(
        report,
        "workflow.fabricated_artifact_ref",
        `/basisArtifactRefs/${index}`,
        "basis artifact must exactly match an approved supplied artifact",
      );
    }
  });
  const suppliedResults = new Set(
    task.permission.suppliedResultRefs.map(refKey),
  );
  const unconsumedResults = new Set(
    task.resultIndex.committedUnconsumedResultRefs.map(refKey),
  );
  const consumedResults = new Set(
    task.resultIndex.consumedResultRefs.map(refKey),
  );
  proposal.basisResultRefs.forEach((ref, index) => {
    if (
      !suppliedResults.has(refKey(ref)) ||
      (!unconsumedResults.has(refKey(ref)) &&
        !consumedResults.has(refKey(ref)))
    ) {
      addError(
        report,
        "workflow.fabricated_result_ref",
        `/basisResultRefs/${index}`,
        "basis result ref must be a supplied committed result",
      );
    }
  });
  proposal.proposedStageContract?.requiredInputs.forEach((artifact, index) => {
    if (
      !artifactIds.has(artifact.artifactId) ||
      !task.approvedArtifacts.some((approved) =>
        canonicalEqual(approved, artifact),
      )
    ) {
      addError(
        report,
        "workflow.required_input_not_exactly_supplied",
        `/proposedStageContract/requiredInputs/${index}`,
        "Stage required input must exactly match an approved supplied ArtifactRef",
      );
    }
  });
  const suppliedObjects = new Set(
    task.permission.suppliedObjectRefs.map(refKey),
  );
  const proposedObject = domainProposalObjectRef(proposal);
  proposal.proposedStageContract?.scope.forEach((ref, index) => {
    if (
      !suppliedObjects.has(refKey(ref)) &&
      (!proposedObject || !canonicalEqual(ref, proposedObject))
    ) {
      addError(
        report,
        "workflow.fabricated_stage_scope_ref",
        `/proposedStageContract/scope/${index}`,
        "Stage scope must reference a supplied current object or the proposal's new domain object",
      );
    }
  });
}

function validateEvidenceStageNeedBinding(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  const stage = proposal.proposedStageContract;
  if (stage?.stageType !== "EVIDENCE_READ") return;
  if (proposal.domainProposal?.kind === "search_need") {
    const proposedNeed = proposal.domainProposal.value;
    const needRefs = stage.scope.filter(
      (ref) => ref.objectType === "search_need",
    );
    if (
      needRefs.length !== 1 ||
      needRefs[0]!.objectId !== proposedNeed.needId ||
      needRefs[0]!.revision !== proposedNeed.revision
    ) {
      addError(
        report,
        "workflow.evidence_stage_need_scope",
        "/proposedStageContract/scope",
        "EVIDENCE_READ creating a SearchNeed must scope exactly that proposed Need revision",
      );
    }
    return;
  }
  if (proposal.domainProposal !== null) {
    domainError(
      report,
      "EVIDENCE_READ may only carry a SearchNeed DomainProposal",
    );
    return;
  }
  const needRefs = stage.scope.filter(
    (ref) => ref.objectType === "search_need",
  );
  if (needRefs.length !== 1) {
    addError(
      report,
      "workflow.evidence_stage_need_scope",
      "/proposedStageContract/scope",
      "EVIDENCE_READ without a new SearchNeed must scope exactly one supplied pending SearchNeed",
    );
    return;
  }
  const ref = needRefs[0]!;
  const current = task.domainProjection.searchNeeds.find(
    (need) =>
      need.needId === ref.objectId &&
      need.revision === ref.revision &&
      need.status === "pending",
  );
  if (
    !current ||
    !task.permission.suppliedObjectRefs.some((item) =>
      canonicalEqual(item, ref),
    )
  ) {
    addError(
      report,
      "workflow.evidence_stage_need_scope",
      "/proposedStageContract/scope",
      "scoped SearchNeed must be current, pending, and supplied by the Controller",
    );
  }
}

function domainProposalObjectRef(
  proposal: WorkflowDecisionProposal,
): { objectType: string; objectId: string; revision: number } | null {
  const domain = proposal.domainProposal;
  if (!domain) return null;
  if (domain.kind === "topic_frame") {
    return {
      objectType: "topic",
      objectId: domain.value.topicId,
      revision: domain.value.revision,
    };
  }
  if (domain.kind === "search_need") {
    return {
      objectType: "search_need",
      objectId: domain.value.needId,
      revision: domain.value.revision,
    };
  }
  if (
    domain.kind === "semantic_delta" &&
    domain.value.proposedObject
  ) {
    const identity = semanticObjectIdentity(domain.value.proposedObject);
    return identity
      ? {
          objectType: identity.type,
          objectId: identity.id,
          revision: identity.revision,
        }
      : null;
  }
  if (domain.kind === "direction_review_request") {
    return domain.value.directionRef;
  }
  return null;
}

function validateDomainProposal(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  const domain = proposal.domainProposal;
  if (!domain) return;
  switch (domain.kind) {
    case "topic_frame": {
      const nested = validateTopicFrame(domain.value);
      append(report, nested);
      if (
        proposal.decision !== "RUN_STAGE" ||
        proposal.proposedStageContract?.stageType !==
          "SCRIPT_APPLY_TOPIC_FRAME"
      ) {
        domainError(report, "TopicFrame must use SCRIPT_APPLY_TOPIC_FRAME");
      }
      break;
    }
    case "search_need": {
      append(report, validateSearchNeed(domain.value));
      if (
        proposal.decision !== "RUN_STAGE" ||
        proposal.proposedStageContract?.stageType !== "EVIDENCE_READ"
      ) {
        domainError(report, "SearchNeed must use EVIDENCE_READ");
      }
      break;
    }
    case "semantic_delta": {
      if (
        proposal.decision !== "RUN_STAGE" ||
        proposal.proposedStageContract?.stageType !==
          "SCRIPT_APPLY_SEMANTIC_DELTA"
      ) {
        domainError(
          report,
          "SemanticDelta must use SCRIPT_APPLY_SEMANTIC_DELTA",
        );
      }
      const object = domain.value.proposedObject;
      if (object && "topicId" in object && "userTopic" in object) {
        append(report, validateTopicFrame(object));
      } else if (object && "anchorId" in object && "scenario" in object) {
        append(report, validateAnchor(object));
      } else if (object && "directionId" in object && "changes" in object) {
        if (task.domainProjection.topic === null) {
          domainError(
            report,
            "Direction cannot be proposed before a canonical TopicFrame",
          );
        } else {
          append(
            report,
            validateDirection(object, task.domainProjection.topic),
          );
        }
      }
      validateSemanticDeltaBinding(proposal, task, report);
      break;
    }
    case "direction_review_request": {
      if (
        proposal.decision !== "REQUEST_EVALUATION" ||
        proposal.proposedStageContract?.stageType !== "DIRECTION_REVIEW" ||
        !task.permission.registeredRubrics.includes(
          domain.value.rubric.rubricId,
        ) ||
        !canonicalEqual(
          domain.value.rubric,
          RUBRIC_REGISTRY.direction_readiness_v1,
        ) ||
        !task.permission.suppliedObjectRefs.some((ref) =>
          canonicalEqual(ref, domain.value.directionRef),
        )
      ) {
        domainError(
          report,
          "Direction review request requires the registered Direction evaluator/rubric",
        );
      }
      break;
    }
    case "stop_candidate": {
      if (proposal.decision !== "PROPOSE_COMPLETE") {
        domainError(report, "StopCandidate is legal only for PROPOSE_COMPLETE");
      }
      const candidate = domain.value.candidate;
      const proof = domain.value.proof;
      const completion = task.domainProjection.completionProjection;
      if (
        candidate.runId !== task.runId ||
        candidate.topicId !== task.domainProjection.topic?.topicId ||
        candidate.stopCandidateId !== proof.stopCandidateId ||
        candidate.stopProofId !== proof.proofId ||
        candidate.canonicalRevision !== task.stateSnapshot.canonicalRevision ||
        proof.canonicalRevision !== task.stateSnapshot.canonicalRevision
      ) {
        domainError(
          report,
          "StopCandidate/Proof must bind this run and current canonical revision",
        );
      }
      if (
        !completion ||
        !completion.eligibleForProposal ||
        completion.blockingClaims.length > 0 ||
        !canonicalEqual(
          stopProofProjection(proof),
          completionProjectionFacts(completion),
        )
      ) {
        domainError(
          report,
          "StopProof must exactly copy the Controller-supplied eligible completion projection",
        );
      }
      if (Object.values(proof.claims).some((claim) => claim !== true)) {
        domainError(
          report,
          "PROPOSE_COMPLETE requires all ten StopProof claims true",
        );
      }
      break;
    }
  }
}

function stopProofProjection(
  proof: Extract<
    NonNullable<WorkflowDecisionProposal["domainProposal"]>,
    { kind: "stop_candidate" }
  >["value"]["proof"],
) {
  const {
    proofId: _proofId,
    stopCandidateId: _stopCandidateId,
    ...projection
  } = proof;
  return projection;
}

function completionProjectionFacts(
  projection: NonNullable<
    WorkflowTurnTask["domainProjection"]["completionProjection"]
  >,
) {
  const {
    eligibleForProposal: _eligible,
    blockingClaims: _blocking,
    ...facts
  } = projection;
  return facts;
}

function validateSemanticDeltaBinding(
  proposal: WorkflowDecisionProposal,
  task: WorkflowTurnTask,
  report: ValidationReport,
): void {
  if (proposal.domainProposal?.kind !== "semantic_delta") return;
  const delta = proposal.domainProposal.value;
  if (
    delta.basisResultRefs.length === 0 ||
    !sameRefSet(delta.basisResultRefs, proposal.basisResultRefs)
  ) {
    domainError(
      report,
      "SemanticDelta basisResultRefs must exactly equal the proposal's non-empty committed-unconsumed basis",
    );
  }
  const unconsumed = new Set(
    task.resultIndex.committedUnconsumedResultRefs.map(refKey),
  );
  if (
    delta.basisResultRefs.some(
      (ref) => !unconsumed.has(refKey(ref)),
    )
  ) {
    domainError(
      report,
      "SemanticDelta may consume only Controller-supplied committed-unconsumed results",
    );
  }
  const target = task.permission.suppliedObjectRefs.find(
    (ref) =>
      ref.objectType === delta.target.type &&
      ref.objectId === delta.target.id,
  );
  if (delta.action === "create") {
    if (target || delta.expectedTargetRevision !== 0) {
      domainError(
        report,
        "create requires an absent target and expectedTargetRevision=0",
      );
    }
  } else if (!target || target.revision !== delta.expectedTargetRevision) {
    domainError(
      report,
      "existing-object SemanticDelta must bind the supplied active target revision",
    );
  }
  if (delta.action === "no_semantic_delta") {
    if (delta.proposedObject !== null || delta.changedFields.length !== 0) {
      domainError(
        report,
        "no_semantic_delta requires proposedObject=null and no changed fields",
      );
    }
    return;
  }
  if (!delta.proposedObject) {
    domainError(
      report,
      "a mutating SemanticDelta requires one complete proposed object",
    );
    return;
  }
  const objectIdentity = semanticObjectIdentity(delta.proposedObject);
  if (
    !objectIdentity ||
    objectIdentity.type !== delta.target.type ||
    objectIdentity.id !== delta.target.id ||
    objectIdentity.revision !==
      (delta.action === "create"
        ? 1
        : delta.expectedTargetRevision + 1)
  ) {
    domainError(
      report,
      "proposed object identity/revision must match the SemanticDelta target and CAS revision",
    );
  }
}

function semanticObjectIdentity(
  value: NonNullable<
    Extract<
      NonNullable<WorkflowDecisionProposal["domainProposal"]>,
      { kind: "semantic_delta" }
    >["value"]["proposedObject"]
  >,
): { type: "topic" | "anchor" | "direction"; id: string; revision: number } | null {
  if ("topicId" in value && "userTopic" in value) {
    return { type: "topic", id: value.topicId, revision: value.revision };
  }
  if ("anchorId" in value && "scenario" in value) {
    return { type: "anchor", id: value.anchorId, revision: value.revision };
  }
  if ("directionId" in value) {
    return {
      type: "direction",
      id: value.directionId,
      revision: value.revision,
    };
  }
  return null;
}

function sameRefSet(
  left: Array<{ objectType: string; objectId: string; revision: number }>,
  right: Array<{ objectType: string; objectId: string; revision: number }>,
): boolean {
  const a = new Set(left.map(refKey));
  const b = new Set(right.map(refKey));
  return a.size === b.size && [...a].every((value) => b.has(value));
}

function refKey(ref: {
  objectType: string;
  objectId: string;
  revision: number;
}): string {
  return `${ref.objectType}\0${ref.objectId}\0${ref.revision}`;
}

function append(target: ValidationReport, source: ValidationReport): void {
  target.errors.push(...source.errors);
  target.valid = target.errors.length === 0;
}

function domainError(report: ValidationReport, message: string): void {
  addError(
    report,
    "workflow.domain_proposal_binding",
    "/domainProposal",
    message,
  );
}
