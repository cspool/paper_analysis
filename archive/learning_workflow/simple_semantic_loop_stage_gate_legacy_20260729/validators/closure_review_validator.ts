import type {
  ClosureChecks,
  ClosureFindingCode,
  ClosureReviewEnvelope,
  ClosureReviewTaskEnvelope,
  ClosureScopeRef,
  ValidationReport,
} from "../contracts/index.ts";
import {
  CLOSURE_CHECK_NAMES,
  CLOSURE_FINDING_REGISTRY,
  FINALIZATION_REQUIREMENTS,
  canonicalJson,
} from "../contracts/index.ts";
import {
  addError,
  mergeReports,
  validateSchema,
} from "./schema_validator.ts";
import {
  validatePayloadEnvelopeBinding,
  type ExpectedTurnBinding,
} from "./envelope_validator.ts";

interface ExpectedIssue {
  check: keyof ClosureChecks;
  code: ClosureFindingCode;
  scope: ClosureScopeRef;
}

export function validateClosureReview(
  result: ClosureReviewEnvelope,
  task: ClosureReviewTaskEnvelope,
): ValidationReport {
  const schema = validateSchema("CLOSURE_REVIEW", result);
  if (!schema.valid) return schema;
  const expected: ExpectedTurnBinding = {
    role: "closure_reviewer",
    task,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
  };
  const review = result.payload;
  const report = mergeReports(
    schema,
    validatePayloadEnvelopeBinding(result, expected),
  );
  const bundle = task.payload.stopCandidateBundle;
  if (
    review.stopCandidateId !== bundle.candidate.stopCandidateId ||
    review.canonicalRevision !== bundle.candidate.canonicalRevision
  ) {
    addError(
      report,
      "closure.review_binding",
      "/payload",
      "ClosureReview must bind the supplied StopCandidate and canonical revision",
    );
  }

  const { checks, issues } = deriveClosureFacts(task);
  for (const check of CLOSURE_CHECK_NAMES) {
    if (review.closureChecks[check] !== checks[check]) {
      addError(
        report,
        "closure.check_fact_mismatch",
        `/payload/closureChecks/${check}`,
        "reported closure check disagrees with the supplied canonical projection",
      );
    }
  }
  validateClosureDecision(result, task, checks, issues, report);
  return report;
}

export function deriveClosureFacts(task: ClosureReviewTaskEnvelope): {
  checks: ClosureChecks;
  issues: ExpectedIssue[];
} {
  const payload = task.payload;
  const bundle = payload.stopCandidateBundle;
  const issues: ExpectedIssue[] = [];
  const add = (
    check: keyof ClosureChecks,
    code: ClosureFindingCode,
    scope: ClosureScopeRef,
  ) => {
    const issue = { check, code, scope };
    if (!issues.some((candidate) => issueKey(candidate) === issueKey(issue))) {
      issues.push(issue);
    }
  };

  const current =
    bundle.candidate.canonicalRevision === payload.currentCanonicalRevision &&
    bundle.proof.canonicalRevision === payload.currentCanonicalRevision &&
    task.stateBinding.canonicalRevision === payload.currentCanonicalRevision;
  if (!current) {
    add("stopProofRevisionCurrent", "stale_stop_proof_revision", scope("run", task.runId));
  }

  const proof = bundle.proof;
  const anchorRefs = payload.anchors.map((item) => item.anchorRef);
  const directionRefs = payload.directions.map((item) => item.directionRef);
  const openNeeds = payload.needs
    .filter((need) => need.status === "pending")
    .map((need) => need.needId)
    .sort();
  const matchesCanonical =
    bundle.candidate.runId === task.runId &&
    bundle.candidate.topicId === payload.topic.topicId &&
    bundle.candidate.stopProofId === proof.proofId &&
    proof.stopCandidateId === bundle.candidate.stopCandidateId &&
    proof.topicFrameRevision === payload.topic.revision &&
    sameSet(proof.anchorRefs, anchorRefs) &&
    sameSet(proof.directionRefs, directionRefs) &&
    sameStringSet(proof.openNeedIds, openNeeds) &&
    sameStringSet(
      proof.pendingTaskIds,
      payload.taskIndex
        .filter((item) => item.status === "pending")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.inFlightTaskIds,
      payload.taskIndex
        .filter((item) => item.status === "in_flight")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.pendingOutputRetryTaskIds,
      payload.outputAttemptIndex
        .filter((item) => item.status === "pending_retry")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.unconsumedResultIds,
      payload.resultIndex
        .filter((item) => item.status === "unconsumed")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.uncommittedDeltaIds,
      payload.deltaIndex
        .filter((item) => item.status === "uncommitted")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.unresolvedValidationFailureIds,
      payload.validationFailureIndex
        .filter((item) => item.status === "unresolved")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.failedTaskIds,
      payload.taskIndex
        .filter((item) => item.status === "failed_unresolved")
        .map((item) => item.id),
    ) &&
    sameStringSet(
      proof.unreviewedCriticalContradictionIds,
      payload.contradictions
        .filter(
          (item) => item.critical && !item.dispositionReviewId,
        )
        .map((item) => item.contradictionId),
    ) &&
    sameStringSet(
      proof.experimentHandoffIds,
      payload.experimentHandoffs.map((item) => item.handoffId),
    ) &&
    proof.lastTopicExpansionNeedId ===
      (payload.lastTopicExpansion?.needId ?? null) &&
    proof.outputCoverageProjectionId === payload.outputCoverage.projectionId;
  if (!matchesCanonical) {
    add(
      "stopProofMatchesCanonical",
      "stop_proof_canonical_mismatch",
      scope("stop_proof", proof.proofId, proof.canonicalRevision),
    );
  }

  const preflight =
    payload.mechanicalPreflight.passed &&
    payload.mechanicalPreflight.stopCandidateId ===
      bundle.candidate.stopCandidateId &&
    payload.mechanicalPreflight.canonicalRevision ===
      payload.currentCanonicalRevision &&
    payload.mechanicalPreflight.checks.every((check) => check.passed);
  if (!preflight) {
    add(
      "mechanicalPreflightPassed",
      "mechanical_preflight_failed",
      scope("preflight", payload.mechanicalPreflight.preflightId),
    );
  }

  const scopePreserved =
    payload.topic.scopeAudit.initialFingerprint ===
      payload.topic.scopeAudit.currentFingerprint ||
    payload.topic.scopeAudit.changes.every((change) => change.userAuthorized);
  if (!scopePreserved) {
    add(
      "topicScopePreserved",
      "topic_scope_silently_narrowed",
      scope("topic", payload.topic.topicId, payload.topic.revision),
    );
  }

  for (const need of payload.needs) {
    if (
      need.status === "pending" &&
      need.critical &&
      ["knowledge_base", "unknown"].includes(need.answerability)
    ) {
      add(
        "noKnowledgeAnswerableCriticalNeed",
        "knowledge_answerable_open_need",
        scope("search_need", need.needId, need.revision),
      );
    }
  }
  for (const anchor of payload.anchors) {
    if (!["saturated", "rejected"].includes(anchor.status)) {
      add(
        "allAnchorsClosed",
        "anchor_not_closed",
        asScope(anchor.anchorRef),
      );
    } else if (
      anchor.status === "saturated" &&
      !anchor.saturationReason?.trim()
    ) {
      add(
        "allAnchorsClosed",
        "anchor_missing_saturation_reason",
        asScope(anchor.anchorRef),
      );
    } else if (anchor.status === "rejected" && !anchor.statusReason.trim()) {
      add(
        "allAnchorsClosed",
        "anchor_missing_status_reason",
        asScope(anchor.anchorRef),
      );
    }
  }
  for (const direction of payload.directions) {
    if (
      !["testable", "experiment_required", "rejected"].includes(direction.status)
    ) {
      add(
        "allDirectionsTerminal",
        "direction_nonterminal",
        asScope(direction.directionRef),
      );
    } else if (!direction.statusReason.trim()) {
      add(
        "allDirectionsTerminal",
        "direction_missing_terminal_reason",
        asScope(direction.directionRef),
      );
    }
  }

  if (!payload.lastTopicExpansion) {
    add(
      "lastTopicExpansionNoDelta",
      "last_topic_expansion_missing",
      scope("topic", payload.topic.topicId, payload.topic.revision),
    );
  } else if (
    !payload.lastTopicExpansion.completed ||
    payload.lastTopicExpansion.outcome !== "no_new_anchor_no_critical_delta" ||
    !payload.lastTopicExpansion.noDeltaRecordId ||
    payload.lastTopicExpansion.semanticDeltaId !== null ||
    !payload.recentNoDeltaRecords.some(
      (ref) =>
        ref.objectType === "no_delta" &&
        ref.objectId === payload.lastTopicExpansion!.noDeltaRecordId,
    )
  ) {
    add(
      "lastTopicExpansionNoDelta",
      "last_topic_expansion_not_quiet",
      scope(
        "search_need",
        payload.lastTopicExpansion.needId,
        payload.lastTopicExpansion.needRevision,
      ),
    );
  }

  addWorkIssues(payload.taskIndex, "pending", "pending_task", add);
  addWorkIssues(payload.taskIndex, "in_flight", "in_flight_task", add);
  addWorkIssues(
    payload.outputAttemptIndex,
    "pending_retry",
    "pending_output_retry",
    add,
  );
  addWorkIssues(payload.resultIndex, "unconsumed", "unconsumed_result", add);
  addWorkIssues(payload.deltaIndex, "uncommitted", "uncommitted_delta", add);
  addWorkIssues(
    payload.validationFailureIndex,
    "unresolved",
    "unresolved_validation_failure",
    add,
  );
  addWorkIssues(payload.taskIndex, "failed_unresolved", "failed_task", add);

  for (const contradiction of payload.contradictions) {
    if (contradiction.critical && !contradiction.dispositionReviewId) {
      add(
        "criticalContradictionsReviewed",
        "unreviewed_critical_contradiction",
        asScope(contradiction.objectRef),
      );
    }
  }

  const handoffByDirection = new Map<
    string,
    typeof payload.experimentHandoffs
  >();
  for (const handoff of payload.experimentHandoffs) {
    const current = handoffByDirection.get(handoff.directionId) ?? [];
    current.push(handoff);
    handoffByDirection.set(handoff.directionId, current);
  }
  for (const direction of payload.directions) {
    const handoffs = handoffByDirection.get(direction.directionRef.objectId) ?? [];
    if (direction.status === "experiment_required" && handoffs.length === 0) {
      add(
        "experimentHandoffsComplete",
        "experiment_handoff_missing",
        asScope(direction.directionRef),
      );
    } else if (
      direction.status === "experiment_required" &&
      (handoffs.length !== 1 ||
        !handoffs[0]!.complete ||
        handoffs[0]!.executionAuthorized !== false ||
        direction.experimentHandoffId !== handoffs[0]!.handoffId)
    ) {
      add(
        "experimentHandoffsComplete",
        "experiment_handoff_invalid",
        asScope(direction.directionRef),
      );
    } else if (direction.status !== "experiment_required" && handoffs.length > 0) {
      add(
        "experimentHandoffsComplete",
        "experiment_handoff_invalid",
        asScope(direction.directionRef),
      );
    }
  }
  for (const handoff of payload.experimentHandoffs) {
    if (
      !payload.directions.some(
        (direction) =>
          direction.directionRef.objectId === handoff.directionId &&
          direction.status === "experiment_required",
      )
    ) {
      add(
        "experimentHandoffsComplete",
        "experiment_handoff_invalid",
        scope("experiment_handoff", handoff.handoffId, 1),
      );
    }
  }

  if (payload.runtimeEligibility.budgetExhausted) {
    add(
      "runtimeEligibleForCompletion",
      "runtime_budget_exhausted",
      scope("run", task.runId),
    );
  }
  if (
    payload.lifecycle !== "closure_preflight" ||
    payload.runtimeEligibility.paused ||
    payload.runtimeEligibility.blocked ||
    payload.runtimeEligibility.failed ||
    !payload.freshTurn ||
    payload.providerHistoryIncluded ||
    !payload.canonicalOnly
  ) {
    add(
      "runtimeEligibleForCompletion",
      "runtime_failed_or_paused",
      scope("run", task.runId),
    );
  }

  const coverageFields = Object.values(payload.outputCoverage.fields);
  if (
    coverageFields.length !== 7 ||
    coverageFields.some((refs) => refs.length === 0)
  ) {
    add(
      "finalOutputTraceable",
      "final_output_missing_field",
      scope("output_coverage", payload.outputCoverage.projectionId),
    );
  }
  const suppliedRefs = new Set(
    [
      { objectType: "topic", objectId: payload.topic.topicId, revision: payload.topic.revision },
      ...anchorRefs,
      ...directionRefs,
      ...payload.needs.map((need) => ({
        objectType: "search_need",
        objectId: need.needId,
        revision: need.revision,
      })),
      ...payload.resultIndex
        .map((item) => item.objectRef)
        .filter((ref): ref is NonNullable<typeof ref> => ref !== null),
      ...payload.deltaIndex
        .map((item) => item.objectRef)
        .filter((ref): ref is NonNullable<typeof ref> => ref !== null),
      ...payload.contradictions.map((item) => item.objectRef),
      ...payload.experimentHandoffs.map((item) => ({
        objectType: "experiment_handoff",
        objectId: item.handoffId,
        revision: 1,
      })),
    ].map(refKey),
  );
  if (
    coverageFields.flat().some((ref) => !suppliedRefs.has(refKey(ref)))
  ) {
    add(
      "finalOutputTraceable",
      "final_output_untraceable",
      scope("output_coverage", payload.outputCoverage.projectionId),
    );
  }

  const derivedClaims = {
    topicScopePreserved: !issues.some(
      (issue) => issue.check === "topicScopePreserved",
    ),
    noKnowledgeAnswerableCriticalNeed: !issues.some(
      (issue) => issue.check === "noKnowledgeAnswerableCriticalNeed",
    ),
    allAnchorsClosed: !issues.some(
      (issue) => issue.check === "allAnchorsClosed",
    ),
    allDirectionsTerminal: !issues.some(
      (issue) => issue.check === "allDirectionsTerminal",
    ),
    lastTopicExpansionNoDelta: !issues.some(
      (issue) => issue.check === "lastTopicExpansionNoDelta",
    ),
    noUnconsumedOrUncommittedWork: !issues.some(
      (issue) => issue.check === "noUnconsumedOrUncommittedWork",
    ),
    criticalContradictionsReviewed: !issues.some(
      (issue) => issue.check === "criticalContradictionsReviewed",
    ),
    experimentHandoffsComplete: !issues.some(
      (issue) => issue.check === "experimentHandoffsComplete",
    ),
    runtimeEligibleForCompletion: !issues.some(
      (issue) => issue.check === "runtimeEligibleForCompletion",
    ),
    finalOutputTraceable: !issues.some(
      (issue) => issue.check === "finalOutputTraceable",
    ),
  };
  if (canonicalJson(proof.claims) !== canonicalJson(derivedClaims)) {
    add(
      "stopProofMatchesCanonical",
      "stop_proof_canonical_mismatch",
      scope("stop_proof", proof.proofId, proof.canonicalRevision),
    );
  }

  const checks = Object.fromEntries(
    CLOSURE_CHECK_NAMES.map((check) => [
      check,
      !issues.some((issue) => issue.check === check),
    ]),
  ) as unknown as ClosureChecks;
  return { checks, issues };
}

function validateClosureDecision(
  result: ClosureReviewEnvelope,
  task: ClosureReviewTaskEnvelope,
  expectedChecks: ClosureChecks,
  expectedIssues: ExpectedIssue[],
  report: ValidationReport,
): void {
  const review = result.payload;
  const allTrue = CLOSURE_CHECK_NAMES.every((check) => expectedChecks[check]);
  if (review.decision === "accept") {
    if (
      !allTrue ||
      review.blockingFindings.length !== 0 ||
      review.reopenScopes.length !== 0 ||
      !review.allowsFinalization ||
      !sameStringSet(
        review.finalizationRequirements,
        FINALIZATION_REQUIREMENTS as unknown as string[],
      )
    ) {
      addError(
        report,
        "closure.accept_matrix",
        "/payload",
        "accept requires all thirteen checks, no findings/scopes, and the five fixed finalization requirements",
      );
    }
    const basisChecks = new Set(
      review.verifiedClosureBasis.map((basis) => basis.check),
    );
    if (CLOSURE_CHECK_NAMES.some((check) => !basisChecks.has(check))) {
      addError(
        report,
        "closure.accept_basis",
        "/payload/verifiedClosureBasis",
        "accept requires at least one resolvable basis for every closure check",
      );
    }
  } else {
    if (
      allTrue ||
      review.blockingFindings.length === 0 ||
      review.allowsFinalization ||
      review.finalizationRequirements.length !== 0
    ) {
      addError(
        report,
        "closure.reject_matrix",
        "/payload",
        "reject requires at least one false check/finding and cannot allow finalization",
      );
    }
  }

  review.verifiedClosureBasis.forEach((basis, index) => {
    if (!expectedChecks[basis.check]) {
      addError(
        report,
        "closure.basis_for_false_check",
        `/payload/verifiedClosureBasis/${index}/check`,
        "verified basis may cover only true checks",
      );
    }
    const supplied = closureScopeKeys(task);
    if (
      basis.objectRefs.length === 0 ||
      basis.objectRefs.some((ref) => !supplied.has(scopeKey(ref)))
    ) {
      addError(
        report,
        "closure.fabricated_basis_ref",
        `/payload/verifiedClosureBasis/${index}/objectRefs`,
        "verified closure basis must contain only resolvable supplied refs",
      );
    }
  });

  const actualIssueKeys = new Set<string>();
  review.blockingFindings.forEach((finding, index) => {
    const rule = CLOSURE_FINDING_REGISTRY[finding.code];
    if (
      finding.check !== rule.check ||
      finding.type !== rule.type ||
      finding.recoveryAction !== rule.recoveryAction
    ) {
      addError(
        report,
        "closure.finding_registry",
        `/payload/blockingFindings/${index}`,
        "finding check/type/code/recoveryAction violates the fixed registry",
      );
    }
    const key = issueKey({
      check: finding.check,
      code: finding.code,
      scope: finding.reopenScope,
    });
    if (actualIssueKeys.has(key)) {
      addError(
        report,
        "closure.duplicate_finding",
        `/payload/blockingFindings/${index}`,
        "blocking finding is duplicated",
      );
    }
    actualIssueKeys.add(key);
  });
  const expectedKeys = new Set(expectedIssues.map(issueKey));
  if (
    [...expectedKeys].some((key) => !actualIssueKeys.has(key)) ||
    [...actualIssueKeys].some((key) => !expectedKeys.has(key))
  ) {
    addError(
      report,
      "closure.finding_coverage",
      "/payload/blockingFindings",
      "findings must exactly cover every supplied blocking object and no others",
    );
  }
  const expectedScopes = new Set(
    review.blockingFindings.map((finding) => canonicalJson(finding.reopenScope)),
  );
  const actualScopes = new Set(
    review.reopenScopes.map((scope) => canonicalJson(scope)),
  );
  if (
    [...expectedScopes].some((key) => !actualScopes.has(key)) ||
    [...actualScopes].some((key) => !expectedScopes.has(key))
  ) {
    addError(
      report,
      "closure.reopen_scope_set",
      "/payload/reopenScopes",
      "reopenScopes must be the exact deduplicated set from blocking findings",
    );
  }
}

function closureScopeKeys(
  task: ClosureReviewTaskEnvelope,
): Set<string> {
  const payload = task.payload;
  const refs: ClosureScopeRef[] = [
    scope("run", task.runId),
    scope("topic", payload.topic.topicId, payload.topic.revision),
    scope(
      "stop_proof",
      payload.stopCandidateBundle.proof.proofId,
      payload.stopCandidateBundle.proof.canonicalRevision,
    ),
    scope("preflight", payload.mechanicalPreflight.preflightId),
    scope("output_coverage", payload.outputCoverage.projectionId),
    ...payload.anchors.map((item) => asScope(item.anchorRef)),
    ...payload.directions.map((item) => asScope(item.directionRef)),
    ...payload.needs.map((item) =>
      scope("search_need", item.needId, item.revision),
    ),
    ...[
      ...payload.taskIndex,
      ...payload.resultIndex,
      ...payload.deltaIndex,
      ...payload.outputAttemptIndex,
      ...payload.validationFailureIndex,
    ].map((item) =>
      item.objectRef ? asScope(item.objectRef) : scope("work_item", item.id),
    ),
    ...payload.contradictions.map((item) => asScope(item.objectRef)),
    ...payload.experimentHandoffs.map((item) =>
      scope("experiment_handoff", item.handoffId, 1),
    ),
  ];
  return new Set(refs.map(scopeKey));
}

function scopeKey(ref: ClosureScopeRef): string {
  return `${ref.objectType}\0${ref.objectId}\0${ref.revision ?? ""}`;
}

function addWorkIssues(
  items: ClosureReviewTaskEnvelope["payload"]["taskIndex"],
  status: string,
  code: ClosureFindingCode,
  add: (
    check: keyof ClosureChecks,
    code: ClosureFindingCode,
    scope: ClosureScopeRef,
  ) => void,
): void {
  for (const item of items) {
    if (item.status === status && !item.resolvedById) {
      add(
        "noUnconsumedOrUncommittedWork",
        code,
        item.objectRef ? asScope(item.objectRef) : scope("work_item", item.id),
      );
    }
  }
}

function scope(
  objectType: string,
  objectId: string,
  revision: number | null = null,
): ClosureScopeRef {
  return { objectType, objectId, revision };
}

function asScope(ref: {
  objectType: string;
  objectId: string;
  revision: number;
}): ClosureScopeRef {
  return { ...ref };
}

function refKey(ref: {
  objectType: string;
  objectId: string;
  revision: number;
}): string {
  return `${ref.objectType}\0${ref.objectId}\0${ref.revision}`;
}

function sameSet(
  left: Array<{ objectType: string; objectId: string; revision: number }>,
  right: Array<{ objectType: string; objectId: string; revision: number }>,
): boolean {
  return sameStringSet(left.map(refKey), right.map(refKey));
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  const a = new Set(left);
  const b = new Set(right);
  return a.size === b.size && [...a].every((value) => b.has(value));
}

function issueKey(issue: ExpectedIssue): string {
  return `${issue.check}\0${issue.code}\0${canonicalJson(issue.scope)}`;
}
