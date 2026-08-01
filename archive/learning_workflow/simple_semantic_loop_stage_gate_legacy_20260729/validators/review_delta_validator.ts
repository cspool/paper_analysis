import type {
  DirectionReviewTaskEnvelope,
  ReviewDeltaEnvelope,
  ValidationReport,
} from "../contracts/index.ts";
import {
  validateExperimentHandoff,
} from "../security/no_experiment_guard.ts";
import {
  addError,
  mergeReports,
  validateSchema,
} from "./schema_validator.ts";
import {
  validatePayloadEnvelopeBinding,
  type ExpectedTurnBinding,
} from "./envelope_validator.ts";

const CORE_CHECKS = [
  "inTopicAndAnchorScope",
  "baselineFair",
  "minimumChangeSetExplicit",
  "causalChainFalsifiable",
  "implementationPathBounded",
  "measurementPlanComplete",
  "falsifiersPresent",
  "criticalCounterexampleResolved",
  "evidenceTraceable",
] as const;

export function validateReviewDelta(
  result: ReviewDeltaEnvelope,
  task: DirectionReviewTaskEnvelope,
): ValidationReport {
  const schema = validateSchema("REVIEW_DELTA", result);
  if (!schema.valid) return schema;
  const expected: ExpectedTurnBinding = {
    role: "direction_reviewer",
    task,
    stateBinding: task.stateBinding,
    inputHash: task.inputHash,
  };
  const review = result.payload;
  const report = mergeReports(
    schema,
    validatePayloadEnvelopeBinding(result, expected),
    validateExperimentHandoff(review.experimentHandoff),
  );
  if (
    review.directionId !== task.payload.direction.directionId ||
    review.directionRevision !== task.payload.direction.revision
  ) {
    addError(
      report,
      "review.direction_binding",
      "/payload/directionId",
      "ReviewDelta must bind the supplied Direction revision",
    );
  }
  const allowedEvidence = new Set(task.payload.allowedEvidenceIds);
  review.evidenceRefsUsed.forEach((evidenceId, index) => {
    if (!allowedEvidence.has(evidenceId)) {
      addError(
        report,
        "review.fabricated_evidence_ref",
        `/payload/evidenceRefsUsed/${index}`,
        "Reviewer may cite only supplied committed Evidence",
      );
    }
  });
  validateReadinessCoupling(result, task, report);
  validateDecisionMatrix(result, task, report);
  return report;
}

function validateReadinessCoupling(
  result: ReviewDeltaEnvelope,
  task: DirectionReviewTaskEnvelope,
  report: ValidationReport,
): void {
  const review = result.payload;
  const checks = review.readinessChecks;
  if (checks.baselineFair !== (review.baselineProblem === null)) {
    addError(
      report,
      "review.baseline_problem_coupling",
      "/payload/baselineProblem",
      "baselineFair is false iff baselineProblem is non-null",
    );
  }
  if (
    checks.implementationPathBounded !==
    (review.implementationProblem === null)
  ) {
    addError(
      report,
      "review.implementation_problem_coupling",
      "/payload/implementationProblem",
      "implementationPathBounded is false iff implementationProblem is non-null",
    );
  }
  const measurementGap =
    !checks.measurementPlanComplete || !checks.falsifiersPresent;
  if (measurementGap !== (review.measurementProblem !== null)) {
    addError(
      report,
      "review.measurement_problem_coupling",
      "/payload/measurementProblem",
      "measurementProblem is non-null iff measurement or falsifier readiness is false",
    );
  }
  if (!checks.causalChainFalsifiable && !review.weakestCausalLink) {
    addError(
      report,
      "review.weakest_causal_link",
      "/payload/weakestCausalLink",
      "a non-falsifiable causal chain requires the weakest link",
    );
  }
  if (!checks.criticalCounterexampleResolved) {
    if (!review.strongestCounterexample || review.counterexampleResolution !== null) {
      addError(
        report,
        "review.unresolved_counterexample_coupling",
        "/payload",
        "unresolved critical counterexample requires a statement and null resolution",
      );
    }
  }
  if (task.payload.counterexamples.length === 0) {
    if (
      !checks.criticalCounterexampleResolved ||
      review.strongestCounterexample !== null ||
      review.counterexampleResolution !== null
    ) {
      addError(
        report,
        "review.no_counterexample_matrix",
        "/payload",
        "without supplied counterexamples the check is true and both fields are null",
      );
    }
  } else if (
    checks.criticalCounterexampleResolved &&
    (review.strongestCounterexample === null ||
      review.counterexampleResolution === null)
  ) {
    addError(
      report,
      "review.resolved_counterexample_pair",
      "/payload",
      "a supplied resolved counterexample requires both statement and resolution",
    );
  }
  if (task.payload.counterexamples.length > 0) {
    const strongest = task.payload.counterexamples.find(
      (counterexample) =>
        counterexample.statement === review.strongestCounterexample,
    );
    if (!strongest) {
      addError(
        report,
        "review.counterexample_not_supplied",
        "/payload/strongestCounterexample",
        "the strongest counterexample must exactly match one supplied counterexample",
      );
    } else if (
      checks.criticalCounterexampleResolved &&
      !strongest.evidenceRefs.some((evidenceId) =>
        review.evidenceRefsUsed.includes(evidenceId),
      )
    ) {
      addError(
        report,
        "review.counterexample_resolution_evidence",
        "/payload/counterexampleResolution",
        "a resolved counterexample must use counterevidence bound to that supplied counterexample",
      );
    }
  }
  if (checks.evidenceTraceable && review.evidenceRefsUsed.length === 0) {
    addError(
      report,
      "review.traceable_evidence_missing",
      "/payload/evidenceRefsUsed",
      "evidenceTraceable=true requires at least one supplied Evidence ref",
    );
  }
}

function validateDecisionMatrix(
  result: ReviewDeltaEnvelope,
  task: DirectionReviewTaskEnvelope,
  report: ValidationReport,
): void {
  const review = result.payload;
  const checks = review.readinessChecks;
  const duplicateFieldsNull =
    review.duplicateOfDirectionRef === null &&
    review.duplicateComparison === null;
  switch (review.decision) {
    case "continue_search": {
      if (
        !review.nextQuestion ||
        !review.nextQuestionAnswerableFromKnowledgeBase ||
        !checks.knowledgeAnswerableCriticalGapRemaining ||
        checks.newExperimentRequired ||
        !checks.inTopicAndAnchorScope ||
        review.rejectionCategory !== null ||
        review.experimentHandoff !== null ||
        !duplicateFieldsNull
      ) {
        addError(
          report,
          "review.continue_search_matrix",
          "/payload",
          "continue_search requires one knowledge-base question and no rejection, duplicate, or experiment payload",
        );
      }
      const concreteGap =
        CORE_CHECKS.some((name) => !checks[name]) ||
        [
          review.weakestCausalLink,
          review.baselineProblem,
          review.implementationProblem,
          review.measurementProblem,
          review.strongestCounterexample,
        ].some((value) => value !== null);
      if (!concreteGap) {
        addError(
          report,
          "review.continue_search_gap_missing",
          "/payload",
          "continue_search must identify a concrete critical gap",
        );
      }
      break;
    }
    case "testable": {
      if (
        CORE_CHECKS.some((name) => !checks[name]) ||
        checks.knowledgeAnswerableCriticalGapRemaining ||
        checks.newExperimentRequired ||
        review.nextQuestion !== null ||
        review.nextQuestionAnswerableFromKnowledgeBase ||
        !duplicateFieldsNull ||
        review.rejectionCategory !== null ||
        review.experimentHandoff !== null ||
        review.baselineProblem !== null ||
        review.implementationProblem !== null ||
        review.measurementProblem !== null ||
        review.supportedParts.length === 0
      ) {
        addError(
          report,
          "review.testable_matrix",
          "/payload",
          "testable requires all nine core checks, no remaining gap payload, and supported parts",
        );
      }
      break;
    }
    case "experiment_required": {
      const hardChecks = [
        "inTopicAndAnchorScope",
        "baselineFair",
        "minimumChangeSetExplicit",
        "causalChainFalsifiable",
        "falsifiersPresent",
        "evidenceTraceable",
      ] as const;
      if (
        hardChecks.some((name) => !checks[name]) ||
        checks.knowledgeAnswerableCriticalGapRemaining ||
        !checks.newExperimentRequired ||
        review.nextQuestion !== null ||
        review.nextQuestionAnswerableFromKnowledgeBase ||
        !duplicateFieldsNull ||
        review.rejectionCategory !== null ||
        review.experimentHandoff === null ||
        review.supportedParts.length === 0
      ) {
        addError(
          report,
          "review.experiment_required_matrix",
          "/payload",
          "experiment_required requires semantic readiness, no knowledge-base gap, and a non-executable handoff",
        );
      } else if (
        review.experimentHandoff.directionId !== review.directionId ||
        review.experimentHandoff.executionAuthorized !== false
      ) {
        addError(
          report,
          "review.experiment_handoff_binding",
          "/payload/experimentHandoff",
          "handoff must bind the reviewed Direction and remain non-executable",
        );
      }
      break;
    }
    case "rejected": {
      if (
        review.rejectionCategory === null ||
        review.nextQuestion !== null ||
        review.nextQuestionAnswerableFromKnowledgeBase ||
        checks.knowledgeAnswerableCriticalGapRemaining ||
        checks.newExperimentRequired ||
        review.experimentHandoff !== null
      ) {
        addError(
          report,
          "review.rejected_matrix",
          "/payload",
          "rejected requires one category and no next-question/gap/experiment payload",
        );
      }
      validateRejectionCategory(result, task, report);
      break;
    }
  }
  if (review.decision !== "rejected" && review.rejectionCategory !== null) {
    addError(
      report,
      "review.category_only_on_reject",
      "/payload/rejectionCategory",
      "rejectionCategory is only legal for rejected",
    );
  }
}

function validateRejectionCategory(
  result: ReviewDeltaEnvelope,
  task: DirectionReviewTaskEnvelope,
  report: ValidationReport,
): void {
  const review = result.payload;
  const checks = review.readinessChecks;
  const noDuplicate =
    review.duplicateOfDirectionRef === null &&
    review.duplicateComparison === null;
  const coreAllTrue = CORE_CHECKS.every((name) => checks[name]);
  switch (review.rejectionCategory) {
    case "duplicate": {
      const comparison = review.duplicateComparison;
      const siblingRefs = new Set(
        task.payload.siblingDirections.map(
          (sibling) =>
            `${sibling.directionRef.objectType}:${sibling.directionRef.objectId}:${sibling.directionRef.revision}`,
        ),
      );
      const ref = review.duplicateOfDirectionRef;
      const refKey = ref
        ? `${ref.objectType}:${ref.objectId}:${ref.revision}`
        : "";
      if (
        !ref ||
        !siblingRefs.has(refKey) ||
        !comparison ||
        !comparison.baselineScopeEquivalent ||
        !comparison.primaryChangeEquivalent ||
        !comparison.causalTargetEquivalent ||
        comparison.materialDifference !== null
      ) {
        addError(
          report,
          "review.duplicate_matrix",
          "/payload",
          "duplicate requires a supplied sibling ref and three exact equivalence findings with no material difference",
        );
      }
      return;
    }
    case "out_of_scope":
      if (!noDuplicate || checks.inTopicAndAnchorScope) {
        rejectionError(report, "out_of_scope requires scope=false");
      }
      break;
    case "causal_contradiction":
      if (
        !noDuplicate ||
        task.payload.contradictingEvidence.length === 0 ||
        (checks.causalChainFalsifiable &&
          checks.criticalCounterexampleResolved)
      ) {
        rejectionError(
          report,
          "causal_contradiction requires supplied contradiction and a failed causal/counterexample check",
        );
      }
      break;
    case "unfair_comparison":
      if (!noDuplicate || checks.baselineFair) {
        rejectionError(report, "unfair_comparison requires baselineFair=false");
      }
      break;
    case "no_performance_mechanism":
      if (!noDuplicate || checks.causalChainFalsifiable) {
        rejectionError(
          report,
          "no_performance_mechanism requires causalChainFalsifiable=false",
        );
      }
      break;
    case "invalid_evidence":
      if (!noDuplicate || checks.evidenceTraceable) {
        rejectionError(report, "invalid_evidence requires evidenceTraceable=false");
      }
      break;
    case "other":
      if (!noDuplicate || coreAllTrue) {
        rejectionError(
          report,
          "other requires no duplicate binding and at least one failed core check",
        );
      }
      break;
    case null:
      break;
  }
  if (review.rejectionCategory !== "duplicate" && !noDuplicate) {
    rejectionError(
      report,
      "duplicate fields must both be null outside duplicate rejection",
    );
  }
  if (review.rejectionCategory !== "duplicate" && coreAllTrue) {
    rejectionError(
      report,
      "non-duplicate rejection cannot have all nine core checks true",
    );
  }
}

function rejectionError(report: ValidationReport, message: string): void {
  addError(
    report,
    "review.rejection_category_matrix",
    "/payload/rejectionCategory",
    message,
  );
}
