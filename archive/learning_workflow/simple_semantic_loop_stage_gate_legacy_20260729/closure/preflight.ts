import { randomUUID } from "node:crypto";
import type {
  ClosureReviewTaskEnvelope,
  MechanicalPreflightReport,
  ValidationReport,
} from "../contracts/index.ts";
import { deriveClosureFacts } from "../validators/closure_review_validator.ts";
import { addError, emptyReport } from "../validators/schema_validator.ts";

export interface ClosurePreflightResult {
  report: MechanicalPreflightReport;
  validation: ValidationReport;
}

export function runClosurePreflight(
  task: ClosureReviewTaskEnvelope,
): ClosurePreflightResult {
  const validation = emptyReport();
  const { checks, issues } = deriveClosureFacts(task);
  const preflightChecks: MechanicalPreflightReport["checks"] = [
    {
      checkId: "candidate_proof_revision_current",
      passed: checks.stopProofRevisionCurrent,
      issueIds: issues
        .filter((issue) => issue.check === "stopProofRevisionCurrent")
        .map((issue) => issue.code),
    },
    {
      checkId: "proof_matches_canonical_projection",
      passed: checks.stopProofMatchesCanonical,
      issueIds: issues
        .filter((issue) => issue.check === "stopProofMatchesCanonical")
        .map((issue) => issue.code),
    },
    {
      checkId: "no_unconsumed_or_uncommitted_work",
      passed: checks.noUnconsumedOrUncommittedWork,
      issueIds: issues
        .filter((issue) => issue.check === "noUnconsumedOrUncommittedWork")
        .map((issue) => issue.code),
    },
    {
      checkId: "handoffs_non_executable_and_complete",
      passed: checks.experimentHandoffsComplete,
      issueIds: issues
        .filter((issue) => issue.check === "experimentHandoffsComplete")
        .map((issue) => issue.code),
    },
    {
      checkId: "runtime_and_independence_eligible",
      passed: checks.runtimeEligibleForCompletion,
      issueIds: issues
        .filter((issue) => issue.check === "runtimeEligibleForCompletion")
        .map((issue) => issue.code),
    },
    {
      checkId: "final_output_coverage_resolves",
      passed: checks.finalOutputTraceable,
      issueIds: issues
        .filter((issue) => issue.check === "finalOutputTraceable")
        .map((issue) => issue.code),
    },
  ];
  const passed = preflightChecks.every((check) => check.passed);
  if (!passed) {
    preflightChecks.forEach((check, index) => {
      if (!check.passed) {
        addError(
          validation,
          "closure.preflight_failed",
          `/checks/${index}`,
          `${check.checkId}: ${check.issueIds.join(", ")}`,
        );
      }
    });
  }
  return {
    report: {
      preflightId: `preflight-${randomUUID()}`,
      stopCandidateId:
        task.payload.stopCandidateBundle.candidate.stopCandidateId,
      canonicalRevision: task.payload.currentCanonicalRevision,
      checks: preflightChecks,
      passed,
    },
    validation,
  };
}

