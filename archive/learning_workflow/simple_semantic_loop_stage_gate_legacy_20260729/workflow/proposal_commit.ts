import { randomUUID } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import type {
  StateBinding,
  ValidationReport,
  WorkflowDecisionProposal,
  WorkflowPlanPatch,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import { canonicalJson } from "../contracts/index.ts";
import {
  CasConflictError,
  type WorkflowStore,
} from "../db/workflow_store.ts";
import { validateWorkflowDecisionProposal } from "../validators/index.ts";
import { addError } from "../validators/schema_validator.ts";
import {
  applyPlanPatch,
  loadCurrentPlan,
  persistPlanRevision,
} from "./plan_store.ts";
import type { GateEvaluation } from "../stages/gate_engine.ts";

export interface ProposalCommitResult {
  proposalId: string;
  accepted: boolean;
  nextState: StateBinding | null;
  rejectionCode: string | null;
}

export interface WorkflowProposalCommitAudit {
  gateId: string;
  gateEvaluation: GateEvaluation;
  rawResponseArtifactId: string;
}

export function commitWorkflowProposal(
  store: WorkflowStore,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
  audit: WorkflowProposalCommitAudit | null = null,
): ProposalCommitResult {
  const report = preflightWorkflowProposal(store, task, proposal);
  const validationReportId = store.insertValidationReport(
    task.runId,
    report,
    task.taskId,
    task.attemptId,
  );
  if (!report.valid) {
    recordRejectedProposal(
      store,
      task,
      proposal,
      validationReportId,
      "proposal_validation_failed",
    );
    return {
      proposalId: proposal.proposalId,
      accepted: false,
      nextState: null,
      rejectionCode: "proposal_validation_failed",
    };
  }
  try {
    const nextState = commitValidatedProposal(
      store,
      task,
      proposal,
      validationReportId,
      audit,
    );
    return {
      proposalId: proposal.proposalId,
      accepted: true,
      nextState,
      rejectionCode: null,
    };
  } catch (error) {
    if (error instanceof CasConflictError) {
      recordRejectedProposal(
        store,
        task,
        proposal,
        validationReportId,
        "stale_state_binding",
      );
      return {
        proposalId: proposal.proposalId,
        accepted: false,
        nextState: null,
        rejectionCode: "stale_state_binding",
      };
    }
    throw error;
  }
}

/**
 * Validate and compile an untrusted proposal without mutating durable state.
 * This converts errors that would otherwise be thrown by plan compilation
 * into normal semantic validation failures eligible for a fresh correction
 * Turn.
 */
export function preflightWorkflowProposal(
  store: WorkflowStore,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
): ValidationReport {
  const report = validateWorkflowDecisionProposal(proposal, task);
  if (!report.valid) return report;
  try {
    switch (proposal.decision) {
      case "RUN_STAGE":
      case "REQUEST_EVALUATION": {
        const current = loadCurrentPlan(store, task.runId);
        applyPlanPatch(
          current,
          proposalStagePatch(current, proposal),
          proposal.expectedState.snapshotVersion,
        );
        break;
      }
      case "REPLAN":
        applyPlanPatch(
          loadCurrentPlan(store, task.runId),
          proposal.proposedPlanPatch!,
          proposal.expectedState.snapshotVersion,
        );
        break;
      case "RETRY_STAGE": {
        const retryable = store.db
          .prepare(
            `SELECT 1 FROM workflow_plan_nodes
             WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
               AND lifecycle IN ('failed', 'blocked', 'runnable')`,
          )
          .get(
            task.runId,
            proposal.expectedState.workflowPlanRevision,
            proposal.targetStageId,
          );
        if (!retryable) {
          throw new Error("retry target is not an active retryable Stage");
        }
        break;
      }
      default:
        break;
    }
  } catch (error) {
    addError(
      report,
      "workflow.proposal_compile_failed",
      "/",
      error instanceof Error ? error.message : String(error),
    );
  }
  return report;
}

function proposalStagePatch(
  current: ReturnType<typeof loadCurrentPlan>,
  proposal: WorkflowDecisionProposal,
): WorkflowPlanPatch {
  const contract = proposal.proposedStageContract!;
  const gate = proposal.proposedGateDefinition!;
  return {
    expectedPlanRevision: current.revision,
    objectiveHash: current.objectiveHash,
    acceptanceCriteriaHash: current.acceptanceCriteriaHash,
    rationale: proposal.reason,
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
          gate,
        },
      },
    ],
  };
}

function commitValidatedProposal(
  store: WorkflowStore,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
  validationReportId: string,
  audit: WorkflowProposalCommitAudit | null,
): StateBinding {
  const expected = proposal.expectedState;
  switch (proposal.decision) {
    case "RUN_STAGE":
    case "REQUEST_EVALUATION": {
      const current = loadCurrentPlan(store, task.runId);
      const patch = proposalStagePatch(current, proposal);
      const { plan, frozenStages } = applyPlanPatch(
        current,
        patch,
        expected.snapshotVersion,
      );
      return persistPlanRevision(
        store,
        task.runId,
        expected,
        plan,
        frozenStages,
        "workflow_proposal_stage_accepted",
        (db) =>
          insertAcceptedProposal(
            db,
            task,
            proposal,
            validationReportId,
            audit,
          ),
      );
    }
    case "REPLAN": {
      const current = loadCurrentPlan(store, task.runId);
      const { plan, frozenStages } = applyPlanPatch(
        current,
        proposal.proposedPlanPatch!,
        expected.snapshotVersion,
      );
      return persistPlanRevision(
        store,
        task.runId,
        expected,
        plan,
        frozenStages,
        "workflow_replan_accepted",
        (db) =>
          insertAcceptedProposal(
            db,
            task,
            proposal,
            validationReportId,
            audit,
          ),
      );
    }
    case "RETRY_STAGE":
      return store.casTransition(
        task.runId,
        expected,
        {
          lifecycle: "running",
          currentStageId: proposal.targetStageId,
          eventType: "workflow_retry_stage_accepted",
          eventPayload: {
            proposalId: proposal.proposalId,
            targetStageId: proposal.targetStageId,
          },
        },
        (db) => {
          const changed = db
            .prepare(
              `UPDATE workflow_plan_nodes SET lifecycle = 'runnable'
               WHERE run_id = ? AND plan_revision = ? AND stage_id = ?
               AND lifecycle IN ('failed', 'blocked', 'runnable')`,
            )
            .run(
              task.runId,
              expected.workflowPlanRevision,
              proposal.targetStageId,
            );
          if (Number(changed.changes) !== 1) {
            throw new Error("retry target is not an active retryable Stage");
          }
          insertAcceptedProposal(
            db,
            task,
            proposal,
            validationReportId,
            audit,
          );
        },
      );
    case "ASK_USER":
      return commitControlProposal(
        store,
        task,
        proposal,
        validationReportId,
        audit,
        "waiting_user",
        "workflow_user_question_accepted",
        proposal.requestedUserInput,
      );
    case "REPORT_BLOCKED":
      return commitControlProposal(
        store,
        task,
        proposal,
        validationReportId,
        audit,
        proposal.blockedReport!.kind === "external"
          ? "blocked_external"
          : "blocked_semantic",
        "workflow_blocked_report_accepted",
        proposal.blockedReport,
      );
    case "PROPOSE_PAUSE":
      return commitControlProposal(
        store,
        task,
        proposal,
        validationReportId,
        audit,
        proposal.pauseProposal!.category === "budget"
          ? "paused_budget"
          : "paused_operator",
        "workflow_pause_proposal_accepted",
        proposal.pauseProposal,
      );
    case "PROPOSE_COMPLETE":
      return store.casTransition(
        task.runId,
        expected,
        {
          lifecycle: "closure_preflight",
          eventType: "stop_candidate_accepted_for_preflight",
          eventPayload: {
            proposalId: proposal.proposalId,
            stopCandidateId:
              proposal.domainProposal!.kind === "stop_candidate"
                ? proposal.domainProposal.value.candidate.stopCandidateId
                : null,
          },
        },
        (db) => {
          insertAcceptedProposal(
            db,
            task,
            proposal,
            validationReportId,
            audit,
          );
          if (proposal.domainProposal!.kind !== "stop_candidate") {
            throw new Error("validated completion proposal lost domain binding");
          }
          const bundle = proposal.domainProposal.value;
          db.prepare(
            `UPDATE canonical_objects SET active = 0
             WHERE run_id = ?
               AND object_type IN ('stop_candidate', 'stop_proof')
               AND active = 1`,
          ).run(task.runId);
          store.saveCanonicalObject(
            db,
            task.runId,
            "stop_candidate",
            bundle.candidate.stopCandidateId,
            1,
            bundle.candidate,
            null,
          );
          store.saveCanonicalObject(
            db,
            task.runId,
            "stop_proof",
            bundle.proof.proofId,
            1,
            bundle.proof,
            null,
          );
        },
      );
  }
}

function commitControlProposal(
  store: WorkflowStore,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
  validationReportId: string,
  audit: WorkflowProposalCommitAudit | null,
  lifecycle:
    | "waiting_user"
    | "blocked_external"
    | "blocked_semantic"
    | "paused_budget"
    | "paused_operator",
  eventType: string,
  payload: unknown,
): StateBinding {
  return store.casTransition(
    task.runId,
    proposal.expectedState,
    {
      lifecycle,
      pauseOrBlockReason: proposal.reason,
      eventType,
      eventPayload: { proposalId: proposal.proposalId, payload },
    },
    (db) => {
      insertAcceptedProposal(
        db,
        task,
        proposal,
        validationReportId,
        audit,
      );
      db.prepare(
        `INSERT INTO operator_requests(
           request_id, run_id, request_type, payload_json, status, created_at
         ) VALUES (?, ?, ?, ?, 'pending', ?)`,
      ).run(
        `operator-${randomUUID()}`,
        task.runId,
        proposal.decision,
        canonicalJson(payload),
        timestamp(),
      );
    },
  );
}

function insertAcceptedProposal(
  db: DatabaseSync,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
  validationReportId: string,
  audit: WorkflowProposalCommitAudit | null,
): void {
  db.prepare(
    `INSERT INTO decision_proposals(
       proposal_id, run_id, task_id, attempt_id, expected_state_json,
       decision_input_hash, decision, proposal_json, validation_report_id,
       status, rejection_code, created_at
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', NULL, ?)`,
  ).run(
    proposal.proposalId,
    task.runId,
    task.taskId,
    task.attemptId,
    canonicalJson(proposal.expectedState),
    proposal.decisionInputHash,
    proposal.decision,
    canonicalJson(proposal),
    validationReportId,
    timestamp(),
  );
  const now = timestamp();
  db.prepare(
    "UPDATE tasks SET status = 'committed', updated_at = ? WHERE task_id = ?",
  ).run(now, task.taskId);
  db.prepare(
    `UPDATE attempts
     SET status = 'committed', raw_response_artifact_id = ?,
         error_code = NULL, finished_at = ?
     WHERE attempt_id = ?`,
  ).run(audit?.rawResponseArtifactId ?? null, now, task.attemptId);
  if (audit) {
    db.prepare(
      `INSERT INTO gate_results(
         gate_result_id, run_id, stage_id, gate_id, result_id, passed,
         report_json, created_at
       ) VALUES (?, ?, ?, ?, NULL, 1, ?, ?)`,
    ).run(
      `gate-result-${randomUUID()}`,
      task.runId,
      task.stageId,
      audit.gateId,
      canonicalJson(audit.gateEvaluation),
      now,
    );
  }
  db.prepare(
    `UPDATE validation_reports SET resolved_by_id = ?
     WHERE task_id = ? AND valid = 0 AND resolved_by_id IS NULL`,
  ).run(task.attemptId, task.taskId);
}

function recordRejectedProposal(
  store: WorkflowStore,
  task: WorkflowTurnTask,
  proposal: WorkflowDecisionProposal,
  validationReportId: string,
  rejectionCode: string,
): void {
  store.db
    .prepare(
      `INSERT OR IGNORE INTO decision_proposals(
         proposal_id, run_id, task_id, attempt_id, expected_state_json,
         decision_input_hash, decision, proposal_json, validation_report_id,
         status, rejection_code, created_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'rejected', ?, ?)`,
    )
    .run(
      proposal.proposalId,
      task.runId,
      task.taskId,
      task.attemptId,
      canonicalJson(proposal.expectedState),
      proposal.decisionInputHash,
      proposal.decision,
      canonicalJson(proposal),
      validationReportId,
      rejectionCode,
      timestamp(),
    );
}

function timestamp(): string {
  return new Date().toISOString();
}
