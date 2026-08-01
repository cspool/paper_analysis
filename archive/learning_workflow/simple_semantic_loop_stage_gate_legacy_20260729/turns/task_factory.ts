import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type {
  Anchor,
  ClosureReviewTaskEnvelope,
  Direction,
  DirectionDuplicateProjection,
  DirectionReviewTaskEnvelope,
  EvidenceFinding,
  EvidencePacketEnvelope,
  EvidenceReaderTaskEnvelope,
  GateDefinition,
  KnowledgeDimension,
  ObjectRef,
  OutputCoverageProjection,
  RegisteredRole,
  ReviewDeltaEnvelope,
  SearchNeed,
  StageContract,
  TopicFrame,
  WorkflowCompletionProjection,
  WorkflowDecisionProposal,
  WorkflowPermissionEnvelope,
  WorkflowTurnTask,
} from "../contracts/index.ts";
import {
  DIMENSION_PATHS,
} from "../validators/domain_validator.ts";
import {
  ROLE_MESSAGE_TYPES,
  ROLE_SKILLS,
  RUBRIC_REGISTRY,
  TRIGGER_ALLOWED_ACTIONS,
  canonicalSha256,
} from "../contracts/index.ts";
import type { WorkflowStore } from "../db/workflow_store.ts";
import { loadSkillPackage } from "./skill_package.ts";
import {
  bindPayloadTaskHash,
  buildWorkflowTurnPrompt,
} from "./prompt_builder.ts";
import { buildWorkflowTurnTask } from "../workflow/snapshot_builder.ts";
import type { RegisteredTrigger } from "../workflow/trigger_engine.ts";

export interface TaskFactoryConfig {
  projectRoot: string;
  skillRoot: string;
  schemaManifestPath: string;
  skillVersion?: string;
}

interface SchemaManifest {
  protocolVersion: 1;
  schemas: Record<string, { path: string; sha256: string }>;
}

export class TurnTaskFactory {
  private readonly store: WorkflowStore;
  private readonly config: TaskFactoryConfig;
  private readonly manifest: SchemaManifest;
  private readonly manifestSha256: string;
  private readonly skillVersion: string;

  constructor(
    store: WorkflowStore,
    config: TaskFactoryConfig,
  ) {
    this.store = store;
    this.config = config;
    this.manifest = JSON.parse(
      readFileSync(config.schemaManifestPath, "utf8"),
    );
    if (this.manifest.protocolVersion !== 1) {
      throw new Error("unsupported schema manifest protocol");
    }
    this.manifestSha256 = canonicalSha256(this.manifest);
    this.skillVersion = config.skillVersion ?? "1.0.0";
  }

  schemaManifestSha256(): string {
    return this.manifestSha256;
  }

  schemaForMessage(messageType: string): unknown {
    const entry = this.manifest.schemas[messageType];
    if (!entry) throw new Error(`schema manifest has no ${messageType}`);
    const schemaPath = resolve(dirname(this.config.schemaManifestPath), entry.path);
    return JSON.parse(readFileSync(schemaPath, "utf8"));
  }

  skillPackage(role: RegisteredRole) {
    return loadSkillPackage(
      `${this.config.skillRoot}/${ROLE_SKILLS[role]}`,
    );
  }

  buildCompletionProjection(
    runId: string,
  ): WorkflowCompletionProjection | null {
    const topic = this.activeCanonical<TopicFrame>(runId, "topic")[0];
    if (!topic) return null;
    const anchors = this.activeCanonical<Anchor>(runId, "anchor");
    const directions = this.activeCanonical<Direction>(runId, "direction");
    const needs = this.activeCanonical<SearchNeed>(runId, "search_need");
    const taskIndex = this.taskWorkIndex(runId);
    const resultIndex = this.resultWorkIndex(runId);
    const deltaIndex = this.deltaWorkIndex(runId);
    const outputAttemptIndex = this.outputAttemptWorkIndex(runId);
    const validationFailureIndex = this.validationFailureWorkIndex(runId);
    const contradictions = this.activeCanonical<Record<string, unknown>>(
      runId,
      "contradiction",
    );
    const handoffs = this.activeCanonical<Record<string, unknown>>(
      runId,
      "experiment_handoff",
    );
    const expansion = this.latestActiveCanonical<Record<string, unknown>>(
      runId,
      "topic_expansion",
    );
    const noDeltaRefs = this.activeObjectRefsByType(runId, "no_delta");
    const state = this.store.readWorkflowState(runId);

    const anchorRefs: ObjectRef[] = anchors.map((anchor) => ({
      objectType: "anchor",
      objectId: anchor.anchorId,
      revision: anchor.revision,
    }));
    const directionRefs: ObjectRef[] = directions.map((direction) => ({
      objectType: "direction",
      objectId: direction.directionId,
      revision: direction.revision,
    }));
    const openNeedIds = needs
      .filter((need) => need.status === "pending")
      .map((need) => need.needId)
      .sort();
    const pendingTaskIds = idsWithStatus(taskIndex, "pending");
    const inFlightTaskIds = idsWithStatus(taskIndex, "in_flight");
    const pendingOutputRetryTaskIds = idsWithStatus(
      outputAttemptIndex,
      "pending_retry",
    );
    const unconsumedResultIds = idsWithStatus(resultIndex, "unconsumed");
    const uncommittedDeltaIds = idsWithStatus(deltaIndex, "uncommitted");
    const unresolvedValidationFailureIds = idsWithStatus(
      validationFailureIndex,
      "unresolved",
    );
    const failedTaskIds = idsWithStatus(taskIndex, "failed_unresolved");
    const unreviewedCriticalContradictionIds = contradictions
      .filter(
        (item) => Boolean(item.critical) && !item.dispositionReviewId,
      )
      .map((item) => String(item.contradictionId))
      .sort();
    const experimentHandoffIds = handoffs
      .map((handoff) => String(handoff.handoffId))
      .sort();
    const lastExpansionQuiet =
      Boolean(expansion) &&
      expansion!.completed === true &&
      expansion!.outcome === "no_new_anchor_no_critical_delta" &&
      typeof expansion!.noDeltaRecordId === "string" &&
      expansion!.semanticDeltaId === null &&
      noDeltaRefs.some(
        (ref) => ref.objectId === expansion!.noDeltaRecordId,
      );
    const handoffsComplete =
      directions.every((direction) => {
        const matching = handoffs.filter(
          (handoff) => handoff.directionId === direction.directionId,
        );
        if (direction.status !== "experiment_required") {
          return matching.length === 0;
        }
        return (
          matching.length === 1 &&
          matching[0]!.handoffId === direction.experimentHandoffId &&
          matching[0]!.executionAuthorized === false &&
          Boolean(matching[0]!.complete ?? true)
        );
      }) &&
      handoffs.every((handoff) =>
        directions.some(
          (direction) =>
            direction.directionId === handoff.directionId &&
            direction.status === "experiment_required",
        ),
      );
    const claims: WorkflowCompletionProjection["claims"] = {
      topicScopePreserved:
        topic.scopeAudit.initialFingerprint ===
          topic.scopeAudit.currentFingerprint ||
        topic.scopeAudit.changes.every((change) => change.userAuthorized),
      noKnowledgeAnswerableCriticalNeed: !needs.some(
        (need) =>
          need.status === "pending" &&
          need.critical &&
          ["knowledge_base", "unknown"].includes(need.answerability),
      ),
      allAnchorsClosed: anchors.every(
        (anchor) =>
          (anchor.status === "saturated" &&
            Boolean(anchor.saturationReason?.trim())) ||
          (anchor.status === "rejected" &&
            Boolean(anchor.statusReason.trim())),
      ),
      allDirectionsTerminal: directions.every(
        (direction) =>
          ["testable", "experiment_required", "rejected"].includes(
            direction.status,
          ) && Boolean(direction.statusReason.trim()),
      ),
      lastTopicExpansionNoDelta: lastExpansionQuiet,
      noUnconsumedOrUncommittedWork:
        pendingTaskIds.length === 0 &&
        inFlightTaskIds.length === 0 &&
        pendingOutputRetryTaskIds.length === 0 &&
        unconsumedResultIds.length === 0 &&
        uncommittedDeltaIds.length === 0 &&
        unresolvedValidationFailureIds.length === 0 &&
        failedTaskIds.length === 0,
      criticalContradictionsReviewed:
        unreviewedCriticalContradictionIds.length === 0,
      experimentHandoffsComplete: handoffsComplete,
      runtimeEligibleForCompletion:
        state.lifecycle === "running" &&
        state.budgetState.exhaustedDimensions.length === 0,
      finalOutputTraceable: true,
    };
    const blockingClaims = (
      Object.keys(claims) as Array<keyof typeof claims>
    ).filter((claim) => !claims[claim]);
    const coverageBasis = {
      canonicalRevision: state.canonicalRevision,
      topicFrameRevision: topic.revision,
      anchorRefs,
      directionRefs,
      openNeedIds,
      pendingTaskIds,
      inFlightTaskIds,
      pendingOutputRetryTaskIds,
      unconsumedResultIds,
      uncommittedDeltaIds,
      unresolvedValidationFailureIds,
      failedTaskIds,
      unreviewedCriticalContradictionIds,
      experimentHandoffIds,
      lastTopicExpansionNeedId:
        typeof expansion?.needId === "string" ? expansion.needId : null,
    };
    return {
      ...coverageBasis,
      outputCoverageProjectionId:
        `coverage-${canonicalSha256(coverageBasis).slice(0, 24)}`,
      claims,
      eligibleForProposal: blockingClaims.length === 0,
      blockingClaims,
    };
  }

  buildStageTask(
    runId: string,
    stage: {
      stageId: string;
      stageType: string;
      role: RegisteredRole;
      contract: StageContract;
      gate: GateDefinition;
    },
    taskId = `task-${randomUUID()}`,
    attemptId = `attempt-${randomUUID()}`,
  ):
    | EvidenceReaderTaskEnvelope
    | DirectionReviewTaskEnvelope
    | ClosureReviewTaskEnvelope {
    switch (stage.role) {
      case "evidence_reader":
        return this.buildEvidenceTask(runId, stage, taskId, attemptId);
      case "direction_reviewer":
        return this.buildDirectionTask(runId, stage, taskId, attemptId);
      case "closure_reviewer":
        return this.buildClosureTask(runId, stage, taskId, attemptId);
      case "workflow_decision":
        throw new Error("use buildWorkflowTask for Controller triggers");
    }
  }

  buildWorkflowTask(
    runId: string,
    trigger: RegisteredTrigger,
    meta: {
      stageId: string;
      contract: StageContract;
      permission: WorkflowPermissionEnvelope;
      domainProjection: WorkflowTurnTask["domainProjection"];
    },
    taskId = `task-${randomUUID()}`,
    attemptId = `attempt-${randomUUID()}`,
  ): WorkflowTurnTask {
    const skill = this.skillPackage("workflow_decision");
    const task = buildWorkflowTurnTask(this.store, runId, {
      taskId,
      attemptId,
      stageId: meta.stageId,
      stageContractHash: meta.contract.sha256,
      skill: {
        name: ROLE_SKILLS.workflow_decision,
        version: this.skillVersion,
        sha256: skill.sha256,
      },
      schema: this.schemaBinding("WORKFLOW_DECISION_PROPOSAL"),
      permission: meta.permission,
      domainProjection: meta.domainProjection,
      approvedArtifacts: [],
      trigger,
    });
    return buildWorkflowTurnPrompt({
      task,
      skillMarkdown: skill.skillMarkdown,
      expectedSchema: this.schemaForMessage("WORKFLOW_DECISION_PROPOSAL"),
    }).task;
  }

  private buildEvidenceTask(
    runId: string,
    stage: {
      stageId: string;
      role: RegisteredRole;
      contract: StageContract;
    },
    taskId: string,
    attemptId: string,
  ): EvidenceReaderTaskEnvelope {
    const proposal = this.proposalForStage(runId, stage.contract);
    const need =
      proposal.domainProposal?.kind === "search_need"
        ? proposal.domainProposal.value
        : this.searchNeedFromStageScope(runId, stage.contract);
    const topic = this.requireCanonical<TopicFrame>(runId, "topic", need.owner.topicId);
    const anchor = need.owner.anchorId
      ? this.requireCanonical<Anchor>(runId, "anchor", need.owner.anchorId)
      : null;
    const direction = need.owner.directionId
      ? this.requireCanonical<Direction>(
          runId,
          "direction",
          need.owner.directionId,
        )
      : null;
    const skill = this.skillPackage("evidence_reader");
    const task: EvidenceReaderTaskEnvelope = {
      protocolVersion: 1,
      messageType: "EVIDENCE_READER_TASK",
      workflowId: this.store.getRun(runId).workflowId,
      runId,
      taskId,
      attemptId,
      stageId: stage.stageId,
      stageContractHash: stage.contract.sha256,
      stateBinding: this.store.stateBinding(runId),
      inputHash: "",
      payload: {
        searchNeed: need,
        focus: { topic, anchor, direction },
        previousQueries: [],
        previousReads: [],
        consumedSourceUnitIds: need.excludedSourceUnits,
        allowedVaultRoots: need.targetDimensions.map((dimension) => ({
          dimension,
          relativePathPrefix: DIMENSION_PATHS[dimension],
        })),
        budget: stage.contract.budget,
        skill: {
          name: ROLE_SKILLS.evidence_reader,
          version: this.skillVersion,
          sha256: skill.sha256,
        },
        schema: this.schemaBinding("EVIDENCE_PACKET"),
        permission: {
          role: "evidence_reader",
          tools: [...stage.contract.requestedTools],
          allowedPathPrefixes: [...stage.contract.requestedPaths],
          filesystem: "vault_read_only",
          network: false,
          delegation: false,
          goals: false,
          stateWrite: false,
          experimentExecution: false,
          maxBudget: stage.contract.budget,
        },
        correctionFeedback: null,
        terminationCondition:
          "Emit exactly one EVIDENCE_PACKET JSON value, then terminate.",
      },
    };
    return bindPayloadTaskHash(task);
  }

  private searchNeedFromStageScope(
    runId: string,
    contract: StageContract,
  ): SearchNeed {
    const refs = contract.scope.filter(
      (ref) => ref.objectType === "search_need",
    );
    if (refs.length !== 1) {
      throw new Error(
        "EVIDENCE_READ Stage must scope exactly one existing SearchNeed",
      );
    }
    const ref = refs[0]!;
    const need = this.requireCanonical<SearchNeed>(
      runId,
      "search_need",
      ref.objectId,
    );
    if (need.revision !== ref.revision || need.status !== "pending") {
      throw new Error(
        "EVIDENCE_READ scoped SearchNeed is stale or non-pending",
      );
    }
    return need;
  }

  private buildDirectionTask(
    runId: string,
    stage: {
      stageId: string;
      role: RegisteredRole;
      contract: StageContract;
    },
    taskId: string,
    attemptId: string,
  ): DirectionReviewTaskEnvelope {
    const proposal = this.proposalForStage(runId, stage.contract);
    if (proposal.domainProposal?.kind !== "direction_review_request") {
      throw new Error("DIRECTION_REVIEW Stage has no frozen review request");
    }
    const request = proposal.domainProposal.value;
    const direction = this.requireCanonical<Direction>(
      runId,
      "direction",
      request.directionRef.objectId,
    );
    const anchor = this.requireCanonical<Anchor>(
      runId,
      "anchor",
      direction.anchorId,
    );
    const topic = this.requireCanonical<TopicFrame>(
      runId,
      "topic",
      anchor.topicId,
    );
    const evidence = this.committedEvidence(runId);
    const evidenceById = new Map(
      evidence.map((finding) => [finding.evidenceId, finding]),
    );
    const citedIds = new Set([
      ...direction.supportingEvidenceRefs,
      ...direction.contradictingEvidenceRefs,
    ]);
    const cited = [...citedIds]
      .map((id) => evidenceById.get(id))
      .filter((item): item is EvidenceFinding => Boolean(item));
    const siblings = this.activeCanonical<Direction>(runId, "direction")
      .filter(
        (candidate) =>
          candidate.anchorId === direction.anchorId &&
          candidate.directionId !== direction.directionId,
      )
      .map(directionProjection);
    const skill = this.skillPackage("direction_reviewer");
    const task: DirectionReviewTaskEnvelope = {
      protocolVersion: 1,
      messageType: "DIRECTION_REVIEW_TASK",
      workflowId: this.store.getRun(runId).workflowId,
      runId,
      taskId,
      attemptId,
      stageId: stage.stageId,
      stageContractHash: stage.contract.sha256,
      stateBinding: this.store.stateBinding(runId),
      inputHash: "",
      payload: {
        topic,
        anchor,
        direction,
        siblingDirections: siblings,
        evidenceFindings: cited,
        contradictingEvidence: direction.contradictingEvidenceRefs
          .map((id) => evidenceById.get(id))
          .filter((item): item is EvidenceFinding => Boolean(item)),
        unresolvedSearchNeeds: this.activeCanonical<SearchNeed>(
          runId,
          "search_need",
        ).filter(
          (need) =>
            need.owner.directionId === direction.directionId &&
            need.status === "pending",
        ),
        counterexamples: direction.degradationConditions.map(
          (condition, index) => ({
            counterexampleId: `degradation-${index + 1}`,
            statement: condition,
            evidenceRefs: direction.contradictingEvidenceRefs,
            degradationCondition: condition,
          }),
        ),
        priorReviews: this.committedReviewEnvelopes(runId)
          .filter(
            (review) =>
              review.payload.directionId === direction.directionId,
          )
          .map((review) => ({
            reviewId: review.payload.reviewId,
            directionRevision: review.payload.directionRevision,
            decision: review.payload.decision,
          })),
        reviewPurpose: request.purpose,
        rubric: request.rubric,
        allowedEvidenceIds: cited.map((finding) => finding.evidenceId),
        allowedObjectRefs: [
          request.directionRef,
          ...siblings.map((sibling) => sibling.directionRef),
        ],
        inputArtifacts: [],
        budget: stage.contract.budget,
        skill: {
          name: ROLE_SKILLS.direction_reviewer,
          version: this.skillVersion,
          sha256: skill.sha256,
        },
        schema: this.schemaBinding("REVIEW_DELTA"),
        permission: {
          role: "direction_reviewer",
          tools: [],
          allowedPathPrefixes: [],
          filesystem: "none",
          network: false,
          delegation: false,
          goals: false,
          stateWrite: false,
          experimentExecution: false,
          maxBudget: stage.contract.budget,
        },
        correctionFeedback: null,
        terminationCondition:
          "Emit exactly one REVIEW_DELTA JSON value, then terminate.",
      },
    };
    return bindPayloadTaskHash(task);
  }

  private buildClosureTask(
    runId: string,
    stage: {
      stageId: string;
      role: RegisteredRole;
      contract: StageContract;
    },
    taskId: string,
    attemptId: string,
  ): ClosureReviewTaskEnvelope {
    const candidate = this.latestActiveCanonical<Record<string, unknown>>(
      runId,
      "stop_candidate",
    );
    const proof = this.latestActiveCanonical<Record<string, unknown>>(
      runId,
      "stop_proof",
    );
    if (!candidate || !proof) {
      throw new Error("closure path requires active StopCandidate and StopProof");
    }
    const topic = this.activeCanonical<TopicFrame>(runId, "topic")[0];
    if (!topic) throw new Error("closure path requires TopicFrame");
    const anchors = this.activeCanonical<Anchor>(runId, "anchor");
    const directions = this.activeCanonical<Direction>(runId, "direction");
    const needs = this.activeCanonical<SearchNeed>(runId, "search_need");
    const expansion = this.latestActiveCanonical<Record<string, unknown>>(
      runId,
      "topic_expansion",
    ) as ClosureReviewTaskEnvelope["payload"]["lastTopicExpansion"];
    const taskIndex = this.taskWorkIndex(runId);
    const resultIndex = this.resultWorkIndex(runId);
    const deltaIndex = this.deltaWorkIndex(runId);
    const outputAttemptIndex = this.outputAttemptWorkIndex(runId);
    const validationFailureIndex = this.validationFailureWorkIndex(runId);
    const contradictions = this.activeCanonical<Record<string, unknown>>(
      runId,
      "contradiction",
    ).map((item) => ({
      contradictionId: String(item.contradictionId),
      critical: Boolean(item.critical),
      dispositionReviewId: item.dispositionReviewId
        ? String(item.dispositionReviewId)
        : null,
      objectRef: item.objectRef as never,
    }));
    const experimentHandoffs = this.activeCanonical<
      Record<string, unknown>
    >(runId, "experiment_handoff").map((item) => ({
      handoffId: String(item.handoffId),
      directionId: String(item.directionId),
      complete: Boolean(item.complete ?? true),
      executionAuthorized: false as const,
    }));
    const outputCoverage = this.buildOutputCoverage(
      runId,
      String(proof.outputCoverageProjectionId),
      topic,
      anchors,
      directions,
      needs,
      contradictions,
      experimentHandoffs,
    );
    const skill = this.skillPackage("closure_reviewer");
    const runState = this.store.readWorkflowState(runId);
    const task: ClosureReviewTaskEnvelope = {
      protocolVersion: 1,
      messageType: "CLOSURE_REVIEW_TASK",
      workflowId: this.store.getRun(runId).workflowId,
      runId,
      taskId,
      attemptId,
      stageId: stage.stageId,
      stageContractHash: stage.contract.sha256,
      stateBinding: this.store.stateBinding(runId),
      inputHash: "",
      payload: {
        stopCandidateBundle: {
          candidate: candidate as never,
          proof: proof as never,
        },
        currentCanonicalRevision: runState.canonicalRevision,
        topic,
        anchors: anchors.map((anchor) => ({
          anchorRef: {
            objectType: "anchor",
            objectId: anchor.anchorId,
            revision: anchor.revision,
          },
          status: anchor.status,
          statusReason: anchor.statusReason,
          saturationReason: anchor.saturationReason,
        })),
        directions: directions.map((direction) => ({
          directionRef: {
            objectType: "direction",
            objectId: direction.directionId,
            revision: direction.revision,
          },
          anchorId: direction.anchorId,
          status: direction.status,
          statusReason: direction.statusReason,
          experimentHandoffId: direction.experimentHandoffId,
        })),
        needs: needs.map((need) => ({
          needId: need.needId,
          revision: need.revision,
          owner: need.owner,
          critical: need.critical,
          answerability: need.answerability,
          status: need.status,
        })),
        taskIndex,
        resultIndex,
        deltaIndex,
        outputAttemptIndex,
        validationFailureIndex,
        recentSemanticRecords: this.activeObjectRefsByType(
          runId,
          "semantic_delta",
        ),
        recentNoDeltaRecords: this.activeObjectRefsByType(runId, "no_delta"),
        lastTopicExpansion: expansion ?? null,
        contradictions,
        experimentHandoffs,
        mechanicalPreflight: {
          preflightId: `preflight-placeholder-${randomUUID()}`,
          stopCandidateId: String(candidate.stopCandidateId),
          canonicalRevision: runState.canonicalRevision,
          checks: [{ checkId: "projection_built", passed: true, issueIds: [] }],
          passed: true,
        },
        rubric: RUBRIC_REGISTRY.closure_v1,
        budgetState: runState.budgetState,
        lifecycle: runState.lifecycle,
        runtimeEligibility: {
          budgetExhausted:
            runState.budgetState.exhaustedDimensions.length > 0,
          paused: runState.lifecycle.startsWith("paused"),
          blocked: runState.lifecycle.startsWith("blocked"),
          failed: runState.lifecycle.startsWith("failed"),
          reason: runState.pauseOrBlockReason,
        },
        outputCoverage,
        freshTurn: true,
        providerHistoryIncluded: false,
        canonicalOnly: true,
        budget: stage.contract.budget,
        skill: {
          name: ROLE_SKILLS.closure_reviewer,
          version: this.skillVersion,
          sha256: skill.sha256,
        },
        schema: this.schemaBinding("CLOSURE_REVIEW"),
        permission: {
          role: "closure_reviewer",
          tools: [],
          allowedPathPrefixes: [],
          filesystem: "none",
          network: false,
          delegation: false,
          goals: false,
          stateWrite: false,
          experimentExecution: false,
          maxBudget: stage.contract.budget,
        },
        correctionFeedback: null,
        terminationCondition:
          "Emit exactly one CLOSURE_REVIEW JSON value, then terminate.",
      },
    };
    return bindPayloadTaskHash(task);
  }

  private schemaBinding(messageType: string) {
    const entry = this.manifest.schemas[messageType];
    if (!entry) throw new Error(`schema manifest has no ${messageType}`);
    return {
      manifestSha256: this.manifestSha256,
      expectedOutputMessageType: messageType as never,
      expectedOutputSchemaSha256: entry.sha256,
    };
  }

  private proposalForStage(
    runId: string,
    contract: StageContract,
  ): WorkflowDecisionProposal {
    const rows = this.store.query(
      `SELECT proposal_json FROM decision_proposals
       WHERE run_id = ? AND status = 'accepted' ORDER BY created_at DESC`,
      runId,
    );
    for (const row of rows) {
      const proposal = JSON.parse(
        String(row.proposal_json),
      ) as WorkflowDecisionProposal;
      if (
        proposal.proposedStageContract?.proposalLocalStageKey ===
        contract.proposalLocalStageKey
      ) {
        return proposal;
      }
      if (
        proposal.proposedPlanPatch?.operations.some(
          (operation) =>
            operation.op === "add_stage" &&
            operation.stage.proposalLocalStageKey ===
              contract.proposalLocalStageKey,
        )
      ) {
        return proposal;
      }
    }
    throw new Error(
      `no accepted proposal owns Stage key ${contract.proposalLocalStageKey}`,
    );
  }

  private requireCanonical<T>(
    runId: string,
    objectType: string,
    objectId: string,
  ): T {
    const row = this.store.db
      .prepare(
        `SELECT object_json FROM canonical_objects
         WHERE run_id = ? AND object_type = ? AND object_id = ? AND active = 1`,
      )
      .get(runId, objectType, objectId) as { object_json: string } | undefined;
    if (!row) throw new Error(`missing canonical ${objectType}/${objectId}`);
    return JSON.parse(row.object_json) as T;
  }

  private activeCanonical<T>(runId: string, objectType: string): T[] {
    return this.store
      .query(
        `SELECT object_json FROM canonical_objects
         WHERE run_id = ? AND object_type = ? AND active = 1 ORDER BY object_id`,
        runId,
        objectType,
      )
      .map((row) => JSON.parse(String(row.object_json)) as T);
  }

  private latestActiveCanonical<T>(
    runId: string,
    objectType: string,
  ): T | undefined {
    const row = this.store.db
      .prepare(
        `SELECT object_json FROM canonical_objects
         WHERE run_id = ? AND object_type = ? AND active = 1
         ORDER BY created_at DESC, rowid DESC LIMIT 1`,
      )
      .get(runId, objectType) as { object_json: string } | undefined;
    return row ? (JSON.parse(row.object_json) as T) : undefined;
  }

  private committedEvidence(runId: string): EvidenceFinding[] {
    return this.store
      .query(
        `SELECT payload_json FROM turn_results
         WHERE run_id = ? AND message_type = 'EVIDENCE_PACKET'
         AND status = 'committed' ORDER BY result_id`,
        runId,
      )
      .flatMap((row) => {
        const envelope = JSON.parse(
          String(row.payload_json),
        ) as EvidencePacketEnvelope;
        return envelope.payload.findings;
      });
  }

  private committedReviewEnvelopes(runId: string): ReviewDeltaEnvelope[] {
    return this.store
      .query(
        `SELECT payload_json FROM turn_results
         WHERE run_id = ? AND message_type = 'REVIEW_DELTA'
         AND status = 'committed' ORDER BY result_id`,
        runId,
      )
      .map(
        (row) =>
          JSON.parse(String(row.payload_json)) as ReviewDeltaEnvelope,
      );
  }

  private taskWorkIndex(runId: string) {
    return this.store
      .query(
        `SELECT task_id AS id, status FROM tasks
         WHERE run_id = ? ORDER BY task_id`,
        runId,
      )
      .map((row) => ({
        id: String(row.id),
        status:
          row.status === "pending"
            ? "pending"
            : ["dispatched", "running"].includes(String(row.status))
              ? "in_flight"
              : row.status === "failed"
                ? "failed_unresolved"
                : String(row.status),
        resolvedById: null,
        objectRef: null,
      }));
  }

  private resultWorkIndex(runId: string) {
    return this.store
      .query(
        `SELECT r.result_id AS id, r.status, c.commit_id
         FROM turn_results r
         LEFT JOIN result_consumptions c ON c.result_id = r.result_id
         WHERE r.run_id = ? ORDER BY r.result_id`,
        runId,
      )
      .map((row) => ({
        id: String(row.id),
        status: row.commit_id ? "consumed" : "unconsumed",
        resolvedById: row.commit_id ? String(row.commit_id) : null,
        objectRef: {
          objectType: "turn_result",
          objectId: String(row.id),
          revision: 1,
        },
      }));
  }

  private deltaWorkIndex(runId: string) {
    return this.store
      .query(
        `SELECT proposal_json FROM decision_proposals
         WHERE run_id = ? AND status = 'accepted' ORDER BY proposal_id`,
        runId,
      )
      .map((row) => JSON.parse(String(row.proposal_json)) as WorkflowDecisionProposal)
      .filter(
        (
          proposal,
        ): proposal is WorkflowDecisionProposal & {
          domainProposal: Extract<
            NonNullable<WorkflowDecisionProposal["domainProposal"]>,
            { kind: "semantic_delta" }
          >;
        } => proposal.domainProposal?.kind === "semantic_delta",
      )
      .map((proposal) => {
        const deltaId = proposal.domainProposal.value.deltaId;
        const consumption = this.store.db
          .prepare(
            `SELECT commit_id FROM result_consumptions
             WHERE run_id = ? AND delta_id = ? LIMIT 1`,
          )
          .get(runId, deltaId) as { commit_id: string } | undefined;
        return {
          id: deltaId,
          status: consumption ? "committed" : "uncommitted",
          resolvedById: consumption?.commit_id ?? null,
          objectRef: {
            objectType: "semantic_delta",
            objectId: deltaId,
            revision: 1,
          },
        };
      });
  }

  private outputAttemptWorkIndex(runId: string) {
    return this.store
      .query(
        `SELECT task_id FROM tasks
         WHERE run_id = ? AND status = 'pending_output_retry'
         ORDER BY task_id`,
        runId,
      )
      .map((row) => ({
        id: String(row.task_id),
        status: "pending_retry",
        resolvedById: null,
        objectRef: null,
      }));
  }

  private validationFailureWorkIndex(runId: string) {
    return this.store
      .query(
        `SELECT validation_report_id, valid, resolved_by_id
         FROM validation_reports WHERE run_id = ?
         ORDER BY validation_report_id`,
        runId,
      )
      .map((row) => ({
        id: String(row.validation_report_id),
        status:
          Number(row.valid) === 0 && !row.resolved_by_id
            ? "unresolved"
            : "resolved",
        resolvedById: row.resolved_by_id
          ? String(row.resolved_by_id)
          : null,
        objectRef: null,
      }));
  }

  private activeObjectRefsByType(
    runId: string,
    objectType: string,
  ): ObjectRef[] {
    return this.store
      .query(
        `SELECT object_id, revision FROM canonical_objects
         WHERE run_id = ? AND object_type = ? AND active = 1
         ORDER BY object_id`,
        runId,
        objectType,
      )
      .map((row) => ({
        objectType,
        objectId: String(row.object_id),
        revision: Number(row.revision),
      }));
  }

  private buildOutputCoverage(
    runId: string,
    projectionId: string,
    topic: TopicFrame,
    anchors: Anchor[],
    directions: Direction[],
    needs: SearchNeed[],
    contradictions: ClosureReviewTaskEnvelope["payload"]["contradictions"],
    handoffs: ClosureReviewTaskEnvelope["payload"]["experimentHandoffs"],
  ): OutputCoverageProjection {
    const topicRef: ObjectRef = {
      objectType: "topic",
      objectId: topic.topicId,
      revision: topic.revision,
    };
    const anchorRefs = anchors.map((anchor) => ({
      objectType: "anchor",
      objectId: anchor.anchorId,
      revision: anchor.revision,
    }));
    const directionRefs = directions.map((direction) => ({
      objectType: "direction",
      objectId: direction.directionId,
      revision: direction.revision,
    }));
    const evidenceRefs = this.store
      .query(
        `SELECT result_id FROM turn_results
         WHERE run_id = ? AND message_type = 'EVIDENCE_PACKET'
         AND status = 'committed' ORDER BY result_id`,
        runId,
      )
      .map((row) => ({
        objectType: "turn_result",
        objectId: String(row.result_id),
        revision: 1,
      }));
    const contradictionRefs = contradictions.map((item) => item.objectRef);
    const handoffRefs = handoffs.map((item) => ({
      objectType: "experiment_handoff",
      objectId: item.handoffId,
      revision: 1,
    }));
    const needRefs = needs
      .filter((need) => need.status === "pending")
      .map((need) => ({
        objectType: "search_need",
        objectId: need.needId,
        revision: need.revision,
      }));
    return {
      projectionId,
      fields: {
        topic_scope: [topicRef],
        anchor_summaries: anchorRefs.length ? anchorRefs : [topicRef],
        direction_statuses: directionRefs.length ? directionRefs : [topicRef],
        evidence_provenance: evidenceRefs.length ? evidenceRefs : [topicRef],
        contradictions_and_limits: contradictionRefs.length
          ? contradictionRefs
          : [topicRef],
        experiment_handoffs: handoffRefs.length ? handoffRefs : [topicRef],
        unresolved_questions: needRefs.length ? needRefs : [topicRef],
      },
    };
  }
}

export function defaultWorkflowPermission(
  trigger: RegisteredTrigger,
  budgets: {
    workflow: StageContract["budget"];
    evidence: StageContract["budget"];
    direction: StageContract["budget"];
  },
): WorkflowPermissionEnvelope {
  return {
    allowedActions: [...TRIGGER_ALLOWED_ACTIONS[trigger.trigger]],
    allowedStageTypes: [
      "SCRIPT_APPLY_TOPIC_FRAME",
      "SCRIPT_APPLY_SEMANTIC_DELTA",
      "EVIDENCE_READ",
      "DIRECTION_REVIEW",
    ],
    allowedRoles: ["evidence_reader", "direction_reviewer"],
    allowedTools: [
      "mcp__obsidian__obsidian_search_notes",
      "mcp__obsidian__obsidian_get_note",
    ],
    allowedPathPrefixes: Object.values(DIMENSION_PATHS),
    registeredRubrics: [RUBRIC_REGISTRY.direction_readiness_v1.rubricId],
    maxBudgetByRole: {
      workflow_decision: budgets.workflow,
      evidence_reader: budgets.evidence,
      direction_reviewer: budgets.direction,
    },
    suppliedObjectRefs: [],
    suppliedArtifactIds: [],
    suppliedResultRefs: [],
  };
}

function directionProjection(direction: Direction): DirectionDuplicateProjection {
  return {
    directionRef: {
      objectType: "direction",
      objectId: direction.directionId,
      revision: direction.revision,
    },
    baseline: direction.comparison.baseline,
    comparisonScope: [],
    controlledVariables: direction.comparison.controlledVariables,
    primaryChanges: direction.changes
      .filter((change) => change.role === "primary")
      .map((change) => ({
        layer: change.layer,
        object: change.object,
        fromState: change.fromState,
        toState: change.toState,
        conditions: change.conditions,
      })),
    causalTargets: direction.causalLinks.map((link) => link.to),
    hypothesis: direction.hypothesis,
  };
}

function idsWithStatus(
  items: Array<{ id: string; status: string }>,
  status: string,
): string[] {
  return items
    .filter((item) => item.status === status)
    .map((item) => item.id)
    .sort();
}
