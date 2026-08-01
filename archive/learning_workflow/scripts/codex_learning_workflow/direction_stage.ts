import fs from "node:fs";
import path from "node:path";

import { atomicWriteJson, CanonicalStore } from "./canonical_store.ts";
import {
  buildExpertReview,
  reviewDimension,
  validateDirectionProposal,
} from "./domain_validators.ts";
import { REVIEW_DIMENSIONS, REVIEW_REFERENCE_KEYS } from "./role_profiles.ts";
import { validateClaimCandidates } from "./source_validator.ts";
import { mapWithConcurrency } from "./task_scheduler.ts";
import type {
  AgentHandle,
  Direction,
  ExpertReview,
  JsonValue,
  ReviewQuestionAnswer,
  RunState,
  TaskRecord,
} from "./types.ts";
import { RuntimeManager } from "./runtime_manager.ts";

interface ReviewProgress {
  questionAnswers: ReviewQuestionAnswer[];
  referenceKeysUsed: string[];
  lastReviewerOutput: string | null;
  lastResult: JsonValue | null;
}

function asObject(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? [...new Set(value.map((item) => String(item).trim()).filter(Boolean))]
    : [];
}

function anchorBundle(state: RunState, anchorId: string): JsonValue {
  const anchor = state.anchors.find((item) => item.anchorId === anchorId);
  const baselines = state.baselines.filter((baseline) => baseline.anchorId === anchorId);
  const entries = state.entries.filter((entry) => entry.anchorId === anchorId);
  const edges = state.edges.filter((edge) => edge.anchorId === anchorId);
  const claimIds = new Set([
    ...(anchor?.evidenceRefs ?? []),
    ...baselines.flatMap((baseline) => baseline.evidenceRefs),
    ...entries.flatMap((entry) => entry.evidenceRefs),
    ...edges.flatMap((edge) => edge.evidenceRefs),
  ]);
  return {
    anchor,
    baselines,
    entries,
    edges,
    evidence_claims: state.claims.filter((claim) => claimIds.has(claim.claimId)),
  } as unknown as JsonValue;
}

function plannerPrompt(
  state: RunState,
  anchorId: string,
  lastOutput: string | null,
  commitResult: JsonValue | null,
): string {
  const directions = state.directions
    .filter((direction) => direction.anchorId === anchorId && direction.status === "accepted")
    .map((direction) => ({
      direction_id: direction.directionId,
      selected_entry_ids: direction.selectedEntryIds,
      selected_edge_ids: direction.selectedEdgeIds,
      hypothesis: direction.hypothesis,
    }));
  return [
    "对单个 Anchor 的候选 Entry Graph 做一次 Direction Planner 决策。",
    "只提出一个显著不同的兼容子图，或声明该 Anchor 已无新方向。",
    `ANCHOR_CONTEXT:\n${JSON.stringify(anchorBundle(state, anchorId))}`,
    `ACCEPTED_DIRECTIONS:\n${JSON.stringify(directions)}`,
    `PREVIOUS_PLANNER_OUTPUT:\n${lastOutput ?? "<none>"}`,
    `DIRECTION_COMMIT_RESULT:\n${commitResult === null ? "<none>" : JSON.stringify(commitResult)}`,
    [
      "Direction 可只含一层，不要求覆盖 L1-L6。",
      "多 entry Direction 必须由 selected_edge_ids 构成连通子图。",
      "conflict/substitutes/incompatible entry 不能被当作协同组合。",
      "conditional edge 必须把条件写入 hypothesis/ablation。",
      "payload 必须给出 selected_entry_ids,selected_edge_ids,baseline_ids,hypothesis,ablation_plan[],implementation_plan[]。",
      "hypothesis 必须能用 Anchor 指标证伪，并区分单 entry 与组合增益。",
    ].join("\n"),
  ].join("\n\n");
}

function plannerCompletionPrompt(
  state: RunState,
  anchorId: string,
  commitResult: JsonValue | null,
): string {
  const acceptedDirectionIds = state.directions
    .filter((direction) => direction.anchorId === anchorId && direction.status === "accepted")
    .map((direction) => direction.directionId)
    .sort();
  return [
    "脚本已按确定性的 per-Anchor Direction 上限结束本 Anchor 的规划。",
    "不得再提出 Direction。只输出 DIRECTION_PLANNING_COMPLETE，并以 §TERMINATED 结束。",
    `ANCHOR_ID: ${anchorId}`,
    "SCRIPT_STOP_REASON: direction_cap_reached",
    "PROTOCOL_COMPLETION_REASON: budget_exhausted",
    `ACCEPTED_DIRECTION_IDS:\n${JSON.stringify(acceptedDirectionIds)}`,
    `LAST_DIRECTION_COMMIT_RESULT:\n${commitResult === null ? "<none>" : JSON.stringify(commitResult)}`,
  ].join("\n\n");
}

async function finalizePlannerAtDirectionCap(
  runtime: RuntimeManager,
  state: RunState,
  planner: AgentHandle,
  anchorId: string,
  commitResult: JsonValue | null,
): Promise<AgentHandle> {
  const completion = await runtime.runProtocolTurn(
    planner,
    plannerCompletionPrompt(state, anchorId, commitResult),
  );
  if (completion.protocol.marker !== "DIRECTION_PLANNING_COMPLETE") {
    throw new Error(
      `planner emitted ${completion.protocol.marker} after the script reached the Direction cap`,
    );
  }
  const completedAnchorId = completion.protocol.control.anchor_id
    ?? completion.protocol.control.anchorId;
  if (completedAnchorId !== anchorId) {
    throw new Error(
      `planner completion anchor mismatch: expected ${anchorId}, got ${completedAnchorId || "<missing>"}`,
    );
  }
  return completion.handle;
}

async function planAnchor(
  runtime: RuntimeManager,
  store: CanonicalStore,
  state: RunState,
  anchorId: string,
): Promise<void> {
  const progress = state.stage2.anchorPlanning[anchorId] ?? {
    status: "pending" as const,
    reason: null,
    turns: 0,
  };
  state.stage2.anchorPlanning[anchorId] = progress;
  if (progress.status === "complete" || progress.status === "pending_budget") return;
  progress.status = "running";
  let planner = await runtime.persistentAgent("direction_planner", anchorId);
  let lastOutput: string | null = null;
  let commitResult: JsonValue | null = null;

  while (progress.turns < state.config.maxPlannerTurnsPerAnchor) {
    const acceptedCount = state.directions.filter(
      (direction) => direction.anchorId === anchorId && direction.status === "accepted",
    ).length;
    if (acceptedCount >= state.config.maxDirectionsPerAnchor) {
      try {
        planner = await finalizePlannerAtDirectionCap(
          runtime,
          state,
          planner,
          anchorId,
          commitResult,
        );
        progress.turns += 1;
        progress.status = "complete";
        progress.reason = "direction_cap_reached";
        store.save(state, `direction_plan_${anchorId}_complete`);
      } catch (error) {
        progress.status = "pending_budget";
        progress.reason = `planner_finalization_error: ${error instanceof Error ? error.message : String(error)}`;
        store.save(state, `direction_plan_${anchorId}_finalization_error`);
      }
      return;
    }
    if (state.usage.turns >= state.config.maxTotalTurns) {
      progress.status = "pending_budget";
      progress.reason = "run_turn_budget_exhausted";
      store.save(state, `direction_plan_${anchorId}_pending`);
      return;
    }
    try {
      const decision = await runtime.runProtocolTurn(
        planner,
        plannerPrompt(state, anchorId, lastOutput, commitResult),
      );
      planner = decision.handle;
      progress.turns += 1;
      lastOutput = decision.protocol.rawText;
      if (decision.protocol.marker === "DIRECTION_PLANNING_COMPLETE") {
        progress.status = "complete";
        progress.reason = decision.protocol.control.reason || "exhausted_distinct_subgraphs";
        store.save(state, `direction_plan_${anchorId}_complete`);
        return;
      }
      const proposalIndex = state.directions.filter((direction) => direction.anchorId === anchorId).length + 1;
      const validated = validateDirectionProposal(
        state,
        anchorId,
        decision.protocol.payload!,
        proposalIndex,
      );
      if (validated.direction) {
        state.directions.push(validated.direction);
        state.stage2.directionReview[validated.direction.directionId] = {
          status: "pending",
          reason: null,
          rounds: 0,
        };
        commitResult = {
          accepted: true,
          direction_id: validated.direction.directionId,
          proposal_index: proposalIndex,
        };
      } else {
        commitResult = {
          accepted: false,
          proposal_index: proposalIndex,
          errors: validated.errors,
        };
      }
      store.recordEvent("direction_commit_result", {
        anchor_id: anchorId,
        ...commitResult as Record<string, JsonValue>,
      });
      store.save(state, `direction_plan_${anchorId}_turn_${progress.turns}`);
    } catch (error) {
      progress.status = "pending_budget";
      progress.reason = `planner_error: ${error instanceof Error ? error.message : String(error)}`;
      store.save(state, `direction_plan_${anchorId}_error`);
      return;
    }
  }
  progress.status = "pending_budget";
  progress.reason = "planner_turn_budget_exhausted";
  store.save(state, `direction_plan_${anchorId}_pending`);
}

function directionBundle(state: RunState, direction: Direction): JsonValue {
  const anchor = state.anchors.find((item) => item.anchorId === direction.anchorId);
  const entries = state.entries.filter((entry) => direction.selectedEntryIds.includes(entry.entryId));
  const edges = state.edges.filter((edge) => direction.selectedEdgeIds.includes(edge.edgeId));
  const baselines = state.baselines.filter((baseline) => direction.baselineIds.includes(baseline.baselineId));
  const claimIds = new Set(direction.evidenceRefs);
  return {
    direction,
    anchor,
    selected_entries: entries,
    selected_edges: edges,
    baselines,
    evidence_claims: state.claims.filter((claim) => claimIds.has(claim.claimId)),
    experiment_bundle: {
      hypothesis: direction.hypothesis,
      implementation_plan: direction.implementationPlan,
      ablation_plan: direction.ablationPlan,
      metrics: direction.metrics,
    },
  } as unknown as JsonValue;
}

function reviewerPrompt(
  state: RunState,
  direction: Direction,
  progress: ReviewProgress,
): string {
  const coverage = Object.fromEntries(
    REVIEW_DIMENSIONS.map((dimension) => [
      dimension,
      progress.questionAnswers.filter((qa) => qa.dimension === dimension).map((qa) => ({
        question_id: qa.questionId,
        conclusion: qa.conclusion,
        evidence_refs: qa.evidenceRefs,
        gaps: qa.gaps,
      })),
    ]),
  );
  return [
    "推进这个 Direction 的一次专家审阅决策。每轮只能 ask、request_reference 或 complete 三选一。",
    `DIRECTION_EXPERIMENT_BUNDLE:\n${JSON.stringify(directionBundle(state, direction))}`,
    `REVIEW_DIMENSION_COVERAGE:\n${JSON.stringify(coverage)}`,
    `FULL_QA_LEDGER:\n${JSON.stringify(progress.questionAnswers)}`,
    `REFERENCE_KEYS_ALREADY_USED:\n${JSON.stringify(progress.referenceKeysUsed)}`,
    `PREVIOUS_REVIEWER_OUTPUT:\n${progress.lastReviewerOutput ?? "<none>"}`,
    `NEW_RESULT_FROM_SCRIPT:\n${progress.lastResult === null ? "<none>" : JSON.stringify(progress.lastResult)}`,
    [
      "必须覆盖 scenario_opportunity, baseline_fairness, entry_validity, cross_layer_validity, implementation_reuse, experiment_measurement。",
      "ask payload：question_id,dimension,question,evidence_need；一个问题只绑定一个主 dimension。",
      "request_reference payload：reference_key,purpose；只可选择白名单并且每类最多一次。",
      "complete 之前，每个 dimension 必须有 evidence answer 或明确 unknown/not_applicable gap。",
      "最终价值顺序固定为 exploration > implementation reuse > method reference；有效 baseline 始终保留。",
      "complete payload 必须含 exploration_value,implementation_reuse,method_reference,baseline_quality,cross_layer_validity,experiment_readiness,decision,rationale,minimum_implementation_plan[],baseline_ablation_matrix[],metrics_tools[],failure_stop_conditions[],selected_refs[],alternative_refs[],gaps[]。",
    ].join("\n"),
  ].join("\n\n");
}

function evidenceAnswerPrompt(
  state: RunState,
  direction: Direction,
  questionId: string,
  dimension: string,
  question: string,
  evidenceNeed: string,
): string {
  return [
    "回答一个 Direction Review 的原子证据问题；回答后终止。你不做最终价值裁决。",
    `QUESTION:\n${JSON.stringify({
      direction_id: direction.directionId,
      question_id: questionId,
      dimension,
      question,
      evidence_need: evidenceNeed,
    })}`,
    `EXPERIMENT_BUNDLE_AND_EXISTING_EVIDENCE:\n${JSON.stringify(directionBundle(state, direction))}`,
    `允许检索的 vault 根目录：${JSON.stringify(state.config.evidenceRoots)}`,
    [
      "先使用提供的 canonical claims；只有证据不足时才用 Obsidian 只读 search/get/list 补查。",
      "SOURCES 必须是 JSON array。引用已有 claim 时写 {\"claim_id\":\"C-...\"}；新增来源时写完整原子 claim（statement,claim_type,evidence_kind,source_path,line_start,line_end,quote,applicable_scope,confidence）。",
      "外层 control 字段必须写 dimension 和 conclusion(supported|contradicted|partial|unknown|not_applicable)。",
      "ANSWER 要明确区分 direct、inferred 和 unknown，并回答退化边界或缺口。",
      "不要提出新的 Direction，不要做最终 decision。",
    ].join("\n"),
  ].join("\n\n");
}

async function runReviewEvidence(
  runtime: RuntimeManager,
  store: CanonicalStore,
  state: RunState,
  direction: Direction,
  round: number,
  questionId: string,
  dimension: string,
  question: string,
  evidenceNeed: string,
): Promise<ReviewQuestionAnswer> {
  const taskId = `RE-${direction.directionId}-R${String(round).padStart(2, "0")}`;
  const inputPath = path.join(store.workDir, "tasks/review_evidence", `${taskId}.input.json`);
  const outputPath = path.join(store.workDir, "tasks/review_evidence", `${taskId}.output.json`);
  atomicWriteJson(inputPath, {
    task_id: taskId,
    direction_id: direction.directionId,
    question_id: questionId,
    dimension,
    question,
    evidence_need: evidenceNeed,
  });
  const record: TaskRecord = {
    taskId,
    role: "review_evidence_worker",
    scopeId: direction.directionId,
    status: "dispatched",
    attempts: 1,
    inputPath,
    outputPath,
    error: null,
  };
  state.tasks[taskId] = record;
  store.save(state, `dispatch_${taskId}`);
  try {
    const handle = await runtime.startAgent("review_evidence_worker", taskId, false);
    const result = await runtime.runProtocolTurn(
      handle,
      evidenceAnswerPrompt(state, direction, questionId, dimension, question, evidenceNeed),
    );
    record.status = "protocol_valid";
    const payload = asObject(result.protocol.payload) ?? {};
    const sourceItems = Array.isArray(payload.sources) ? payload.sources : [];
    const existingRefs: string[] = [];
    const newCandidates: unknown[] = [];
    for (const source of sourceItems) {
      const object = asObject(source);
      const existingId = object && typeof object.claim_id === "string" ? object.claim_id : null;
      if (existingId && state.claims.some((claim) => claim.claimId === existingId)) {
        existingRefs.push(existingId);
      } else {
        newCandidates.push(source);
      }
    }
    const validated = validateClaimCandidates(
      newCandidates,
      taskId,
      state.config.vaultRoot,
      state.config.evidenceRoots,
    );
    for (const claim of validated.accepted) {
      if (!state.claims.some((existing) => existing.claimId === claim.claimId)) {
        state.claims.push(claim);
      }
    }
    state.rejectedClaims.push(...validated.rejected);
    const gaps = stringArray(payload.gaps);
    const conclusionRaw = result.protocol.control.conclusion;
    const conclusion = ["supported", "contradicted", "partial", "unknown", "not_applicable"].includes(conclusionRaw)
      ? conclusionRaw as ReviewQuestionAnswer["conclusion"]
      : (existingRefs.length + validated.accepted.length > 0 ? "partial" : "unknown");
    const qa: ReviewQuestionAnswer = {
      round,
      questionId,
      dimension: dimension as ReviewQuestionAnswer["dimension"],
      question,
      answer: typeof payload.answer === "string" ? payload.answer : "",
      conclusion,
      evidenceRefs: [...new Set([...existingRefs, ...validated.accepted.map((claim) => claim.claimId)])].sort(),
      gaps,
    };
    record.status = "committed";
    atomicWriteJson(outputPath, {
      protocol: result.protocol,
      normalized_qa: qa,
      rejected_new_source_count: validated.rejected.length,
    } as unknown as JsonValue);
    store.save(state, `commit_${taskId}`);
    return qa;
  } catch (error) {
    record.status = "failed_terminal";
    record.error = error instanceof Error ? error.message : String(error);
    store.save(state, `fail_${taskId}`);
    return {
      round,
      questionId,
      dimension: dimension as ReviewQuestionAnswer["dimension"],
      question,
      answer: "",
      conclusion: "unknown",
      evidenceRefs: [],
      gaps: [record.error],
    };
  }
}

function loadProgress(progressPath: string): ReviewProgress {
  if (!fs.existsSync(progressPath)) {
    return {
      questionAnswers: [],
      referenceKeysUsed: [],
      lastReviewerOutput: null,
      lastResult: null,
    };
  }
  return JSON.parse(fs.readFileSync(progressPath, "utf8")) as ReviewProgress;
}

function pendingReview(
  directionId: string,
  progress: ReviewProgress,
  reason: string,
): ExpertReview {
  return {
    directionId,
    status: "pending",
    pendingReason: reason,
    explorationValue: "unknown",
    implementationReuse: "unknown",
    methodReference: "unknown",
    baselineQuality: "unknown",
    crossLayerValidity: "unknown",
    experimentReadiness: "unknown",
    decision: "needs_evidence",
    rationale: "Review did not reach a validated terminal judgment within the configured budget.",
    minimumImplementationPlan: [],
    baselineAblationMatrix: [],
    metricsTools: [],
    failureStopConditions: [],
    selectedRefs: [],
    alternativeRefs: [],
    gaps: [...new Set(progress.questionAnswers.flatMap((qa) => qa.gaps))],
    questionAnswers: progress.questionAnswers,
    referenceKeysUsed: progress.referenceKeysUsed,
  };
}

function upsertReview(state: RunState, review: ExpertReview): void {
  const index = state.reviews.findIndex((item) => item.directionId === review.directionId);
  if (index < 0) state.reviews.push(review);
  else state.reviews[index] = review;
}

async function reviewDirection(
  runtime: RuntimeManager,
  store: CanonicalStore,
  state: RunState,
  direction: Direction,
): Promise<void> {
  const status = state.stage2.directionReview[direction.directionId] ?? {
    status: "pending" as const,
    reason: null,
    rounds: 0,
  };
  state.stage2.directionReview[direction.directionId] = status;
  if (state.reviews.some((review) => review.directionId === direction.directionId)) {
    status.status = "complete";
    return;
  }
  status.status = "running";
  const progressPath = path.join(store.workDir, "reviews", `${direction.directionId}.progress.json`);
  const progress = loadProgress(progressPath);
  let reviewer: AgentHandle = await runtime.persistentAgent("direction_reviewer", direction.directionId);

  while (status.rounds < state.config.maxReviewRoundsPerDirection) {
    if (state.usage.turns >= state.config.maxTotalTurns) {
      const review = pendingReview(direction.directionId, progress, "run_turn_budget_exhausted");
      upsertReview(state, review);
      status.status = "complete";
      status.reason = review.pendingReason;
      store.save(state, `review_${direction.directionId}_pending`);
      return;
    }
    try {
      const decision = await runtime.runProtocolTurn(
        reviewer,
        reviewerPrompt(state, direction, progress),
      );
      reviewer = decision.handle;
      status.rounds += 1;
      progress.lastReviewerOutput = decision.protocol.rawText;

      if (decision.protocol.marker === "REVIEW_QUESTION") {
        const payload = asObject(decision.protocol.payload);
        const dimension = reviewDimension(payload?.dimension);
        const questionId = String(payload?.question_id ?? payload?.questionId ?? `Q-${status.rounds}`).trim();
        const question = String(payload?.question ?? "").trim();
        const evidenceNeed = String(payload?.evidence_need ?? payload?.evidenceNeed ?? "").trim();
        if (!dimension || !question) {
          progress.lastResult = {
            accepted: false,
            error: "question requires a valid dimension and nonempty question",
          };
        } else {
          const qa = await runReviewEvidence(
            runtime,
            store,
            state,
            direction,
            status.rounds,
            questionId,
            dimension,
            question,
            evidenceNeed,
          );
          progress.questionAnswers.push(qa);
          progress.lastResult = {
            type: "REVIEW_EVIDENCE_RESULT",
            question_id: questionId,
            dimension,
            conclusion: qa.conclusion,
            evidence_refs: qa.evidenceRefs,
            gaps: qa.gaps,
            answer: qa.answer,
          };
        }
      } else if (decision.protocol.marker === "REVIEW_REFERENCE_REQUEST") {
        const payload = asObject(decision.protocol.payload);
        const key = String(payload?.reference_key ?? payload?.referenceKey ?? "").trim();
        if (
          !REVIEW_REFERENCE_KEYS.includes(key as typeof REVIEW_REFERENCE_KEYS[number])
          || progress.referenceKeysUsed.includes(key)
        ) {
          progress.lastResult = {
            accepted: false,
            error: "reference key is invalid or was already injected",
            allowed_unused_keys: REVIEW_REFERENCE_KEYS.filter((candidate) => !progress.referenceKeysUsed.includes(candidate)),
          };
        } else {
          const referencePath = path.join(
            state.config.skillRoot,
            "learning-direction-reviewer",
            "references",
            `${key}.md`,
          );
          if (!fs.existsSync(referencePath)) {
            progress.lastResult = { accepted: false, error: `reference file missing: ${key}` };
          } else {
            progress.referenceKeysUsed.push(key);
            progress.lastResult = {
              type: "REVIEW_REFERENCE",
              reference_key: key,
              content: fs.readFileSync(referencePath, "utf8"),
            };
          }
        }
      } else {
        const built = buildExpertReview(
          direction.directionId,
          decision.protocol.payload!,
          progress.questionAnswers,
          progress.referenceKeysUsed,
        );
        if (built.review) {
          upsertReview(state, built.review);
          status.status = "complete";
          status.reason = null;
          atomicWriteJson(progressPath, progress as unknown as JsonValue);
          store.save(state, `review_${direction.directionId}_complete`);
          return;
        }
        progress.lastResult = {
          accepted: false,
          errors: built.errors,
          missing_dimensions: REVIEW_DIMENSIONS.filter(
            (dimension) => !progress.questionAnswers.some((qa) => qa.dimension === dimension),
          ),
        };
      }
      atomicWriteJson(progressPath, progress as unknown as JsonValue);
      store.save(state, `review_${direction.directionId}_round_${status.rounds}`);
    } catch (error) {
      const review = pendingReview(
        direction.directionId,
        progress,
        `reviewer_error: ${error instanceof Error ? error.message : String(error)}`,
      );
      upsertReview(state, review);
      status.status = "complete";
      status.reason = review.pendingReason;
      store.save(state, `review_${direction.directionId}_error`);
      return;
    }
  }
  const review = pendingReview(direction.directionId, progress, "review_round_budget_exhausted");
  upsertReview(state, review);
  status.status = "complete";
  status.reason = review.pendingReason;
  store.save(state, `review_${direction.directionId}_pending`);
}

export async function runDirectionStage(
  state: RunState,
  store: CanonicalStore,
  runtime: RuntimeManager,
  stopAfter: "direction-plan" | "direction-review" | null = null,
): Promise<void> {
  if (state.stage1.status !== "complete") {
    throw new Error("Stage 2 cannot start before Stage 1 is complete");
  }
  const acceptedAnchorIds = [...state.stage1.acceptedAnchorIds].sort();
  if (state.stage2.status === "pending" || state.stage2.status === "planning") {
    state.stage2.status = "planning";
    store.save(state, "stage2_planning_started");
    await mapWithConcurrency(
      acceptedAnchorIds,
      state.config.directionConcurrency,
      (anchorId) => planAnchor(runtime, store, state, anchorId),
    );
    if (stopAfter === "direction-plan") return;
    state.stage2.status = "reviewing";
    store.save(state, "stage2_review_started");
  }

  if (state.stage2.status === "reviewing") {
    const directions = state.directions
      .filter((direction) => direction.status === "accepted")
      .sort((left, right) => left.directionId.localeCompare(right.directionId));
    await mapWithConcurrency(
      directions,
      state.config.directionConcurrency,
      (direction) => reviewDirection(runtime, store, state, direction),
    );
    if (stopAfter === "direction-review") return;
  }

  const allPlanningTerminal = acceptedAnchorIds.every((anchorId) => {
    const status = state.stage2.anchorPlanning[anchorId]?.status;
    return status === "complete" || status === "pending_budget";
  });
  const allReviewsTerminal = state.directions
    .filter((direction) => direction.status === "accepted")
    .every((direction) => state.reviews.some((review) => review.directionId === direction.directionId));
  if (!allPlanningTerminal || !allReviewsTerminal) {
    throw new Error("Stage 2 terminal-state invariant failed");
  }
  state.stage2.status = "complete";
  state.status = "complete";
  store.save(state, "stage2_complete");
}
