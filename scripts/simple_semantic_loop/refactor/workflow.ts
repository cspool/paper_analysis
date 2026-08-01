import { FileLoopStore } from "./store.ts";
import type {
  AnchorIndexEntry,
  BranchEffect,
  CommittedResult,
  DecisionContext,
  DirectionIndexEntry,
  LoopDecision,
  ObjectKind,
  ObjectsIndex,
  PendingPair,
  PendingResults,
  PreReview,
  ReviewVerdict,
  SequenceStep,
  StateFile,
  TaskBinding,
  TurnTask,
  WorkAction,
} from "./types.ts";

export function initialSequence(): SequenceStep[] {
  return [
    { role: "WORKER", mode: "NORMAL_WORK", bindingRef: null },
    { role: "REVIEWER", mode: "PAIR_REVIEW", bindingRef: null },
    { role: "DECISION", mode: "DECISION", bindingRef: null },
  ];
}

export function sequenceAfterDecision(
  decision: LoopDecision,
  bindingRef: string | null = null,
): SequenceStep[] {
  switch (decision) {
    case "RUN_WORKER":
      return initialSequence();
    case "RUN_REVIEWER":
      return [
        { role: "REVIEWER", mode: "PRE_REVIEW", bindingRef: null },
        { role: "WORKER", mode: "NORMAL_WORK", bindingRef: null },
        { role: "REVIEWER", mode: "PAIR_REVIEW", bindingRef: null },
        { role: "DECISION", mode: "DECISION", bindingRef: null },
      ];
    case "RETRY_WORKER":
      if (!bindingRef) throw new Error("RETRY_WORKER requires a binding");
      return [
        { role: "WORKER", mode: "RETRY_WORK", bindingRef },
        { role: "REVIEWER", mode: "PAIR_REVIEW", bindingRef: null },
        { role: "DECISION", mode: "DECISION", bindingRef: null },
      ];
    case "RETRY_REVIEWER":
      if (!bindingRef) throw new Error("RETRY_REVIEWER requires a binding");
      return [
        { role: "REVIEWER", mode: "RETRY_REVIEW", bindingRef },
        { role: "DECISION", mode: "DECISION", bindingRef: null },
      ];
    case "FINISH_WORKFLOW":
      return [];
  }
}

export function createWorkerBinding(
  store: FileLoopStore,
  state: StateFile,
): { bindingRef: string; binding: TaskBinding; task: TurnTask } {
  const choice = chooseWorkerAction(store, state);
  const bindingId = store.newId("binding");
  const taskRef = `tasks/${bindingId}/turn_task.json`;
  if (
    choice.action === "CREATE_ANCHOR" &&
    store.exists("observations/research_memory.json")
  ) {
    const memoryRef = `tasks/${bindingId}/research_memory_snapshot.json`;
    store.writeImmutableText(
      memoryRef,
      store.readText("observations/research_memory.json"),
    );
    choice.inputs.researchMemory = memoryRef;
  }
  const binding: TaskBinding = {
    bindingId,
    createdAt: new Date().toISOString(),
    role: "WORKER",
    action: choice.action,
    taskRef,
    objectKind: choice.objectKind,
    objectId: choice.objectId,
    revision: choice.revision,
    parentAnchorId: choice.parentAnchorId,
    resultRefName:
      choice.objectKind === "ANCHOR"
        ? "work-result-anchor-v2"
        : "work-result-direction-v2",
    sourceDecisionTurnRef: state.latestDecisionTurnRef,
  };
  const task = workerTaskForChoice(choice);
  const bindingRef = `bindings/${bindingId}.json`;
  store.writeJson(taskRef, task);
  store.writeJson(bindingRef, binding);
  store.appendEvent("TASK_BOUND", [bindingRef, taskRef]);
  return { bindingRef, binding, task };
}

export function createReviewerBinding(
  store: FileLoopStore,
  state: StateFile,
  mode: "PAIR_REVIEW" | "PRE_REVIEW",
): { bindingRef: string; binding: TaskBinding; task: TurnTask } {
  const target =
    mode === "PAIR_REVIEW"
      ? reviewTargetFromPending(state.pending)
      : selectPreReviewTarget(store);
  const bindingId = store.newId("binding");
  const taskRef = `tasks/${bindingId}/turn_task.json`;
  const action =
    target.objectKind === "ANCHOR"
      ? "REVIEW_ANCHOR"
      : "REVIEW_DIRECTION";
  const binding: TaskBinding = {
    bindingId,
    createdAt: new Date().toISOString(),
    role: "REVIEWER",
    action,
    taskRef,
    objectKind: target.objectKind,
    objectId: target.objectId,
    revision: target.revision,
    parentAnchorId: target.parentAnchorId,
    resultRefName: "review-result-v2",
    sourceDecisionTurnRef: state.latestDecisionTurnRef,
  };
  const inputs: TurnTask["inputs"] = {
    reviewTarget: target.workRef,
  };
  if (mode === "PAIR_REVIEW" && target.revision > 1) {
    const previousReview = previousRevisionReviewRef(store, target);
    if (previousReview) inputs.previousReview = previousReview;
  }
  if (target.objectKind === "DIRECTION") {
    inputs.boundAnchor = latestAnchorWorkRef(
      store.readObjects(),
      target.parentAnchorId!,
    );
  }
  const task: TurnTask = {
    goalRef: "workflow_goal.json",
    action,
    objective:
      mode === "PRE_REVIEW"
        ? "从独立角度重新审阅当前对象，为随后深化该对象或创建同类替代对象提供结论。"
        : "独立审阅当前 Work Result 是否满足最终需求和对象合同。",
    inputs,
    requirements: [
      "核对目标范围、baseline、证据和字段间语义一致性",
      "对 Direction 核对最小可检验主要变化、联合包归因边界和最小充分表达",
      "按对象类型应用 Review Rubric 并给出唯一 verdict",
      "由 Reviewer 记录会改变审阅结论的对象局部 query gaps",
      ...(inputs.previousReview
        ? [
          "使用 previousReview 确认上一轮 correction boundary，但只对当前 reviewTarget 独立作出 verdict",
        ]
        : []),
    ],
    constraints: [
      "只审阅 reviewTarget；boundAnchor 仅作为 Direction 范围上下文",
      "若提供 previousReview，它只界定上一轮修订边界，不是当前 verdict 的替代品",
      "不得替换被审阅内容、作出调度决策或声称全局完成",
      "不得执行新实验",
    ],
  };
  const bindingRef = `bindings/${bindingId}.json`;
  store.writeJson(taskRef, task);
  store.writeJson(bindingRef, binding);
  store.appendEvent("TASK_BOUND", [bindingRef, taskRef]);
  return { bindingRef, binding, task };
}

function previousRevisionReviewRef(
  store: FileLoopStore,
  target: {
    objectKind: ObjectKind;
    objectId: string;
    revision: number;
  },
): string | null {
  const index = store.readObjects();
  const entry = target.objectKind === "ANCHOR"
    ? index.anchors[target.objectId]
    : index.directions[target.objectId];
  return entry?.revisions[String(target.revision - 1)]?.reviewRef ?? null;
}

export function createDecisionContext(
  store: FileLoopStore,
  state: StateFile,
  observationRef: string,
): DecisionContext {
  return {
    goalRef: "workflow_goal.json",
    committedResults: committedResultsProjection(store),
    pendingResults: pendingResultsProjection(store, state.pending),
    remainingRequirementsAfterPendingCommit: computeRemainingRequirements(
      store,
      state,
      true,
    ),
    observationRef,
  };
}

export function previewBranchEffects(
  store: FileLoopStore,
  state: StateFile,
  allowed: LoopDecision[],
): BranchEffect[] {
  const projected = structuredClone(store.readObjects());
  if (state.pending?.reviewRef) {
    applyPairToIndex(store, projected, state.pending, "PENDING_DECISION");
  }
  return allowed.map((decision): BranchEffect => {
    if (decision === "FINISH_WORKFLOW") {
      return {
        decision,
        nextRole: null,
        nextAction: "FINALIZE",
        targetRef: null,
        sequence: [],
      };
    }
    if (decision === "RETRY_WORKER") {
      const ref = state.pending?.workTaskBindingRef;
      if (!ref) throw new Error("RETRY_WORKER preview lacks binding");
      const binding = store.readJson<TaskBinding>(ref);
      return {
        decision,
        nextRole: "WORKER",
        nextAction: binding.action,
        targetRef: binding.taskRef,
        sequence: sequenceAfterDecision(decision, ref).map((step) => step.role),
      };
    }
    if (decision === "RETRY_REVIEWER") {
      const ref = state.pending?.reviewTaskBindingRef;
      if (!ref) throw new Error("RETRY_REVIEWER preview lacks binding");
      const binding = store.readJson<TaskBinding>(ref);
      return {
        decision,
        nextRole: "REVIEWER",
        nextAction: binding.action,
        targetRef: binding.taskRef,
        sequence: sequenceAfterDecision(decision, ref).map((step) => step.role),
      };
    }
    if (decision === "RUN_REVIEWER") {
      const target = selectPreReviewTarget(store, projected);
      return {
        decision,
        nextRole: "REVIEWER",
        nextAction:
          target.objectKind === "ANCHOR"
            ? "REVIEW_ANCHOR"
            : "REVIEW_DIRECTION",
        targetRef: target.workRef,
        sequence: sequenceAfterDecision(decision).map((step) => step.role),
      };
    }
    const choice = chooseWorkerAction(
      store,
      { ...state, pending: null, preReview: null },
      projected,
    );
    return {
      decision,
      nextRole: "WORKER",
      nextAction: choice.action,
      targetRef:
        choice.inputs.currentWork ?? choice.inputs.boundAnchor ?? null,
      sequence: sequenceAfterDecision(decision).map((step) => step.role),
    };
  });
}

export function committedResultsProjection(
  store: FileLoopStore,
): CommittedResult[] {
  const index = store.readObjects();
  const results: CommittedResult[] = [];
  for (const anchor of Object.values(index.anchors)) {
    const anchorRevision = latestRevision(anchor);
    results.push({
      objectKind: "ANCHOR",
      work: anchorRevision.workRef,
      review: anchorRevision.reviewRef,
    });
    for (const directionId of anchor.directionIds) {
      const direction = index.directions[directionId];
      if (!direction) continue;
      const directionRevision = latestRevision(direction);
      results.push({
        objectKind: "DIRECTION",
        anchorWork: boundAnchorForRevision(store, directionRevision),
        work: directionRevision.workRef,
        review: directionRevision.reviewRef,
      });
    }
  }
  return results;
}

export function pendingResultsProjection(
  store: FileLoopStore,
  pending: PendingPair | null,
): PendingResults | null {
  if (!pending?.reviewRef) return null;
  if (pending.objectKind === "ANCHOR") {
    return {
      objectKind: "ANCHOR",
      workTask: pending.workTaskRef,
      work: pending.workRef,
      review: pending.reviewRef,
    };
  }
  return {
    objectKind: "DIRECTION",
    anchorWork: latestAnchorWorkRef(
      store.readObjects(),
      pending.parentAnchorId!,
    ),
    workTask: pending.workTaskRef,
    work: pending.workRef,
    review: pending.reviewRef,
  };
}

export function computeRemainingRequirements(
  store: FileLoopStore,
  state: StateFile,
  includePending: boolean,
): string[] {
  const index = structuredClone(store.readObjects());
  if (includePending && state.pending?.reviewRef) {
    applyPairToIndex(store, index, state.pending, "PENDING_DECISION");
  }
  const requirements: string[] = [];
  const activeAnchors = index.activeAnchorIds
    .map((anchorId) => index.anchors[anchorId])
    .filter((anchor): anchor is AnchorIndexEntry =>
      Boolean(anchor && !anchor.rejected)
    );
  if (activeAnchors.length === 0) {
    requirements.push("ANCHOR_REQUIRED");
    return requirements;
  }

  for (const anchor of activeAnchors) {
    const anchorRevision = latestRevision(anchor);
    if (!revisionAccepted(store, anchorRevision)) {
      requirements.push(
        `ANCHOR_REVIEW_PASS_REQUIRED:${anchorRevision.workRef}`,
      );
      continue;
    }
    const directions = anchor.directionIds
      .map((directionId) => index.directions[directionId])
      .filter((direction): direction is DirectionIndexEntry =>
        Boolean(
          direction &&
            !direction.rejected &&
            boundAnchorForRevision(
                store,
                latestRevision(direction),
              ) === anchorRevision.workRef,
        )
      );
    if (directions.length === 0) {
      requirements.push(`DIRECTION_REQUIRED:${anchorRevision.workRef}`);
      continue;
    }
    for (const direction of directions) {
      const revision = latestRevision(direction);
      if (!revisionAccepted(store, revision)) {
        requirements.push(
          `DIRECTION_REVIEW_PASS_REQUIRED:${revision.workRef}`,
        );
      }
    }
  }
  return requirements;
}

export function allowedDecisions(
  store: FileLoopStore,
  state: StateFile,
): LoopDecision[] {
  const allowed: LoopDecision[] = ["RUN_WORKER"];
  if (hasReviewableObjectAfterPending(store, state)) {
    allowed.push("RUN_REVIEWER");
  }
  if (state.pending) {
    allowed.push("RETRY_WORKER");
  }
  if (state.pending?.reviewTaskBindingRef) {
    allowed.push("RETRY_REVIEWER");
  }
  if (computeRemainingRequirements(store, state, true).length === 0) {
    allowed.push("FINISH_WORKFLOW");
  }
  return allowed;
}

export function commitPending(
  store: FileLoopStore,
  state: StateFile,
  decisionTurnRef: string,
): void {
  if (!state.pending?.reviewRef || !state.pending.reviewVerdict) {
    throw new Error("normal Decision requires a complete pending pair");
  }
  const index = store.readObjects();
  const existing =
    state.pending.objectKind === "ANCHOR"
      ? index.anchors[state.pending.objectId]?.revisions[
        String(state.pending.revision)
      ]
      : index.directions[state.pending.objectId]?.revisions[
        String(state.pending.revision)
      ];
  if (
    existing?.workRef === state.pending.workRef &&
    existing.reviewRef === state.pending.reviewRef
  ) {
    store.appendEvent("PENDING_RESULTS_ALREADY_COMMITTED", [
      state.pending.workRef,
      state.pending.reviewRef,
    ]);
    return;
  }
  if (existing) {
    throw new Error(
      `object revision ${state.pending.objectId}@${state.pending.revision} is already bound to different results`,
    );
  }
  applyPairToIndex(store, index, state.pending, decisionTurnRef);
  index.revision += 1;
  store.writeObjects(index);
  const round = store.readRound(state.round);
  round.committedAt = new Date().toISOString();
  store.writeRound(round);
  store.appendEvent("PENDING_RESULTS_COMMITTED", [
    state.pending.workRef,
    state.pending.reviewRef,
    decisionTurnRef,
  ]);
}

export function commitPreReview(
  store: FileLoopStore,
  binding: TaskBinding,
  workRef: string,
  reviewRef: string,
  reviewVerdict: ReviewVerdict,
): void {
  if (binding.role !== "REVIEWER") {
    throw new Error("pre-review commit requires a Reviewer binding");
  }
  const index = store.readObjects();
  const entry =
    binding.objectKind === "ANCHOR"
      ? index.anchors[binding.objectId]
      : index.directions[binding.objectId];
  if (!entry || entry.latestRevision !== binding.revision) {
    throw new Error(
      `pre-review target ${binding.objectId}@${binding.revision} is not the current object revision`,
    );
  }
  const revision = entry.revisions[String(binding.revision)];
  if (!revision || revision.workRef !== workRef) {
    throw new Error(
      `pre-review target ${binding.objectId}@${binding.revision} does not match ${workRef}`,
    );
  }

  revision.reviewRef = reviewRef;
  revision.reviewVerdict = reviewVerdict;

  if (binding.objectKind === "ANCHOR") {
    const anchor = index.anchors[binding.objectId]!;
    anchor.rejected = reviewVerdict === "REJECT";
    if (anchor.rejected) {
      index.activeAnchorIds = index.activeAnchorIds.filter(
        (id) => id !== anchor.objectId,
      );
    } else if (!index.activeAnchorIds.includes(anchor.objectId)) {
      index.activeAnchorIds.push(anchor.objectId);
    }
  } else {
    index.directions[binding.objectId]!.rejected =
      reviewVerdict === "REJECT";
  }

  index.revision += 1;
  store.writeObjects(index);
  store.appendEvent("PRE_REVIEW_COMMITTED", [workRef, reviewRef]);
}

interface WorkerChoice {
  action: WorkAction;
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  inputs: TurnTask["inputs"];
  intent: "STANDARD" | "CONVERGENCE_PROBE";
}

function chooseWorkerAction(
  store: FileLoopStore,
  state: StateFile,
  suppliedIndex?: ObjectsIndex,
): WorkerChoice {
  const index = suppliedIndex ?? store.readObjects();
  if (state.preReview) {
    return choiceFromPreReview(store, index, state.preReview);
  }

  for (const anchorId of index.activeAnchorIds) {
    const anchor = index.anchors[anchorId];
    if (!anchor || anchor.rejected) continue;
    const revision = latestRevision(anchor);
    if (
      revision.reviewVerdict === "REVISE" ||
      (revision.reviewVerdict === "PASS" &&
        !revisionAccepted(store, revision))
    ) {
      return deepenAnchorChoice(anchor, revision.workRef, revision.reviewRef);
    }
  }
  for (const anchorId of index.activeAnchorIds) {
    const anchor = index.anchors[anchorId];
    if (!anchor || anchor.rejected) continue;
    for (const directionId of anchor.directionIds) {
      const direction = index.directions[directionId];
      if (!direction || direction.rejected) continue;
      const revision = latestRevision(direction);
      if (
        revision.reviewVerdict === "REVISE" ||
        (revision.reviewVerdict === "PASS" &&
          !revisionAccepted(store, revision))
      ) {
        return deepenDirectionChoice(
          index,
          direction,
          revision.workRef,
          revision.reviewRef,
        );
      }
    }
  }

  const activeAnchors = index.activeAnchorIds
    .map((anchorId) => index.anchors[anchorId])
    .filter((anchor): anchor is AnchorIndexEntry =>
      Boolean(anchor && !anchor.rejected)
    );
  if (activeAnchors.length === 0) {
    const rejectedAnchor = Object.values(index.anchors)
      .filter((anchor) => anchor.rejected)
      .at(-1);
    const rejectedRevision = rejectedAnchor
      ? latestRevision(rejectedAnchor)
      : null;
    return createAnchorChoice(
      store,
      rejectedRevision?.workRef,
      rejectedRevision?.reviewRef,
    );
  }

  for (const anchor of activeAnchors) {
    const latest = latestRevision(anchor);
    if (!revisionAccepted(store, latest)) continue;
    const viableDirections = anchor.directionIds
      .map((directionId) => index.directions[directionId])
      .filter((direction) =>
        direction &&
        !direction.rejected &&
        boundAnchorForRevision(store, latestRevision(direction)) ===
          latest.workRef
      );
    if (viableDirections.length === 0) {
      const staleDirection = anchor.directionIds
        .map((directionId) => index.directions[directionId])
        .find((direction): direction is DirectionIndexEntry =>
          Boolean(direction && !direction.rejected)
        );
      if (staleDirection) {
        const revision = latestRevision(staleDirection);
        return deepenDirectionChoice(
          index,
          staleDirection,
          revision.workRef,
          revision.reviewRef,
        );
      }
      const rejectedDirection = anchor.directionIds
        .map((directionId) => index.directions[directionId])
        .filter((direction): direction is DirectionIndexEntry =>
          Boolean(direction?.rejected)
        )
        .at(-1);
      return createDirectionChoice(
        store,
        index,
        anchor.objectId,
        rejectedDirection
          ? latestRevision(rejectedDirection).workRef
          : undefined,
        rejectedDirection
          ? latestRevision(rejectedDirection).reviewRef
          : undefined,
      );
    }
  }

  // Minimum closure is already satisfied. A Decision that still selects
  // RUN_WORKER deterministically expands the Topic through one new Anchor.
  return createAnchorChoice(
    store,
    undefined,
    undefined,
    "CONVERGENCE_PROBE",
  );
}

function choiceFromPreReview(
  store: FileLoopStore,
  index: ObjectsIndex,
  preReview: PreReview,
): WorkerChoice {
  if (preReview.objectKind === "ANCHOR") {
    const anchor = index.anchors[preReview.objectId];
    if (!anchor) throw new Error("pre-review Anchor is not indexed");
    if (preReview.reviewVerdict === "REJECT") {
      return createAnchorChoice(
        store,
        preReview.workRef,
        preReview.reviewRef,
      );
    }
    return deepenAnchorChoice(
      anchor,
      preReview.workRef,
      preReview.reviewRef,
    );
  }
  const direction = index.directions[preReview.objectId];
  if (!direction) throw new Error("pre-review Direction is not indexed");
  if (preReview.reviewVerdict === "REJECT") {
    return createDirectionChoice(
      store,
      index,
      direction.parentAnchorId,
      preReview.workRef,
      preReview.reviewRef,
    );
  }
  return deepenDirectionChoice(
    index,
    direction,
    preReview.workRef,
    preReview.reviewRef,
  );
}

function createAnchorChoice(
  store: FileLoopStore,
  currentWork?: string,
  latestReview?: string,
  intent: WorkerChoice["intent"] = "STANDARD",
): WorkerChoice {
  return {
    action: "CREATE_ANCHOR",
    objectKind: "ANCHOR",
    objectId: store.newId("anchor"),
    revision: 1,
    parentAnchorId: null,
    inputs: {
      ...(currentWork ? { currentWork } : {}),
      ...(latestReview ? { latestReview } : {}),
    },
    intent,
  };
}

function deepenAnchorChoice(
  anchor: AnchorIndexEntry,
  workRef: string,
  reviewRef: string,
): WorkerChoice {
  return {
    action: "DEEPEN_ANCHOR",
    objectKind: "ANCHOR",
    objectId: anchor.objectId,
    revision: anchor.latestRevision + 1,
    parentAnchorId: null,
    inputs: {
      currentWork: workRef,
      latestReview: reviewRef,
    },
    intent: "STANDARD",
  };
}

function createDirectionChoice(
  store: FileLoopStore,
  index: ObjectsIndex,
  anchorId: string,
  currentWork?: string,
  latestReview?: string,
): WorkerChoice {
  return {
    action: "CREATE_DIRECTION",
    objectKind: "DIRECTION",
    objectId: store.newId("direction"),
    revision: 1,
    parentAnchorId: anchorId,
    inputs: {
      boundAnchor: latestAnchorWorkRef(index, anchorId),
      ...(currentWork ? { currentWork } : {}),
      ...(latestReview ? { latestReview } : {}),
    },
    intent: "STANDARD",
  };
}

function deepenDirectionChoice(
  index: ObjectsIndex,
  direction: DirectionIndexEntry,
  workRef: string,
  reviewRef: string,
): WorkerChoice {
  return {
    action: "DEEPEN_DIRECTION",
    objectKind: "DIRECTION",
    objectId: direction.objectId,
    revision: direction.latestRevision + 1,
    parentAnchorId: direction.parentAnchorId,
    inputs: {
      boundAnchor: latestAnchorWorkRef(index, direction.parentAnchorId),
      currentWork: workRef,
      latestReview: reviewRef,
    },
    intent: "STANDARD",
  };
}

function workerTaskForChoice(choice: WorkerChoice): TurnTask {
  const isAnchor = choice.objectKind === "ANCHOR";
  const isDeepen =
    choice.action === "DEEPEN_ANCHOR" ||
    choice.action === "DEEPEN_DIRECTION";
  const hasLatestReview = Boolean(choice.inputs.latestReview);
  const isConvergenceProbe = choice.intent === "CONVERGENCE_PROBE";
  return {
    goalRef: "workflow_goal.json",
    action: choice.action,
    objective: isAnchor
      ? isDeepen
        ? "根据最近独立审阅深化同一 Anchor，并返回完整修订结果。"
        : isConvergenceProbe
        ? "执行一次有界的 Topic 收敛探测：只在本地证据支持实质新颖、非重复性能矛盾时创建 Anchor；否则如实返回 BLOCKED_NO_RESULT。"
        : hasLatestReview
        ? "根据最近独立审阅创建同类替代 Anchor，并避免重复已拒绝对象的问题。"
        : "创建一个新的 Anchor，扩展 Topic 约束下由 Anchor 集合定义的 6L 空间。"
      : isDeepen
      ? "根据最近独立审阅深化同一 Direction，并返回完整修订结果。"
      : hasLatestReview
      ? "根据最近独立审阅在绑定 Anchor 内创建同类替代 Direction。"
      : "在绑定 Anchor 内创建一个可证伪的性能优化 Direction。",
    inputs: choice.inputs,
    requirements: isAnchor
      ? [
        "明确具体场景、baseline、可观察性能矛盾和非空 6L 区域",
        "保持 Topic 和最终 acceptance criteria，不与已有内容作无实质差异的重复",
        ...(isConvergenceProbe
          ? [
            "读取 researchMemory 的 accepted、needsRevision、rejectedLessons 和动态 6L coverage，优先检查低证据密度或尚未充分比较的区域",
            "对少量有代表性的候选角度执行有界检索；只有机制、baseline 或适用边界 materially different 时才形成新 Anchor",
            "若候选均重复、证据不足或不能形成可验证矛盾，返回 BLOCKED_NO_RESULT，并在 unresolved 中概括检索覆盖和主要重复路线",
          ]
          : []),
        ...(hasLatestReview
          ? ["处理 latestReview 中所有 BLOCKING findings 和相关 query gaps"]
          : []),
      ]
      : [
        "保持 boundAnchor 绑定并明确唯一主要 baseline change",
        "给出机制、条件化预期影响、权衡、失败条件和可证伪测量计划",
        "使用最小充分表达；可拆分变化不得捆绑归因，不可分联合包只声明 package-level effect",
        ...(hasLatestReview
          ? ["处理 latestReview 中所有 BLOCKING findings 和相关 query gaps"]
          : []),
      ],
    constraints: [
      "不得改变用户 Topic、目标或绑定对象",
      "事实结论必须可追溯；假设必须明确标记",
      "不得执行新实验",
    ],
  };
}

function reviewTargetFromPending(pending: PendingPair | null): {
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  workRef: string;
} {
  if (!pending) throw new Error("PAIR_REVIEW requires pending Worker result");
  return {
    objectKind: pending.objectKind,
    objectId: pending.objectId,
    revision: pending.revision,
    parentAnchorId: pending.parentAnchorId,
    workRef: pending.workRef,
  };
}

function selectPreReviewTarget(
  store: FileLoopStore,
  suppliedIndex?: ObjectsIndex,
): {
  objectKind: ObjectKind;
  objectId: string;
  revision: number;
  parentAnchorId: string | null;
  workRef: string;
} {
  const index = suppliedIndex ?? store.readObjects();
  for (const anchorId of [...index.activeAnchorIds].reverse()) {
    const anchor = index.anchors[anchorId];
    if (!anchor || anchor.rejected) continue;
    for (const directionId of [...anchor.directionIds].reverse()) {
      const direction = index.directions[directionId];
      if (!direction || direction.rejected) continue;
      const revision = latestRevision(direction);
      return {
        objectKind: "DIRECTION",
        objectId: direction.objectId,
        revision: direction.latestRevision,
        parentAnchorId: direction.parentAnchorId,
        workRef: revision.workRef,
      };
    }
    const revision = latestRevision(anchor);
    return {
      objectKind: "ANCHOR",
      objectId: anchor.objectId,
      revision: anchor.latestRevision,
      parentAnchorId: null,
      workRef: revision.workRef,
    };
  }
  throw new Error("RUN_REVIEWER requires at least one committed object");
}

function applyPairToIndex(
  store: FileLoopStore,
  index: ObjectsIndex,
  pending: PendingPair,
  decisionTurnRef: string,
): void {
  if (!pending.reviewRef || !pending.reviewVerdict) {
    throw new Error("pending pair has no Review Result control projection");
  }
  const revision = {
    revision: pending.revision,
    workTaskRef: pending.workTaskRef,
    workRef: pending.workRef,
    workOutcome: pending.workOutcome,
    reviewRef: pending.reviewRef,
    reviewVerdict: pending.reviewVerdict,
    committedByDecisionTurnRef: decisionTurnRef,
  };
  if (pending.objectKind === "ANCHOR") {
    const anchor = index.anchors[pending.objectId] ?? {
      objectId: pending.objectId,
      latestRevision: 0,
      revisions: {},
      directionIds: [],
      rejected: false,
    };
    anchor.latestRevision = pending.revision;
    anchor.revisions[String(pending.revision)] = revision;
    anchor.rejected = pending.reviewVerdict === "REJECT";
    index.anchors[pending.objectId] = anchor;
    if (anchor.rejected) {
      index.activeAnchorIds = index.activeAnchorIds.filter(
        (id) => id !== pending.objectId,
      );
    } else if (!index.activeAnchorIds.includes(pending.objectId)) {
      index.activeAnchorIds.push(pending.objectId);
    }
    return;
  }

  if (!pending.parentAnchorId) {
    throw new Error("Direction pending pair lacks parent Anchor");
  }
  const anchor = index.anchors[pending.parentAnchorId];
  if (!anchor || anchor.rejected) {
    throw new Error("Direction parent Anchor is not active");
  }
  const direction = index.directions[pending.objectId] ?? {
    objectId: pending.objectId,
    parentAnchorId: pending.parentAnchorId,
    latestRevision: 0,
    revisions: {},
    rejected: false,
  };
  direction.latestRevision = pending.revision;
  direction.revisions[String(pending.revision)] = revision;
  direction.rejected = pending.reviewVerdict === "REJECT";
  index.directions[pending.objectId] = direction;
  if (!anchor.directionIds.includes(pending.objectId)) {
    anchor.directionIds.push(pending.objectId);
  }
}

function hasReviewableObjectAfterPending(
  store: FileLoopStore,
  state: StateFile,
): boolean {
  const index = structuredClone(store.readObjects());
  if (state.pending?.reviewRef) {
    applyPairToIndex(store, index, state.pending, "PENDING_DECISION");
  }
  return index.activeAnchorIds.some((anchorId) => {
    const anchor = index.anchors[anchorId];
    return Boolean(anchor && !anchor.rejected);
  });
}

function latestAnchorWorkRef(index: ObjectsIndex, anchorId: string): string {
  const anchor = index.anchors[anchorId];
  if (!anchor || anchor.rejected) {
    throw new Error(`Anchor ${anchorId} is not active`);
  }
  return latestRevision(anchor).workRef;
}

function latestRevision(
  entry: AnchorIndexEntry | DirectionIndexEntry,
) {
  const revision = entry.revisions[String(entry.latestRevision)];
  if (!revision) {
    throw new Error(
      `object ${entry.objectId} lacks revision ${entry.latestRevision}`,
    );
  }
  return revision;
}

function revisionAccepted(
  _store: FileLoopStore,
  revision: ReturnType<typeof latestRevision>,
): boolean {
  return (
    revision.reviewVerdict === "PASS" &&
    revision.workOutcome === "READY_FOR_REVIEW"
  );
}

function boundAnchorForRevision(
  store: FileLoopStore,
  revision: ReturnType<typeof latestRevision>,
): string {
  const task = store.readJson<TurnTask>(revision.workTaskRef);
  if (!task.inputs.boundAnchor) {
    throw new Error(
      `Direction task ${revision.workTaskRef} lacks boundAnchor`,
    );
  }
  return task.inputs.boundAnchor;
}
