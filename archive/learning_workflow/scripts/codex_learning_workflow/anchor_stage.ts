import fs from "node:fs";
import path from "node:path";

import { atomicWriteJson, CanonicalStore } from "./canonical_store.ts";
import { mergeCuratorDelta } from "./domain_validators.ts";
import { normalizeIdentityText, stableHash } from "./stable_ids.ts";
import { validateClaimCandidates } from "./source_validator.ts";
import { mapWithConcurrency } from "./task_scheduler.ts";
import type {
  AgentHandle,
  JsonValue,
  Layer,
  RunState,
  TaskRecord,
  ValueAxis,
} from "./types.ts";
import { RuntimeManager } from "./runtime_manager.ts";

interface DiscoveryTask {
  taskId: string;
  focus: string;
  layer: Layer;
  valueAxis: ValueAxis;
  avoid: string[];
}

interface EvidenceTaskResult {
  task: DiscoveryTask;
  acceptedClaimIds: string[];
  gaps: string[];
}

const LAYER_DEFINITIONS: Record<Layer, string> = {
  L1: "算法/Pipeline：计算图、负载分解、动态参数、算法近似与可并行性",
  L2: "Serving/Runtime：请求、batch、stage、资源和执行单元的运行时组织",
  L3: "Compiler：IR、依赖表达、pass、fusion、multiversion、codegen",
  L4: "Kernel：tile/warp/instruction pipeline、同步、数据搬运和 kernel 组合",
  L5: "Architecture：计算/控制单元、存储层次、调度器、NoC 和硬件原语",
  L6: "Chip/System：chiplet、PIM、wafer-scale、封装/互联和芯片级资源边界",
};

function stageStatePacket(state: RunState): JsonValue {
  return {
    round: state.stage1.round,
    accepted_anchor_count: state.stage1.acceptedAnchorIds.length,
    consecutive_rounds_without_new_anchor: state.stage1.consecutiveRoundsWithoutNewAnchor,
    remaining_anchor_capacity: state.config.maxAnchors - state.stage1.acceptedAnchorIds.length,
    remaining_round_budget: state.config.maxStage1Rounds - state.stage1.round,
    remaining_task_budget: state.config.maxStage1Tasks - state.stage1.taskCount,
    anchors: state.anchors
      .filter((anchor) => anchor.status === "accepted")
      .sort((left, right) => left.anchorId.localeCompare(right.anchorId))
      .map((anchor) => ({
        anchor_id: anchor.anchorId,
        signature: anchor.signature,
        layers: Object.fromEntries(
          (Object.keys(LAYER_DEFINITIONS) as Layer[]).map((layer) => [
            layer,
            state.entries.filter((entry) => entry.anchorId === anchor.anchorId && entry.layer === layer).length,
          ]),
        ),
        baseline_kinds: state.baselines
          .filter((baseline) => baseline.anchorId === anchor.anchorId)
          .map((baseline) => baseline.kind)
          .sort(),
        gaps: anchor.gaps,
      })),
  };
}

function controllerPrompt(state: RunState, roundResult: JsonValue | null, forcedPlanReason: string | null): string {
  return [
    "执行 Anchor Explore Controller 的下一次单步决策。",
    "你不读取知识库、不回答证据问题，只规划显著不同的检索 frontier。",
    `TOPIC:\n${state.config.topic}`,
    `CONSTRAINTS:\n${JSON.stringify(state.config.constraints)}`,
    `L1-L6:\n${JSON.stringify(LAYER_DEFINITIONS)}`,
    "价值粗筛轴：exploration > implementation > method；baseline 是并行强制保留轨道。",
    `硬限制：每轮最多 ${state.config.evidenceTasksPerRound} 个任务，Anchor 上限 ${state.config.maxAnchors}，连续 ${state.config.noNewAnchorStop} 轮无新 Anchor 才可收敛。`,
    forcedPlanReason ? `SCRIPT_DECISION: completion request rejected; ${forcedPlanReason}. You must emit ANCHOR_ROUND_PLAN.` : "",
    `CURRENT_STAGE_STATE:\n${JSON.stringify(stageStatePacket(state))}`,
    `PREVIOUS_CONTROLLER_OUTPUT:\n${state.stage1.controllerLastOutput ?? "<none>"}`,
    `ROUND_RESULT:\n${roundResult === null ? "<none>" : JSON.stringify(roundResult)}`,
    [
      "任务 payload 每项必须包含 focus, layer(L1-L6), value_axis(exploration|implementation|method|baseline), avoid[]。",
      "优先寻找新的 workload phase/regime/backend/bottleneck/baseline execution path/metric 组合；也可补齐现有 Anchor 的 layer/baseline gap。",
      "focus 必须是可交给一次短暂 Evidence Worker 的明确检索问题，不要在 focus 中先给结论。",
    ].join("\n"),
  ].filter(Boolean).join("\n\n");
}

function parseDiscoveryTasks(
  payload: JsonValue | null,
  round: number,
  state: RunState,
): DiscoveryTask[] {
  if (!Array.isArray(payload)) return [];
  const previousFocuses = new Set<string>();
  for (const task of Object.values(state.tasks)) {
    if (task.role !== "anchor_evidence_worker" || !task.inputPath || !fs.existsSync(task.inputPath)) continue;
    try {
      const input = JSON.parse(fs.readFileSync(task.inputPath, "utf8")) as Record<string, unknown>;
      previousFocuses.add(normalizeIdentityText(input.focus));
    } catch {
      // A malformed debug copy never becomes canonical state.
    }
  }
  const output: DiscoveryTask[] = [];
  const seen = new Set<string>();
  const remainingBudget = Math.max(0, state.config.maxStage1Tasks - state.stage1.taskCount);
  const limit = Math.min(state.config.evidenceTasksPerRound, remainingBudget);
  for (const rawTask of payload) {
    if (output.length >= limit || !rawTask || typeof rawTask !== "object" || Array.isArray(rawTask)) break;
    const object = rawTask as Record<string, unknown>;
    const focus = String(object.focus ?? "").trim();
    const layer = String(object.layer ?? "").toUpperCase();
    const valueAxis = String(object.value_axis ?? object.valueAxis ?? "").toLowerCase();
    const normalizedFocus = normalizeIdentityText(focus);
    if (
      !focus
      || !/^L[1-6]$/.test(layer)
      || !["exploration", "implementation", "method", "baseline"].includes(valueAxis)
      || seen.has(normalizedFocus)
      || previousFocuses.has(normalizedFocus)
    ) {
      continue;
    }
    seen.add(normalizedFocus);
    output.push({
      taskId: `AE-R${String(round).padStart(2, "0")}-${String(output.length + 1).padStart(2, "0")}`,
      focus,
      layer: layer as Layer,
      valueAxis: valueAxis as ValueAxis,
      avoid: Array.isArray(object.avoid) ? object.avoid.map(String).filter(Boolean) : [],
    });
  }
  return output;
}

function evidencePrompt(state: RunState, task: DiscoveryTask): string {
  return [
    "完成一个 Anchor Explore 原子证据任务；完成后终止。",
    `TOPIC: ${state.config.topic}`,
    `TASK:\n${JSON.stringify({
      task_id: task.taskId,
      focus: task.focus,
      layer: task.layer,
      layer_definition: LAYER_DEFINITIONS[task.layer],
      value_axis: task.valueAxis,
      avoid: task.avoid,
    })}`,
    `允许检索的 vault 根目录：${JSON.stringify(state.config.evidenceRoots)}`,
    [
      "只使用 Obsidian 只读 search/get/list。",
      "将场景、baseline 执行路径、瓶颈、可修改对象、实现入口、方法、约束、指标拆成原子 claim。",
      "每个 claim 必须给出一段真实原文 quote 和 source_path；行号可以尽力估计，脚本会按 quote 校正。",
      "direct 只用于原文直接支持；推导出的加速可能必须标 inferred 并把 statement 写成可证伪假设。",
      "不要用一篇论文的标题或引用列表冒充已经读过的来源。",
    ].join("\n"),
  ].join("\n\n");
}

function curatorStatePacket(state: RunState): JsonValue {
  return {
    anchors: state.anchors
      .filter((anchor) => anchor.status === "accepted")
      .map((anchor) => ({
        anchor_id: anchor.anchorId,
        signature: anchor.signature,
        evidence_refs: anchor.evidenceRefs,
        baseline_ids: anchor.baselineIds,
        entries: state.entries
          .filter((entry) => entry.anchorId === anchor.anchorId)
          .map((entry) => ({
            entry_id: entry.entryId,
            layer: entry.layer,
            role: entry.role,
            claim: entry.claim,
            modifiable_object: entry.modifiableObject,
          })),
        edges: state.edges
          .filter((edge) => edge.anchorId === anchor.anchorId)
          .map((edge) => ({
            edge_id: edge.edgeId,
            from_entry: edge.fromEntryId,
            to_entry: edge.toEntryId,
            relation: edge.relation,
            compatibility: edge.compatibility,
          })),
        gaps: anchor.gaps,
      })),
    global_entities: state.entities.map((entity) => ({
      entity_id: entity.entityId,
      kind: entity.kind,
      name: entity.name,
    })),
  };
}

function curatorPrompt(state: RunState, taskId: string, claimIds: string[]): string {
  const claims = state.claims.filter((claim) => claimIds.includes(claim.claimId));
  return [
    "把给定的、已经过本地 quote 校验的 EvidenceClaim 整理成原子 Anchor delta；完成后终止。",
    `TOPIC: ${state.config.topic}`,
    `TASK_ID: ${taskId}`,
    `L1-L6: ${JSON.stringify(LAYER_DEFINITIONS)}`,
    `CURRENT_CANONICAL_IDS:\n${JSON.stringify(curatorStatePacket(state))}`,
    `VERIFIED_CLAIMS:\n${JSON.stringify(claims)}`,
    [
      "输出 SEMANTIC_PAYLOAD 对象，字段为 entities[] 和 anchors[]。",
      "每个 anchor 必须有 local_id,title,scenario,signature,evidence_refs,baselines[],entries[],edges[],gaps[]。",
      "signature 必须逐项包含 workload,phase,regime,backend,bottleneck,primary_baseline_execution_path,target_metrics[]。",
      "baselines 每项使用 local_id,kind(current_practice|strong_comparison|tool_evaluation|reusable_implementation),name,execution_path,implementation,comparison_scope,evidence_refs,exploration_value。",
      "entries 每项使用 local_id,entity_id/entity_name,layer,role,claim,modifiable_object,applicable_baselines[],preconditions[],expected_effect,evidence_refs,confidence。",
      "edges 必须连接具体 local entry 或已有 canonical entry ID，使用 from_entry,to_entry,relation,interface,compatibility,condition,evidence_refs,confidence。",
      "一个 entry 只保留一个主要 claim；同层允许零到多个 entry。",
      "不要把场景不同、baseline 执行路径不同或指标不可公平比较的对象合成一个 Anchor。",
      "不要凭空补字段。无证据对象放 gaps，不生成伪事实。",
      "baseline 即使 exploration_value=low 也必须保留。",
    ].join("\n"),
  ].join("\n\n");
}

function roundStopReason(state: RunState): string | null {
  if (state.stage1.acceptedAnchorIds.length >= state.config.maxAnchors) return "target_reached";
  if (state.stage1.consecutiveRoundsWithoutNewAnchor >= state.config.noNewAnchorStop) return "no_new_anchor_streak";
  if (state.stage1.round >= state.config.maxStage1Rounds) return "round_budget_exhausted";
  if (state.stage1.taskCount >= state.config.maxStage1Tasks) return "task_budget_exhausted";
  if (state.usage.turns >= state.config.maxTotalTurns) return "turn_budget_exhausted";
  return null;
}

function anchorSpaceVersion(state: RunState): string {
  const accepted = new Set(state.stage1.acceptedAnchorIds);
  return stableHash({
    anchors: state.anchors.filter((anchor) => accepted.has(anchor.anchorId)),
    baselines: state.baselines.filter((baseline) => baseline.anchorId && accepted.has(baseline.anchorId)),
    entries: state.entries.filter((entry) => accepted.has(entry.anchorId)),
    edges: state.edges.filter((edge) => accepted.has(edge.anchorId)),
  } as unknown as JsonValue, 32);
}

async function runEvidenceTask(
  runtime: RuntimeManager,
  store: CanonicalStore,
  state: RunState,
  task: DiscoveryTask,
): Promise<EvidenceTaskResult> {
  const existing = state.tasks[task.taskId];
  if (existing?.status === "committed") {
    return {
      task,
      acceptedClaimIds: state.claims.filter((claim) => claim.taskId === task.taskId).map((claim) => claim.claimId),
      gaps: [],
    };
  }
  const inputPath = path.join(store.workDir, "tasks/anchor_evidence", `${task.taskId}.input.json`);
  const outputPath = path.join(store.workDir, "tasks/anchor_evidence", `${task.taskId}.output.json`);
  atomicWriteJson(inputPath, task as unknown as JsonValue);
  const record: TaskRecord = existing ?? {
    taskId: task.taskId,
    role: "anchor_evidence_worker",
    scopeId: task.taskId,
    status: "pending",
    attempts: 0,
    inputPath,
    outputPath,
    error: null,
  };
  state.tasks[task.taskId] = record;
  record.status = "dispatched";
  record.attempts += 1;
  store.save(state, `dispatch_${task.taskId}`);
  try {
    const handle = await runtime.startAgent("anchor_evidence_worker", task.taskId, false);
    const result = await runtime.runProtocolTurn(handle, evidencePrompt(state, task));
    record.status = "protocol_valid";
    const candidates = result.protocol.payload as JsonValue[];
    const validated = validateClaimCandidates(
      candidates,
      task.taskId,
      state.config.vaultRoot,
      state.config.evidenceRoots,
    );
    for (const claim of validated.accepted) {
      if (!state.claims.some((existingClaim) => existingClaim.claimId === claim.claimId)) {
        state.claims.push(claim);
      }
    }
    state.rejectedClaims.push(...validated.rejected);
    record.status = "committed";
    const gapsRaw = result.protocol.textBlocks.GAPS ?? "";
    const gaps = gapsRaw
      .split(/\r?\n/)
      .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
      .filter(Boolean);
    atomicWriteJson(outputPath, {
      protocol: result.protocol,
      accepted_claim_ids: validated.accepted.map((claim) => claim.claimId),
      rejected_count: validated.rejected.length,
      gaps,
    } as unknown as JsonValue);
    store.save(state, `commit_${task.taskId}`);
    return { task, acceptedClaimIds: validated.accepted.map((claim) => claim.claimId), gaps };
  } catch (error) {
    record.status = "failed_terminal";
    record.error = error instanceof Error ? error.message : String(error);
    store.save(state, `fail_${task.taskId}`);
    return { task, acceptedClaimIds: [], gaps: [record.error] };
  }
}

async function runCuratorBatch(
  runtime: RuntimeManager,
  store: CanonicalStore,
  state: RunState,
  taskId: string,
  claimIds: string[],
  round: number,
): Promise<void> {
  const existing = state.tasks[taskId];
  if (existing?.status === "committed") return;
  const inputPath = path.join(store.workDir, "tasks/anchor_curation", `${taskId}.input.json`);
  const outputPath = path.join(store.workDir, "tasks/anchor_curation", `${taskId}.output.json`);
  atomicWriteJson(inputPath, { task_id: taskId, claim_ids: claimIds });
  const record: TaskRecord = existing ?? {
    taskId,
    role: "anchor_curator_worker",
    scopeId: taskId,
    status: "pending",
    attempts: 0,
    inputPath,
    outputPath,
    error: null,
  };
  state.tasks[taskId] = record;
  record.status = "dispatched";
  record.attempts += 1;
  store.save(state, `dispatch_${taskId}`);
  try {
    const handle = await runtime.startAgent("anchor_curator_worker", taskId, false);
    const result = await runtime.runProtocolTurn(handle, curatorPrompt(state, taskId, claimIds));
    record.status = "protocol_valid";
    const merged = mergeCuratorDelta(state, result.protocol.payload!, round);
    record.status = "committed";
    atomicWriteJson(outputPath, {
      protocol: result.protocol,
      merge_result: merged,
    } as unknown as JsonValue);
    store.recordEvent("curator_merge", merged as unknown as JsonValue);
    store.save(state, `commit_${taskId}`);
  } catch (error) {
    record.status = "failed_terminal";
    record.error = error instanceof Error ? error.message : String(error);
    store.save(state, `fail_${taskId}`);
  }
}

function splitClaims(claimIds: string[], maximum = 40): string[][] {
  const batches: string[][] = [];
  for (let index = 0; index < claimIds.length; index += maximum) {
    batches.push(claimIds.slice(index, index + maximum));
  }
  return batches;
}

async function finalizeController(
  runtime: RuntimeManager,
  state: RunState,
  handle: AgentHandle,
  stopReason: string,
): Promise<void> {
  const prompt = [
    "脚本已按确定性条件结束 Stage 1。输出 ANCHOR_STAGE_COMPLETE；不要再规划任务。",
    `STOP_REASON: ${stopReason}`,
    `CURRENT_STAGE_STATE:\n${JSON.stringify(stageStatePacket(state))}`,
    `ANCHOR_SPACE_VERSION: ${state.stage1.anchorSpaceVersion}`,
  ].join("\n\n");
  try {
    const result = await runtime.runProtocolTurn(handle, prompt);
    state.stage1.controllerLastOutput = result.protocol.rawText;
  } catch {
    // The script's stop decision remains authoritative; raw failure is already logged.
  }
}

export async function runAnchorStage(
  state: RunState,
  store: CanonicalStore,
  runtime: RuntimeManager,
): Promise<void> {
  if (state.stage1.status === "complete") return;
  state.status = "running";
  state.stage1.status = "running";
  store.save(state, "stage1_started");
  let controller = await runtime.persistentAgent("anchor_stage_controller", "stage1");
  let lastRoundResult: JsonValue | null = null;

  while (true) {
    const existingStop = roundStopReason(state);
    if (existingStop) {
      state.stage1.stopReason = existingStop;
      state.stage1.anchorSpaceVersion = anchorSpaceVersion(state);
      state.stage1.status = "complete";
      await finalizeController(runtime, state, controller, existingStop);
      store.save(state, "stage1_complete");
      return;
    }

    const round = state.stage1.round + 1;
    const planPath = path.join(store.workDir, "tasks/anchor_evidence", `round_${String(round).padStart(2, "0")}_plan.json`);
    let tasks: DiscoveryTask[] = [];
    if (fs.existsSync(planPath)) {
      tasks = JSON.parse(fs.readFileSync(planPath, "utf8")) as DiscoveryTask[];
    } else {
      let forcedReason: string | null = null;
      for (let attempt = 0; attempt < 2 && tasks.length === 0; attempt += 1) {
        const decision = await runtime.runProtocolTurn(
          controller,
          controllerPrompt(state, lastRoundResult, forcedReason),
        );
        controller = decision.handle;
        state.stage1.controllerLastOutput = decision.protocol.rawText;
        if (decision.protocol.marker === "ANCHOR_ROUND_PLAN") {
          tasks = parseDiscoveryTasks(decision.protocol.payload, round, state);
        } else {
          forcedReason = [
            `accepted=${state.stage1.acceptedAnchorIds.length}<${state.config.maxAnchors}`,
            `no_new_streak=${state.stage1.consecutiveRoundsWithoutNewAnchor}<${state.config.noNewAnchorStop}`,
            "budget remains",
          ].join(", ");
        }
      }
      if (tasks.length === 0) {
        throw new Error(`controller produced no valid nonduplicate tasks for round ${round}`);
      }
      atomicWriteJson(planPath, tasks as unknown as JsonValue);
    }

    const evidenceResults = await mapWithConcurrency(
      tasks,
      state.config.anchorEvidenceConcurrency,
      (task) => runEvidenceTask(runtime, store, state, task),
    );
    const claimIds = [...new Set(evidenceResults.flatMap((result) => result.acceptedClaimIds))].sort();
    const batches = splitClaims(claimIds);
    await mapWithConcurrency(
      batches.map((batch, index) => ({
        taskId: `AC-R${String(round).padStart(2, "0")}-B${String(index + 1).padStart(2, "0")}`,
        claimIds: batch,
      })),
      state.config.curatorConcurrency,
      (batch) => runCuratorBatch(runtime, store, state, batch.taskId, batch.claimIds, round),
    );

    const newAcceptedAnchorIds = state.anchors
      .filter((anchor) => anchor.status === "accepted" && anchor.firstSeenRound === round)
      .map((anchor) => anchor.anchorId)
      .sort();
    state.stage1.round = round;
    state.stage1.taskCount = Object.values(state.tasks)
      .filter((task) => task.role === "anchor_evidence_worker" && task.status === "committed")
      .length;
    state.stage1.consecutiveRoundsWithoutNewAnchor = newAcceptedAnchorIds.length === 0
      ? state.stage1.consecutiveRoundsWithoutNewAnchor + 1
      : 0;
    lastRoundResult = {
      round,
      executed_tasks: tasks.map((task) => task.taskId),
      accepted_claim_count: claimIds.length,
      new_accepted_anchor_ids: newAcceptedAnchorIds,
      accepted_anchor_count: state.stage1.acceptedAnchorIds.length,
      consecutive_rounds_without_new_anchor: state.stage1.consecutiveRoundsWithoutNewAnchor,
      evidence_gaps: evidenceResults.flatMap((result) => result.gaps),
      frontier_ids: state.stage1.acceptedAnchorIds,
    };
    store.recordEvent("anchor_round_result", lastRoundResult);
    store.save(state, `stage1_round_${round}_complete`);
  }
}

