import { DirectionExperimentStore } from "./store.ts";
import type {
  DirectionGoalRecord,
  DirectionRunFile,
  LabGoalInvocationRecord,
} from "./types.ts";

export function buildDecisionPrompt(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  stateSnapshotRef: string,
  historySnapshotRef: string,
  runtimeEnvelopeRef: string,
  correctionSuffix = "",
): string {
  const state = store.readState();
  return [
    "使用 $direction-experiment-decision Skill；本 run 的冻结 Skill 与方法文件是本次执行权威。",
    `- Frozen Skill: ${store.absolute(run.skills.decision.path)}`,
    `- Frozen method: ${store.absolute(run.skills.decisionMethod.path)}`,
    "",
    "你是 fresh Experiment Decision Turn。理解冻结 Direction、完整实验轨迹和最终验证需求，只选择 Script 允许的下一分支。Script 不理解科研语义。",
    "",
    "冻结研究输入：",
    `- Direction: ${store.absolute(run.inputs.directionResult.path)}`,
    `- Parent Anchor: ${store.absolute(run.inputs.parentAnchorResult.path)}`,
    `- Source review: ${store.absolute(run.inputs.sourceReviewResult.path)}`,
    `- Readable target: ${store.absolute(run.inputs.directionTarget.path)}`,
    `- Evidence manifest: ${store.absolute(run.inputs.evidenceManifest.path)}`,
    `- Experiment policy: ${store.absolute(run.inputs.experimentPolicy.path)}`,
    "",
    "当前执行事实：",
    `- State snapshot: ${store.absolute(stateSnapshotRef)}`,
    `- Complete indexed trajectory: ${store.absolute(historySnapshotRef)}`,
    `- Lab runtime envelope: ${store.absolute(runtimeEnvelopeRef)}`,
    `- Active contract: ${absoluteOrNone(store, state.activeContractRef)}`,
    `- Latest Lab result: ${absoluteOrNone(store, state.latestLabResultRef)}`,
    `- Latest checkpoint: ${absoluteOrNone(store, state.latestCheckpointRef)}`,
    `- Latest independent judgment: ${absoluteOrNone(store, state.latestJudgeRef)}`,
    "",
    "分支优先级：尚未独立审阅的新结果先 RUN_JUDGE；已有 Judgment 已回答问题时完成、拒绝或返回 Learning；只有缺少决定性测量时才 RUN_LAB。不要在公平负结果后做局部 regime hunting。",
    "RUN_LAB 每轮只回答一个会改变下一决策的不确定性。合同必须包含按优先级排列的 stopConditions、条件化 completionEvidence 和不超过运行包络的 estimatedMinutes；不得把 calibration、confirmation、performance 或新环境部署合并成依赖前序结果才能定义的超大流水线。",
    "弱代理局部正结果不能自动满足依赖真实模型、真实数据或本地性能的最终主张。环境失败或无效实验不能形成科学拒绝。改变优化对象、Parent Anchor 或核心因果 lever 时返回 Learning Flow。",
    "",
    "允许决策：RUN_LAB | RUN_JUDGE | COMPLETE_SUPPORT | COMPLETE_REJECT | RETURN_TO_LEARNING | BLOCKED",
    "evidenceScope：DESIGN_AUDIT_ONLY | WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY",
    "只在 final_answer 输出一个 JSON 对象：",
    '{"decision":"RUN_LAB | RUN_JUDGE | COMPLETE_SUPPORT | COMPLETE_REJECT | RETURN_TO_LEARNING | BLOCKED","evidenceScope":"...","reason":"简短理由和结论边界","experimentContract":{"objective":"本轮唯一决策性不确定性","comparison":"baseline、variant 唯一变化与必要消融","conditions":"载体、数据、指标、阶段入口和 guard","stopConditions":["terminal stop，按优先级排列"],"estimatedMinutes":120,"allowedWeakening":["明确边界"],"forbiddenWeakening":["核心因果维度和 guard"],"completionEvidence":"分别说明命中 stop、未命中、无效和证据不足时的最小产物"},"reviewFocus":null}',
    "RUN_LAB 才填写 experimentContract；RUN_JUDGE 才填写精简 reviewFocus；其他决策两者均为 null。",
    correctionSuffix,
  ].join("\n");
}

export function buildLabPrompt(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  goal: DirectionGoalRecord,
  invocation: LabGoalInvocationRecord,
): string {
  return [
    "使用 $direction-lab-goal Skill；本 run 的冻结 Skill 与方法文件是本次执行权威。",
    `- Frozen Skill: ${store.absolute(run.skills.lab.path)}`,
    `- Frozen method: ${store.absolute(run.skills.labMethod.path)}`,
    "",
    `你是 Cycle ${goal.cycle}、Invocation ${invocation.ordinal} 的持久 Direction Lab Goal。只执行一个冻结原子合同；不选择下一分支，不决定整个 Direction。`,
    "",
    "冻结输入与运行边界：",
    `- Direction: ${store.absolute(run.inputs.directionResult.path)}`,
    `- Parent Anchor: ${store.absolute(run.inputs.parentAnchorResult.path)}`,
    `- Source review: ${store.absolute(run.inputs.sourceReviewResult.path)}`,
    `- Readable target: ${store.absolute(run.inputs.directionTarget.path)}`,
    `- Evidence manifest: ${store.absolute(run.inputs.evidenceManifest.path)}`,
    `- Experiment policy: ${store.absolute(run.inputs.experimentPolicy.path)}`,
    `- Experiment contract: ${store.absolute(goal.contractRef)}`,
    `- Cycle binding: ${store.absolute(goal.bindingRef)}`,
    `- Complete trajectory: ${store.absolute("history.jsonl")}`,
    `- Cycle workspace: ${store.absolute(goal.cycleRef)}`,
    `- Isolated mutable source: ${store.absolute(goal.cycleSourceRef)}`,
    `- Shared content-addressed cache: ${run.storage.sharedCacheRoot}`,
    `- Checkpoint: ${store.absolute(goal.checkpointRef)}`,
    `- Required final result: ${store.absolute(goal.outputRef)}`,
    `- Invocation deadline: ${invocation.deadlineAt}`,
    `- Result/checkpoint reserve: ${run.budgets.labResultReserveMs} ms`,
    "",
    "先读取现有 checkpoint；它是恢复事实，旧对话中冲突内容作废。将合同翻译成按退出路径区分的 checklist。每个昂贵 arm、全量 sweep 或长命令前执行 Stop Gate：若已命中合同 stop condition、合同冲突、forbidden weakening 会被改变，或剩余时间不足以打包最小结果，立即停止下游实验。",
    "命中 stop condition 不是全局 SUPPORT/REJECT。保存当前已验证产物、排除 partial shard、写出最窄 observation，交 Judge 审阅。正确早停时不得为了形式完整继续 confirmation/performance。",
    "长实验按固定 shard 执行；机械 runner 记录 heartbeat、完成标记和 hash。只聚合完整 shard，恢复时只补缺失 shard。每次阶段完成、实质错误、Stop Gate 或长时间无进展时原子更新 checkpoint.json。",
    "result.md 只在已经可独立审阅时生成：先写同目录临时文件，再原子 rename 到 Required final result。必须包含 contract revision/hash、carrier、实际范围、baseline/variant、完成 arm/shard、命中的 stop condition、guards、排除的 partial artifacts、最窄 observation 和 code/commands/raw/analysis/freeze 引用。",
    "真实性能必须精确命名：不含媒体预处理的是模型路径延迟；concurrency=1 closed-loop 是串行服务率。小样本无观测差异不能被描述成总体不确定性为零。",
    "最终回答只概括最窄观察并引用结果或 checkpoint，不输出调度决策。",
  ].join("\n");
}

export function buildJudgePrompt(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  stateSnapshotRef: string,
  historySnapshotRef: string,
  requestRef: string,
  correctionSuffix = "",
): string {
  const state = store.readState();
  return [
    "使用 $direction-evidence-judge Skill；本 run 的冻结 Skill 与评判方法是本次执行权威。",
    `- Frozen Skill: ${store.absolute(run.skills.judge.path)}`,
    `- Frozen method: ${store.absolute(run.skills.judgeContract.path)}`,
    "",
    "你是 fresh Evidence Judge Turn。只独立评判当前合同与指定证据，不选择下一 Lab、不授权弱化、不终止整个 Flow。",
    "",
    "冻结研究输入：",
    `- Direction: ${store.absolute(run.inputs.directionResult.path)}`,
    `- Parent Anchor: ${store.absolute(run.inputs.parentAnchorResult.path)}`,
    `- Source review: ${store.absolute(run.inputs.sourceReviewResult.path)}`,
    `- Readable target: ${store.absolute(run.inputs.directionTarget.path)}`,
    `- Evidence manifest: ${store.absolute(run.inputs.evidenceManifest.path)}`,
    `- Experiment policy: ${store.absolute(run.inputs.experimentPolicy.path)}`,
    "",
    "本次审阅：",
    `- Review request: ${store.absolute(requestRef)}`,
    `- Active contract: ${absoluteOrNone(store, state.activeContractRef)}`,
    `- Latest Lab result: ${absoluteOrNone(store, state.latestLabResultRef)}`,
    `- Latest checkpoint: ${absoluteOrNone(store, state.latestCheckpointRef)}`,
    `- State snapshot: ${store.absolute(stateSnapshotRef)}`,
    `- Complete indexed trajectory: ${store.absolute(historySnapshotRef)}`,
    "",
    "先核查 result 声明的 stop condition 是否由正确绑定的产物命中。合同规定命中后停止时，不得因 confirmation/performance 未运行而判不完整；但只能给出该策略定义、预算域、数据、模型、阈值和统计规则内的窄结论。安装失败、无效实现和未触发的无效代理仍不是科学负结果。",
    "核查 baseline/variant、唯一变化、弱化因果接口、raw/trace/统计和 guard。明确性能是否包含预处理、并发模型、batch、到达过程、硬件与时钟控制；小样本零差异不得自动解释为总体确定。不要输出后续调度。",
    "只在 final_answer 输出一个 JSON 对象：",
    '{"assessment":"VALID_POSITIVE | VALID_NEGATIVE | INCONCLUSIVE | INVALID","evidenceScope":"DESIGN_AUDIT_ONLY | WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY","reason":"简短判断和边界","remainingUncertainty":"会影响最终结论的主要未决项；没有则写 NONE"}',
    correctionSuffix,
  ].join("\n");
}

export function decisionDeveloperInstructions(): string {
  return [
    "You are one fresh Experiment Decision Turn controlled by a deterministic Script.",
    "Read and follow the run-local frozen Skill and method named in the prompt; they override conflicting active-skill text.",
    "Own one-question atomic contracts, controlled weakening, runtime feasibility, trajectory-aware scheduling, and global completion.",
    "Do not implement code, run measurements, replace Judge evidence auditing, edit Script authority, or change the original causal identity.",
    "Only final_answer is protocol. Return exactly one JSON object using the allowed decision and conditional fields.",
  ].join("\n");
}

export function labDeveloperInstructions(): string {
  return [
    "You are one persistent Direction Lab Goal controlled by a deterministic Script.",
    "Read and follow the run-local frozen Skill and method named in the prompt; they override conflicting active-skill text.",
    "Execute only the frozen atomic contract. Evaluate its explicit stop conditions before every expensive action.",
    "Use the cycle-isolated mutable source and content-addressed shared caches. Preserve complete shards and exclude partial shards.",
    "Atomically maintain checkpoint.json and commit result.md only when it is independently reviewable.",
    "Do not choose the next branch or overall verdict.",
  ].join("\n");
}

export function judgeDeveloperInstructions(): string {
  return [
    "You are one fresh independent Evidence Judge Turn controlled by a deterministic Script.",
    "Read and follow the run-local frozen Skill and method named in the prompt; they override conflicting active-skill text.",
    "Audit only the frozen contract, explicit early-stop boundary, requested evidence, raw artifacts, comparison validity, and evidence scope.",
    "Do not schedule Lab, authorize weakening, revise the Direction, run experiments, or finish the workflow.",
    "Only final_answer is protocol. Return exactly one JSON object with assessment, evidenceScope, reason, and remainingUncertainty.",
  ].join("\n");
}

function absoluteOrNone(store: DirectionExperimentStore, ref: string | null): string {
  return ref ? store.absolute(ref) : "NONE";
}
