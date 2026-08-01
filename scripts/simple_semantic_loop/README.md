# Simple Semantic Loop 使用说明

当前实现采用 format version 7，通信契约由以下设计统一定义：

- [`03_script_agent_message_and_storage_contract_design.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/03_script_agent_message_and_storage_contract_design.md)
- [`04_script_agent_json_communication_contract_inventory.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/04_script_agent_json_communication_contract_inventory.md)
- [`05_minimal_controller_validation_and_semantic_handoff_design.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/05_minimal_controller_validation_and_semantic_handoff_design.md)
- [`06_outer_loop_memory_trajectory_and_atomic_direction_design.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/06_outer_loop_memory_trajectory_and_atomic_direction_design.md)
- [`07_frozen_decision_snapshots_semantic_convergence_and_runtime_delta_dedup_design.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/07_frozen_decision_snapshots_semantic_convergence_and_runtime_delta_dedup_design.md)
- [`08_official_snapshot_round_lease_convergence_probe_and_delta_batching_design.md`](../../draft/learning_workflow_scheduler_agent_refactor_plans/08_official_snapshot_round_lease_convergence_probe_and_delta_batching_design.md)

## 架构

Controller 是没有业务智能的持久状态机。Decision、Worker、Reviewer 都是
fresh one-turn Agent：

| Agent | effort | 唯一职责 |
|---|---:|---|
| Decision | `max` | 读取目标、待决结论、压缩记忆、轨迹和分支后果，执行全局外循环并选择一个允许分支 |
| Worker | `high` | 创建或深化一个 Anchor/Direction，返回最小充分的 W01 |
| Reviewer | `high` | 独立审阅一个 W01、记录对象局部 query gaps，返回 R01 |

正常 Loop：

```text
START → Worker → Reviewer → Decision

RUN_WORKER
  → commit pending
  → Worker → Reviewer → Decision

RUN_REVIEWER
  → commit pending
  → Reviewer → commit current R01
  → Worker → Reviewer → Decision

FINISH_WORKFLOW
  → commit pending
  → mechanical requirement check
  → deterministic report
  → END
```

语义重试：

```text
RETRY_WORKER
  → supersede pending Worker/Reviewer
  → same Worker TaskBinding → Reviewer → Decision

RETRY_REVIEWER
  → retain pending Worker
  → same Reviewer TaskBinding → Decision
```

语义重试字面量在存在对应 pending 时始终可选。若预算耗尽，Controller 保留
pending、返回 `FAILED`，不会为了继续运行而提交 Decision 已判定错误的结果。

辅助 Agent 不是顶层节点。Worker 或 Reviewer 可在自己的 Turn 内按需使用，
但当前 Turn 必须汇总为唯一 W01 或 R01。

Worker 和 Reviewer 可在读取实际 Topic、Task 和对象后独立选择 0–2 个已安装
领域专家 Skill；没有紧密匹配时使用 0 个。Decision 不加载领域专家 Skill。
专家 Skill 只增强方法，不是当前结论的证据。

## 运行前检查

联网检查 Provider、模型 effort 和 Skill：

```bash
cd /data3/paper_analysis
node scripts/simple_semantic_loop.ts doctor
```

只检查本地 Skill 和 Ref 模板：

```bash
node scripts/simple_semantic_loop.ts doctor --no-provider
```

## 初始化

以下 Topic 只是调用示例，不是 Skill 的预设主题：

```text
多模态推理加速，优先优化延迟，保证较高吞吐
```

为 format version 7 使用一个新的 work directory：

```bash
node scripts/simple_semantic_loop.ts init \
  --topic '多模态推理加速，优先优化延迟，保证较高吞吐' \
  --objective '从本地多维科研知识库识别并形成多模态推理的可验证加速潜力；优先降低延迟，同时保持较高吞吐和必要质量约束。' \
  --acceptance 'Topic 的 6L 空间由未被拒绝的 Anchor 集合动态定义。' \
  --acceptance '每个最终 Anchor 和 Direction 都必须获得独立 Reviewer PASS。' \
  --acceptance '每个最终 Anchor 至少有一个 Direction；Direction 必须明确 baseline change、机制、预期影响、权衡、失败条件和测量计划。' \
  --acceptance '需要新实验的内容只形成测量计划，不执行实验。' \
  --max-rounds 8 \
  --idle-timeout-ms 300000 \
  --hard-timeout-ms 900000 \
  --interrupt-grace-ms 15000 \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v7
```

`init` 写入：

- `workflow_goal.json`：所有 Agent 共用的不可变需求；
- Controller 的 Run、State、Object、Round 和 Ref 记录；
- 空的 Script 派生进展轨迹与 observation/checkpoint 目录；
- 初始 `Worker → Reviewer → Decision` 序列。

它不会启动 Agent，也不会覆盖已有 `run.json`。

## 启动

只读 sandbox：

```bash
node scripts/simple_semantic_loop.ts run \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v7
```

给予 fresh Agent Turn 完整文件系统权限：

```bash
node scripts/simple_semantic_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v7
```

两种模式都固定：

```text
approvalPolicy = never
```

不加 `--yolo` 时 sandbox 为 `read-only`；加 `--yolo` 时为
`danger-full-access`。YOLO 只改变 Provider 工具权限，不改变通信协议，也不会
使 Agent 输出成为权威状态。工具作用可能早于输出校验发生。

Controller 不维护按角色 tool allowlist。Skill、T01 和 Result Ref 区分角色
职能；Provider 环境决定实际可用工具。

## 实时控制台

`run`、`resume` 和 `recover-runtime` 默认实时转发到 stderr。关键运输标签是：

- `agent:stream-start`：首个正文 delta 到达，消息仍可能不完整；
- `agent:message-complete`：一个完整 Agent message item 已结束；
- `turn:complete`：Provider Turn 进入终态；
- `turn:timeout`：显示 idle/hard timeout、`NONE|PARTIAL|COMPLETE`、runtime
  与 partial Ref；
- tool 开始/完成、耗时、token 和 tool 调用数也会实时显示。

O01 写到 stdout。只需要机器输出时使用 `--quiet`。

一个 Codex Turn 可能在工具调用前后产生多个完整 `agentMessage`。控制台会
显示全部消息。RuntimeLog 保存非 delta 原始事件；每个
`item/agentMessage/delta` 仍立即转发到控制台，但持久化端按“最多 100 ms 或累计
2,048 字符”合并为规范化 `output_delta`，并按相同顺序追加到
`partial_output.txt`。这既保留实时体验与精确恢复文本，也避免逐 token JSONL 和
Turn 元数据写放大；delta 不会再同时写一份含相同文本的 `raw_event`。W01/R01
协议校验只读取唯一的 `phase=final_answer` 消息，`phase=commentary` 只用于进度。旧
Provider 只有在整个 Turn 恰好一个非空 phase-unknown 消息时才兼容；其他
歧义不会由 Script 猜测。Runtime 不再把进度 JSON 和最终 JSON 拼成一个多值
文本。

## Agent 通信

Agent 可见 JSON 只有：

```text
G01  workflow_goal.json
T01  turn_task.json
D01  decision_context.json
W01  WORK_RESULT
R01  REVIEW_RESULT
E01  output_error_report.json
```

调用者接收 O01。Turn、Object、Round、Event、Runtime、TaskBinding 和
ValidationAudit 都是 Controller 内部记录。

D01 只包含一个 `observationRef`。format v7 的每个 Context 固定包含
`decision_context.json`、`decision_observation.json`、
`research_memory_snapshot.json` 和 `progress_trajectory_snapshot.jsonl`；
observation 的两个 Ref 只指向同目录快照。runtime retry 和 output correction
复用整个 Context，后续轮次与 `checkpoint` 都不能改写这些历史输入。
`observations/research_memory.json` 和全局 trajectory 仅供 status、checkpoint
和人类查看。原始 W01/R01 仍是语义权威，Worker/Reviewer 的 T01 不内联这些
全局观察。

`CREATE_ANCHOR` T01 可额外包含 `inputs.researchMemory`，指向为本 Task 冻结的
compact research-memory 快照。Worker 用它比较已接受对象、待修订对象、拒绝经验
和动态 6L coverage；它只是结论索引，不替代实际来源证据。机械闭合后，Script
会把新 Anchor Task 标为有界收敛探测，但不要求 Worker 必须制造一个对象。

修订对象的 Reviewer T01 可额外包含 `inputs.previousReview`。它只帮助 Reviewer
确认上一轮 correction boundary；当前 `reviewTarget` 始终是唯一审阅对象。

Worker/Reviewer Prompt：

```text
使用 $<skill-name>

本次任务：<absolute-turn-task-path>
Decision guidance：<原样文字或“无”>

按照 Skill 指定的 Result Ref 输出一个 JSON 对象。
```

结果 Ref：

```text
work-result-anchor-v2
work-result-direction-v2
review-result-v2
```

Decision Prompt 注入允许字面量，返回：

```text
decision = RUN_WORKER
guidance = 关注当前结论尚未覆盖的性能机制，不改变 Script 绑定。
```

Script 只提取唯一且在本次允许集合内的 Decision 字段。guidance 是完全不透明
的可选文本：Script 原样保存并转发，不要求其中出现 Ref，也不用它选择
create/deepen、对象类型、目标或审阅角度。

## 不可信输出与重试

Script 假定 Agent 可能返回：

1. Decision 协议错误；
2. JSON 解析错误；
3. 缺少或使用未知 `workOutcome` / `reviewVerdict` 核心字面量；
4. JSON 和核心字段合法，但 Ref 字段、跨字段关系或专业语义错误；
5. 会错误关闭需求或破坏 workflow 的 Worker/Reviewer follow 错误。

前三类由 Script 写 E01，并用同一冻结输入创建新 Attempt：

```text
[OUTPUT_CORRECTION]
上次输出：<absolute-path>
错误报告：<absolute-output-error-report-path>
正确 Ref：<registered-ref-name>

重做同一任务并返回完整结果。
```

后两类正常进入 Worker → Reviewer → Decision。Reviewer 负责审阅 Worker
正文；Decision 复核 Worker/Reviewer 是否遵循各自 Ref、Task 和 Goal，并可
选择 `RETRY_WORKER` 或 `RETRY_REVIEWER`。Decision Skill 要求语义 retry
guidance 说明错误、闭合影响、正确 Result Ref 和纠正预期，但 Script 不解释
或强制该自然语言。

Worker/Reviewer Turn 不发送完整正文 Provider `outputSchema`。Script 在线
只要求一个 JSON object，以及本角色合法的 `workOutcome` 或
`reviewVerdict`。完整 Anchor/Direction/Review JSON Schema 作为 Result Ref
推荐模板保留；模板偏差只写入 `validation_audit.json` 的 `advisories`，不
触发 `INVALID_OUTPUT`、不消耗 output-correction retry，也不改变 allowed
decisions。

Anchor 仍推荐固定 `L1`–`L6` string-or-null 字段；query gap 仍推荐单个
`dimension`，多种 gap 用多个对象表达。这些设计帮助 Agent 和后续语义审阅，
不构成 Script 对正文的强制假设。

Direction 应表达一个最小可检验主要变化。可独立切换的变化不捆绑归因；只有
技术上不可分的联合包可以作为一个 Direction，并只声明 package-level effect。
Worker 使用紧凑、确定性的生成/采样规则代替未来巨型 manifest；Reviewer 和
Decision 负责这项语义审查，Script 不增加字符数或数组长度 Gate。

Reviewer 的 `queryGaps` 只描述当前对象的 verdict-changing 未知项；空数组不
代表 Topic 已饱和。对开放探索 Goal，Decision 通常应在完成前看到一次受限的
`CREATE_ANCHOR → BLOCKED_NO_RESULT → REJECT` 安静扩展结论，并自行结合 Goal、
动态 6L 覆盖和轨迹判断信息增益。最近一次成功新建 Anchor 后，需要一次新的可信
探测；该探测若由 Worker 返回可信 `BLOCKED_NO_RESULT`、Reviewer 返回 `REJECT`，
通常已经足够，不应无依据要求连续多次空探测。显式限定对象数量或子空间的 Goal
可按用户边界完成。这是 Decision 的语义方法，不是 Script 新 Gate；轮次预算仍
只产生 `PAUSED`。

Turn 状态只使用：

```text
RUNNING
INVALID_OUTPUT
PENDING_DECISION
COMMITTED
SUPERSEDED_BY_RETRY
RUNTIME_FAILED
```

## 持久化与恢复

```text
<work-dir>/
├── workflow_goal.json
├── run.json
├── state.json
├── events.jsonl
├── ref_catalog.json
├── bindings/
├── tasks/
├── contexts/<decision-context-id>/
│   ├── decision_context.json
│   ├── decision_observation.json
│   ├── research_memory_snapshot.json
│   └── progress_trajectory_snapshot.jsonl
├── turns/<turn-id>/
│   ├── turn.json
│   ├── prompt.txt
│   ├── partial_output.txt       # 可选；未完成运输证据，永不作为结果
│   ├── output.txt               # 仅完整、唯一的协议消息
│   ├── control.json             # Script 提取的核心控制投影
│   ├── validation_audit.json
│   ├── output_error_report.json  # 仅无效输出
│   ├── runtime_error.json        # 仅运行时失败/超时
│   └── runtime.jsonl
├── results/
├── objects/index.json
├── rounds/
├── authorizations/rounds/       # resume 追加的不可变轮次授权
├── observations/
│   ├── progress_trajectory.jsonl
│   ├── research_memory.json
│   └── checkpoints/
├── recoveries/
└── final/
    ├── report.md
    ├── manifest.json
    └── outcome.json
```

中断恢复：

- 已捕获唯一完整消息：本地重放并校验，不再次调用 Provider；这也覆盖“已校验但
  尚未来得及写入 pending”的中断窗口；
- 只有 delta 或未完成 item：写 `partial_output.txt`，原 Turn 记为
  `RUNTIME_FAILED`，同一 TaskBinding/DecisionContext 启动全新 Attempt；
- 相同对象 revision 的相同提交幂等；不同引用冲突时失败关闭。

每个角色同时使用 idle timeout 与 hard cap；有意义的 Agent/tool/usage 活动只
重置 idle timer，不能延长 hard cap。timeout 先落盘 snapshot，再 interrupt，
最后等待 grace。自动 runtime retry 的 Prompt 只附加旧 Turn、失败类型与 partial
Ref；它明确要求从头返回完整精简结果，不续写半截 JSON。

format version 7 Controller 不运行、恢复、checkpoint、pause 或 cancel v5/v6
目录。`status`、`events`、`validate` 和已有最终报告读取仍可用于只读审计；v6
仍按其冻结 Decision 快照规则校验，任何旧目录都不会被原地迁移或重解释。需要
继续研究时初始化新的 v7 work directory。

`maxRounds` 是初始化时的首批授权轮数，不是永久总上限。授权耗尽时，Controller
先建立下一 Round 和固定序列，再以
`pauseKind=ROUND_BUDGET_EXHAUSTED` 返回 `PAUSED`，不再调用 Agent。一次恢复可
显式增加多轮：

```bash
node scripts/simple_semantic_loop.ts resume --yolo \
  --additional-rounds 8 \
  --work-dir <run-dir>
```

省略 `--additional-rounds` 时，默认再授权初始化的 `maxRounds` 轮。每次授权写入
`authorizations/rounds/` 的不可变记录；不会每完成一轮再次要求人工 resume。
操作员主动 `pause` 使用 `pauseKind=OPERATOR_REQUESTED`，普通 `resume` 不会暗中
增加轮次，除非显式给出 `--additional-rounds`。

只有 `failureKind=RUNTIME_RETRY_EXHAUSTED` 的 `FAILED` 可由用户显式授权一个
额外的新 Turn：

```bash
node scripts/simple_semantic_loop.ts recover-runtime --yolo \
  --work-dir <run-dir> \
  --recovery-token '<一次性操作令牌>' \
  --hard-timeout-ms 1200000
```

令牌只保存 SHA-256，重复令牌幂等；recovery 写独立 immutable record，不改写
旧 Turn、旧 partial 或 `run.json`。其他 FAILED 原因仍拒绝恢复。

## 状态命令

```bash
node scripts/simple_semantic_loop.ts status --work-dir <run-dir>
node scripts/simple_semantic_loop.ts events --work-dir <run-dir>
node scripts/simple_semantic_loop.ts events --json --work-dir <run-dir>
node scripts/simple_semantic_loop.ts validate --work-dir <run-dir>
node scripts/simple_semantic_loop.ts resume --yolo --additional-rounds 8 --work-dir <run-dir>
node scripts/simple_semantic_loop.ts recover-runtime --yolo --work-dir <run-dir> --recovery-token TOKEN
node scripts/simple_semantic_loop.ts checkpoint --work-dir <run-dir>
node scripts/simple_semantic_loop.ts pause --work-dir <run-dir>
node scripts/simple_semantic_loop.ts cancel --work-dir <run-dir>
node scripts/simple_semantic_loop.ts render --work-dir <run-dir>
```

仅 `FINISHED` 返回退出码 `0`；`PAUSED` 或 `FAILED` 返回 `2`；命令错误返回
`1`。`status` 同时显示 Controller 的 `pauseKind`、下一角色/动作、机械完成阻塞项、
已授权至哪一轮及剩余授权轮数，并显示压缩观察、最近 runtime failure 和 recovery eligibility；
`checkpoint` 只生成可重建的人类观察报告。`render` 不绕过 Loop，只返回已由
Controller 生成的报告路径。

## 测试

不调用付费 Agent：

```bash
node --test scripts/simple_semantic_loop/tests/*.test.ts
```

覆盖：

- 两条正常 Decision 分支；
- Anchor/Direction requirement 闭合；
- Decision 行协议；
- W01/R01 最小核心字段 gate；
- Ref-template 偏差只产生 advisory；
- Worker/Reviewer dispatch 不发送完整 outputSchema；
- Provider invalid-request 立即失败且不消耗 runtime retry；
- E01 同绑定格式重试；
- Worker/Reviewer 语义重试；
- Decision 协议重试；
- captured output 重放；
- 无 captured output 的 fresh Attempt 恢复；
- delta 实时转发、合并持久化，以及 partial 不进入结果/研究记忆；
- idle/hard timeout、activity reset、snapshot-before-interrupt 和 grace；
- runtime retry 的同绑定与 DecisionContext 复用；
- Context-local memory/trajectory 快照、write-once、revision/tail 对齐与
  checkpoint 后字节不变；
- Reviewer 修订任务的 `previousReview` correction boundary；
- v5/v6 只读审计兼容；
- 一次 resume 的多轮授权、不可变授权记录和显式 pause kind；
- 带冻结 research memory 的有界 Anchor 收敛探测；
- runtime budget 用尽后的显式、幂等 recovery；
- Script 派生 trajectory、research memory、branch preview 和 checkpoint；
- 非标准正文的容错最终渲染、核心投影追踪和完整运行校验。
