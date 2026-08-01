# Learning Workflow：Codex CLI 新实现计划

> 归档状态：已被 Simple Semantic Loop 分项实现计划取代，仅供设计追溯。

> 目标模型：`gpt-5.6-sol`  
> Runtime：`codex app-server --listen stdio://`  
> 文档性质：实现计划；本轮不创建或修改任何生产脚本与 skill  
> 旧实现边界：现有脚本与 `.claude/skills`、现有 `.codex/skills` 均保持不变

## 0. 目标与边界

新实现要把以下两份设计变成一个独立、可恢复、可审计的 Codex 工作流：

- [learning_workflow_optimization_discussion.md](learning_workflow_optimization_discussion.md)：定义业务对象、Anchor 的 L1–L6 修改空间、Direction 和最终输出。
- [learning_workflow_agent_orchestration_design.md](learning_workflow_agent_orchestration_design.md)：定义 Agent 拆分、生命周期、Marker/LOOP 和脚本职责。

专家知识和已验证编排原则来自：

- [learning_workflow_reusable_knowledge_extraction.md](learning_workflow_reusable_knowledge_extraction.md)
- [learning_workflow_source_provenance.md](learning_workflow_source_provenance.md)
- `scripts/learning_scheduler.ts`
- `scripts/idea_review_orchestrator.ts`
- `.claude/skills/learning-experiment-from-notes-*`
- `.claude/skills/idea_question`
- `.claude/skills/idea_answer`
- `draft/review_draft.md`
- 已人工筛选的 `review_notes`

它们只作为只读知识来源。新实现：

- 不修改旧脚本；
- 不 import 旧脚本；
- 不调用旧脚本作为子流程；
- 不修改旧 skill；
- 不使用 `idea_brainstorm`、ideastorm 或未验证 brainstorm skill；
- 不延续旧 horizon-summary/vertical-summary 的有损压缩。

虽然归档的
`archive/learning_workflow/scripts/layered_exploration_core.py` 中已经存在
一个可选的 `codex exec` adapter，它仍采用前一版单体角色、纯 JSON action
和 `exec resume` 方式。新实现不 import 该 adapter，只把其中的 JSONL
解析、日志、quote 校验和 stable ID 思想作为参考。

## 1. 完成目标

### Stage 1：Anchor Explore

从 topic 和本地知识库构造：

```text
最多 30 个显著不同的 Anchor
每个 Anchor 的 BaselineSet
每个 Anchor 的 L1–L6 LayerEntry 集合
Entry-level CrossLayerEdge
独立 Baseline/Implementation/Method Registry
```

完成条件由脚本裁决：

```text
accepted_anchor_count >= 30
OR consecutive_rounds_without_new_anchor >= 2
OR round / usage / task budget exhausted
```

即使停止条件触发，已经验证的 baseline、implementation asset、method reference 和现有 Anchor enrichment 仍保留。

### Stage 2：Direction Review

对每个 Anchor：

1. 从 LayerEntry graph 中形成一个或多个兼容 Direction；
2. 不把相互替代或冲突的 entry 拼成协同路径；
3. 为每个 Direction 形成 ExperimentBundle；
4. 按专家流程逐个深审；
5. 所有 Anchor 与 Direction 均进入可审计的终态。

Stage 2 完成条件：

```text
所有 accepted/active Anchor 的 Direction Planning 均终止
AND 所有 Direction 均有 terminal review 或明确 pending reason
AND baseline/reference-only bundle 已进入最终输出
AND deterministic validation 通过
```

### 最终产物

最终输出不是摘要，而是 canonical 对象的确定性渲染：

```text
Global Layer Catalog
每个 Anchor 的 L1–L6 Intervention Map
每个 Direction 的 selected entry/edge subgraph
Baseline/Reference Registry
Experiment Bundle
Expert Review
Evidence Gap 与下一步实验
```

价值优先级固定为：

```text
可探索场景 / 潜在加速机会
  > 实现、代码、工具、软件复用
  > 论文方法参考
```

Baseline 是并行的强制保留轨道，不因 exploration value 低而丢弃。

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────┐
│ CodexLearningWorkflowOrchestrator                   │
│                                                     │
│  canonical state / event log / task graph           │
│  protocol parser / validator / scheduler            │
│  source verifier / stable IDs / deterministic render│
└─────────────────────────┬───────────────────────────┘
                          │ JSONL stdio
┌─────────────────────────▼───────────────────────────┐
│ codex app-server                                    │
│                                                     │
│  Thread A: anchor-stage-controller                  │
│  Thread B*: anchor-evidence-worker (ephemeral)      │
│  Thread C*: anchor-curator-worker (ephemeral)       │
│  Thread D*: direction-planner (per Anchor)          │
│  Thread E*: direction-reviewer (per Direction)      │
│  Thread F*: review-evidence-worker (ephemeral)      │
└─────────────────────────────────────────────────────┘
```

关键约束：

- 一个 Agent 对应一个 Codex thread；
- 一个 Skill 只定义一个 Agent；
- Agent 之间不能直接通信；
- Agent 不能创建、恢复、关闭或调用其他 Agent；
- Orchestrator 是唯一的消息路由器和状态拥有者；
- 同一个 thread 同时最多一个 active turn；
- 不同 thread 可由脚本按 worker pool 并发驱动；
- persistent thread 可恢复，ephemeral thread 执行一个任务后销毁。

## 3. Agent 与 `gpt-5.6-sol` effort 配置

所有语义 Agent 固定使用 `gpt-5.6-sol`，通过 `turn/start.effort` 区分计算强度。

| Agent / Skill | 生命周期 | 知识库 | Effort | 原因 |
|---|---|---:|---|---|
| `learning-anchor-stage-controller` | Stage 1 单个持久 thread | 无 | `high` | 需要规划差异化 frontier、判断重复与收敛，但不做证据检索 |
| `learning-anchor-evidence-worker` | 每个任务短暂 thread | Obsidian 只读 | `medium` | 任务范围窄，重点是检索、原子 claim 和精确引用 |
| `learning-anchor-curator-worker` | 每批 claim 短暂 thread | 无，只看 evidence packet | `high` | 需要拆分 Anchor、Baseline、Entry、Edge，防止错误合并 |
| `learning-direction-planner` | 每个 Anchor 一个持久 thread | 无 | `xhigh` | 需要识别跨层接口、替代/冲突和显著不同的可证伪子图 |
| `learning-direction-reviewer` | 每个 Direction 一个持久 thread | 无 | `xhigh` | 需要反例驱动深审、baseline 公平性和最终专家判断 |
| `learning-review-evidence-worker` | 每个问题短暂 thread | Obsidian 只读 | `high` | 需要围绕一个问题重建因果链、核对定量证据和实现边界 |
| Protocol repair turn | 原 thread 最多一次 | 禁止工具 | `low` | 只补 Marker/LOOP，不重做语义推理 |

约束：

```text
禁止 ultra
不启用 collaborationMode
不允许 Codex subagent
不自动切换到 Terra/Luna
不自动降低 effort
```

理由：

- `ultra` 可能主动使用 subagent，与“Agent 不管理 Agent”的硬约束冲突；
- `xhigh` 只给 Direction Planner 和 Reviewer，避免高召回 Worker 无谓增加成本；
- model/effort 必须在 `doctor` 的 `model/list` 中实际可用，否则在首次付费 turn 前 fail closed。

初始配置建议：

```json
{
  "model": "gpt-5.6-sol",
  "role_effort": {
    "anchor_stage_controller": "high",
    "anchor_evidence_worker": "medium",
    "anchor_curator_worker": "high",
    "direction_planner": "xhigh",
    "direction_reviewer": "xhigh",
    "review_evidence_worker": "high",
    "protocol_repair": "low"
  },
  "forbid_efforts": ["ultra"],
  "allow_model_fallback": false
}
```

## 4. Codex App Server Runtime

### 4.1 启动与握手

Orchestrator 只启动一个长期 App Server：

```text
codex app-server --listen stdio://
```

启动顺序：

1. 建立 stdin/stdout/stderr pipe；
2. 发送 `initialize`；
3. 等待 response；
4. 发送 `initialized`；
5. 调用 `model/list`；
6. 调用 skill/MCP/permission capability probe；
7. 检查模型、effort、认证、CLI 版本和 schema hash；
8. 只有 doctor 全部通过后才允许付费 turn。

需要处理：

- state DB migration/backfill 的延迟；
- 启动超时与指数退避；
- App Server 异常退出；
- request ID 关联；
- notification 按 `threadId`、`turnId` 分流；
- stdout 非法 JSONL；
- stderr 独立保存；
- SIGINT/SIGTERM 下先 checkpoint，再 interrupt active turns，最后结束进程。

### 4.2 协议类型生成

构建或首次 doctor 时运行：

```text
codex app-server generate-ts --out <generated-dir>
codex app-server generate-json-schema --out <generated-dir>
```

保存：

```text
codex_cli_version
app_server_schema_hash
generated_at
supported_methods
```

恢复旧 run 时，如果版本/schema 不一致：

- 不直接 resume；
- 先执行兼容性检查；
- 未编写迁移器时 fail closed；
- 不猜测字段。

### 4.3 Thread 创建

持久角色：

```json
{
  "method": "thread/start",
  "params": {
    "model": "gpt-5.6-sol",
    "ephemeral": false,
    "approvalPolicy": "never",
    "sandbox": "read-only",
    "cwd": "<role-sandbox-dir>"
  }
}
```

短暂角色：

```json
{
  "method": "thread/start",
  "params": {
    "model": "gpt-5.6-sol",
    "ephemeral": true,
    "approvalPolicy": "never",
    "sandbox": "read-only",
    "cwd": "<task-sandbox-dir>"
  }
}
```

模型 effort 每轮显式传入：

```json
{
  "method": "turn/start",
  "params": {
    "threadId": "<thread-id>",
    "model": "gpt-5.6-sol",
    "effort": "high",
    "input": []
  }
}
```

### 4.4 Skill 显式输入

首次 turn 显式传入唯一 skill：

```json
{
  "input": [
    {
      "type": "text",
      "text": "$learning-anchor-stage-controller\n<初始化输入>"
    },
    {
      "type": "skill",
      "name": "learning-anchor-stage-controller",
      "path": "/data3/paper_analysis/archive/learning_workflow/skills/learning-anchor-stage-controller/SKILL.md"
    }
  ]
}
```

后续 turn 依赖 thread 中已记录的 skill，但脚本每次仍回注 canonical state。Checkpoint 保存 `skill_path` 和 `skill_hash`；skill hash 改变后，不恢复旧 thread，而是从 canonical state 创建新 thread。

### 4.5 App Server Client 的内部接口

建议定义：

```ts
interface AgentRuntime {
  startPersistent(role: Role, init: RoleInit): Promise<AgentHandle>;
  startEphemeral(role: Role, task: TaskInput): Promise<AgentHandle>;
  resume(handle: PersistedAgentHandle): Promise<AgentHandle>;
  runTurn(handle: AgentHandle, input: TurnInput): Promise<TurnResult>;
  repairTurn(handle: AgentHandle, repair: RepairInput): Promise<TurnResult>;
  interrupt(handle: AgentHandle): Promise<void>;
  close(handle: AgentHandle): Promise<void>;
}
```

`TurnResult` 只包含 provider-neutral 数据：

```ts
interface TurnResult {
  threadId: string;
  turnId: string;
  status: "completed" | "interrupted" | "failed";
  text: string;
  usage: TokenUsage | null;
  observedEvents: NormalizedEvent[];
  rawLogPath: string;
}
```

## 5. 工具与知识库隔离

### 5.1 Role 权限矩阵

| Role | Shell/File | Obsidian MCP | Web | 写入 |
|---|---:|---:|---:|---:|
| Anchor Controller | 禁止 | 禁止 | 禁止 | 禁止 |
| Anchor Evidence Worker | 禁止 | 只读 search/get/list | 禁止 | 禁止 |
| Anchor Curator | 禁止 | 禁止 | 禁止 | 禁止 |
| Direction Planner | 禁止 | 禁止 | 禁止 | 禁止 |
| Direction Reviewer | 禁止 | 禁止 | 禁止 | 禁止 |
| Review Evidence Worker | 禁止 | 只读 search/get/list | 禁止 | 禁止 |

所有 canonical 文件只能由 Orchestrator 写入。

### 5.2 Event admission gate

无工具角色只允许：

```text
agent message
reasoning
turn lifecycle
usage
```

Evidence role 只额外允许：

```text
MCP server = obsidian
method ∈ read-only allowlist
```

必须拒收：

```text
command/shell
filesystem read/write
web search/fetch
file change/apply patch
非 Obsidian MCP
Obsidian write/patch/delete
subagent/collaboration
```

如果 persistent blind role 出现禁止事件：

1. 当前 turn 判为 `security_invalid`；
2. 不接纳其任何语义输出；
3. 该 thread 视为已污染并永久废弃；
4. 从 canonical checkpoint 创建新 thread；
5. 只允许一次全新 thread 重试；
6. 再次违规则任务停止，等待人工处理。

仅仅重试同一 thread 不足，因为它可能已经看到不应看到的内容。

### 5.3 盲评的物理隔离

Codex 当前没有 Claude `--tools ""` 的直接等价参数。MVP 使用：

```text
空 cwd
read-only sandbox
approval=never
禁用 MCP/Web
event allowlist
污染 thread 丢弃
```

在宣称“Reviewer 绝对无法读取知识库”之前，还要增加单独验收项：

- 用进程级 sandbox/container 启动 blind runtime；
- 不挂载 vault；
- 只挂载空 role sandbox；
- 只允许 Codex 上游认证/推理所需网络；
- 对 Evidence runtime 单独开放 Obsidian MCP。

如果 App Server state DB 不允许安全地并行启动两个 runtime，则保持单 App Server MVP，并明确其隔离等级为 admission-enforced，而不是物理盲隔离。

## 6. 控制协议

控制流不使用一个大型 JSON。每个 Agent 输出：

```text
外层 Marker
少量 key:value 控制字段
语义 JSON payload
LOOP 或 TASK_TERMINATED
```

每个 turn 只允许一种主 Marker。

### 6.1 Anchor Controller

规划下一轮：

```text
___ANCHOR_ROUND_PLAN_START___
round: 3
action: plan_round
task_count: 4
___SEMANTIC_PAYLOAD_START___
[
  {
    "task_id": "AE-R03-01",
    "focus": "...",
    "layer": "L2",
    "value_axis": "exploration"
  }
]
___SEMANTIC_PAYLOAD_END___
___ANCHOR_ROUND_PLAN_END___

[LOOP: §EVAL_ROUND | await=ROUND_RESULT | round=3]
```

完成：

```text
___ANCHOR_STAGE_COMPLETE_START___
reason: target_reached|no_new_anchor_streak|budget_exhausted
accepted_anchor_count: 30
___ANCHOR_STAGE_COMPLETE_END___

[LOOP: §TERMINATED | done]
```

Controller 的完成声明只是请求；脚本重新计算完成条件。

### 6.2 Anchor Evidence Worker

```text
___ANCHOR_EVIDENCE_RESULT_START___
task_id: AE-R03-01
status: complete
___CLAIMS_START___
[
  {
    "statement": "...",
    "claim_type": "scenario",
    "source_path": "...",
    "line_start": 120,
    "line_end": 138,
    "quote": "..."
  }
]
___CLAIMS_END___
___GAPS_START___
- ...
___GAPS_END___
___ANCHOR_EVIDENCE_RESULT_END___

[TASK_TERMINATED]
```

### 6.3 Anchor Curator

```text
___ANCHOR_DELTA_START___
task_id: AC-R03-B01
status: complete
___SEMANTIC_PAYLOAD_START___
{
  "anchors": [],
  "baselines": [],
  "entries": [],
  "edges": [],
  "dispositions": []
}
___SEMANTIC_PAYLOAD_END___
___ANCHOR_DELTA_END___

[TASK_TERMINATED]
```

Curator 使用临时 key；canonical ID 由脚本分配。

### 6.4 Direction Planner

```text
___DIRECTION_PROPOSAL_START___
anchor_id: A-...
proposal_index: 2
___SEMANTIC_PAYLOAD_START___
{
  "selected_entry_ids": [],
  "selected_edge_ids": [],
  "baseline_ids": [],
  "hypothesis": "...",
  "ablation_plan": []
}
___SEMANTIC_PAYLOAD_END___
___DIRECTION_PROPOSAL_END___

[LOOP: §EVAL_DIRECTION | await=DIRECTION_COMMIT_RESULT | anchor_id=A-...]
```

当不存在新的显著不同 Direction：

```text
___DIRECTION_PLANNING_COMPLETE_START___
anchor_id: A-...
reason: exhausted_distinct_subgraphs|budget_exhausted
___DIRECTION_PLANNING_COMPLETE_END___

[LOOP: §TERMINATED | done]
```

### 6.5 Direction Reviewer

每轮只允许：

```text
___REVIEW_QUESTION_START___ ... ___REVIEW_QUESTION_END___
___REVIEW_REFERENCE_REQUEST_START___ ... ___REVIEW_REFERENCE_REQUEST_END___
___DIRECTION_REVIEW_COMPLETE_START___ ... ___DIRECTION_REVIEW_COMPLETE_END___
```

并分别附带：

```text
[LOOP: §EVAL_ANSWER | await=REVIEW_EVIDENCE_RESULT | ...]
[LOOP: §ASK | await=REVIEW_REFERENCE | ...]
[LOOP: §TERMINATED | done]
```

### 6.6 Review Evidence Worker

```text
___REVIEW_EVIDENCE_RESULT_START___
direction_id: D-...
round: 4
status: complete
___SOURCES_START___
- claim-id / path / line
___SOURCES_END___
___GAPS_START___
- ...
___GAPS_END___
___ANSWER_START___
...
___ANSWER_END___
___REVIEW_EVIDENCE_RESULT_END___

[TASK_TERMINATED]
```

## 7. Stage 1：Anchor Explore

### 7.1 初始化

Orchestrator 写入：

```text
topic
constraints
L1–L6 定义
三价值轴
baseline 强制轨道
max_anchors = 30
no_new_anchor_stop = 2
round/task/usage budget
```

创建一个持久 `anchor-stage-controller` thread。Controller 只接收规范状态，不读取知识库。

### 7.2 每轮流程

```text
Controller: ROUND_PLAN
  ↓
脚本校验 task 数、重复 focus、layer/axis 和预算
  ↓
并发启动 N 个 ephemeral Anchor Evidence Worker
  ↓
验证 source path/line/quote，形成 append-only EvidenceClaim
  ↓
按完整 claim 边界分 batch
  ↓
启动 ephemeral Curator Worker
  ↓
脚本分配 stable ID，逐对象验证和接纳
  ↓
Anchor 去重、Baseline 合并、Entry/Edge 校验、Top-30 更新
  ↓
形成 canonical ROUND_RESULT
  ↓
回注 Controller
```

每轮回注：

```text
本次执行语义
CURRENT_STAGE_STATE
PREVIOUS_CONTROLLER_OUTPUT
ROUND_RESULT
FRONTIER_SUMMARY（ID 与 gap，不做有损 prose summary）
```

### 7.3 Anchor 去重与 Top-30

Anchor signature：

```text
workload
× phase
× regime
× backend
× bottleneck
× primary baseline execution path
× target metric set
```

这些字段的实质变化才构成新 Anchor。

超过 30 个候选时采用词典序优先级，不做加权平均：

```text
scenario/opportunity evidence
→ concrete modifiable object
→ direct evidence and falsifiable metric
→ baseline coverage
→ implementation reuse
→ method reference
```

低 exploration 的有效 baseline 不参与 Anchor novelty 竞争，而进入独立 Baseline/Reference Registry。

### 7.4 停止

脚本在每轮 commit 后计算：

```text
new_accepted_anchor_count
accepted_anchor_count
consecutive_rounds_without_new_anchor
remaining_round/task/usage budget
```

只有脚本可以将 Stage 1 标为 complete。Controller 可提出完成，但不能覆盖：

```text
accepted_anchor_count >= 30
OR consecutive_rounds_without_new_anchor >= 2
OR budget exhausted
```

Stage 1 完成后生成不可变 `anchor_space_version`，供 Stage 2 使用。Stage 2 新发现的缺口进入 review evidence ledger，不静默重写 Stage 1 历史。

## 8. Stage 2：Direction Planning 与 Review

### 8.1 每个 Anchor 的 Direction Planner

每个 Anchor 创建一个持久 thread，只输入：

```text
Anchor exact context
BaselineSet
该 Anchor 的全部 LayerEntry
Entry-level Edge
已接纳 Direction signature
未解决 gap
```

Planner 每轮提出一个 Direction 或声明完成。脚本验证：

- 所有 entry 属于同一 Anchor；
- edge endpoints 均被选择；
- selected edge 确实存在；
- conflict/substitute/incompatible edge 不作为 synergy；
- conditional edge 有明确 condition；
- hypothesis 可证伪；
- baseline 明确；
- 与已有 Direction 的 selected subgraph/hypothesis 不重复；
- 一层 Direction 合法；
- 不要求覆盖 L1–L6。

Commit result 必须回注 Planner，避免它把脚本已拒绝的提案误认为已接纳。

### 8.2 每个 Direction 的 Reviewer

每个 Direction 创建一个持久 Reviewer thread。Reviewer 无知识库权限，初始队列固定覆盖：

```text
scenario_opportunity
baseline_fairness
entry_validity
cross_layer_validity
implementation_reuse
experiment_measurement
```

Reviewer 每轮只做一种动作：

```text
ask
request_reference
complete
```

`ask` 后脚本创建一个 ephemeral `review-evidence-worker`：

- 接收一个明确问题；
- 接收 ExperimentBundle 和可引用 claim ledger；
- 必要时通过 Obsidian MCP 做一次受限补查；
- 输出 direct/inferred/unknown、sources 和 gaps；
- 不做最终价值判断。

脚本把规范化回答回注 Reviewer。

### 8.3 专家 reference 注入

采用旧 `idea_question` 已验证的“白名单 + 每类最多一次 + 按需注入”：

```text
scenario_and_acceleration
baseline_and_fairness
layer_modification_and_implementation
cross_layer_interface_and_conflict
experiment_tool_and_measurement
```

Reference 只提供评判问题，不提供当前 Direction 的事实。

### 8.4 Reviewer 完成裁决

Reviewer 的 `complete` 必须包含：

```text
exploration_value
implementation_reuse
method_reference
baseline_quality
cross_layer_validity
experiment_readiness
decision
minimum implementation plan
baseline/ablation matrix
metrics/tools
failure/stop conditions
selected and alternative refs
gaps
```

脚本只有在六个 review dimension 都有证据回答或显式 `not_applicable/gap` 后才接受完成。

决策优先级：

1. 无效证据或图不一致：`rejected`；
2. 有效 baseline/tool/implementation/reference 但不是探索方向：`baseline_reference`；
3. 探索潜力存在但关键证据缺失：`needs_evidence`；
4. 可证伪、baseline 公平、实现与测量路径成立：`experiment_candidate`。

## 9. Canonical State 与目录

建议默认输出到独立根目录，避免与旧 `learning_outputs` 混合：

```text
/data3/paper_analysis/learning_outputs_codex/<run-id>/
```

Run 布局：

```text
<run-dir>/
├── config.json
├── state.json
├── events.jsonl
├── provider/
│   ├── runtime.json
│   ├── generated_schema/
│   ├── threads.json
│   └── raw_turns/
├── sessions/
│   ├── anchor_controller.json
│   ├── direction_planners/
│   └── direction_reviewers/
├── tasks/
│   ├── anchor_evidence/
│   ├── anchor_curation/
│   └── review_evidence/
├── evidence/
│   ├── claims.jsonl
│   ├── rejected_claims.jsonl
│   └── review_claims.jsonl
├── catalog/
│   └── entities.json
├── anchors/
│   ├── index.json
│   └── <anchor-id>.json
├── directions/
│   ├── index.json
│   └── <direction-id>.json
├── reviews/
│   ├── <direction-id>.json
│   └── <direction-id>.md
├── validation.json
└── final.md
```

Persistent session record：

```json
{
  "role": "direction_reviewer",
  "scope_id": "D-...",
  "thread_id": "...",
  "last_turn_id": "...",
  "model": "gpt-5.6-sol",
  "effort": "xhigh",
  "skill_path": "...",
  "skill_hash": "...",
  "last_loop": "...",
  "last_normalized_output_path": "...",
  "status": "yielded",
  "protocol_version": 1
}
```

Session 状态：

```text
not_started
→ initializing
→ waiting_input
→ running
→ yielded
→ running
→ terminated
```

Task 状态：

```text
pending
→ dispatched
→ response_received
→ protocol_valid
→ domain_valid
→ committed

任何中间状态
↘ failed_retriable
↘ failed_terminal
↘ security_invalid
```

## 10. Context、成本与并发控制

### 10.1 不依赖自动压缩

Persistent Agent scope 被严格限制：

- Anchor Controller：仅 Stage 1 的少量 round；
- Direction Planner：仅一个 Anchor；
- Reviewer：仅一个 Direction。

每轮重复输入：

```text
当前 canonical state
上次 normalized output
本轮新增 result
本次执行语义
```

如果出现以下任一条件，脚本轮换 thread：

- 达到 `max_persistent_turns`；
- 本轮 input byte/token 预算超限；
- usage 累计达到配置阈值；
- App Server 报告 compaction；
- skill/protocol/model/effort 配置发生变化；
- thread 发生工具污染。

新 thread 从 canonical checkpoint 恢复，不使用自然语言历史摘要。

### 10.2 Skill 上下文

实测表明全局 skill catalog 可能带来显著固定输入成本。实现时：

1. 调用 `skills/list` 或等价能力获取实际可见 skill；
2. 每个 role 显式输入唯一 skill；
3. 使用 thread config/profile 禁用无关 MCP、Web 和可禁用的无关 skill；
4. 记录首次 turn input usage；
5. 若固定上下文超过阈值，doctor 给出失败/警告，不能无界运行 30 Anchor。

### 10.3 并发

默认：

```text
Anchor Evidence Worker: 2–4
Anchor Curator Worker: 1–2
Direction Planner: 1–2
Direction Review: 1–2
Review Evidence Worker: 跟随 active review 数
```

并发是脚本配置，不由 Agent 决定。每个 thread 串行 turn；App Server client 对不同 thread 的事件进行 multiplex。

### 10.4 Usage budget

至少记录：

```text
input_tokens
cached_input_tokens
output_tokens
reasoning_output_tokens
elapsed time
role
anchor/direction/task id
requested model/effort
actual model/effort（若事件返回）
```

预算可按：

```text
run
stage
role
anchor
direction
task
```

设置。ChatGPT 登录可能不返回美元费用，因此 token/turn 限额是强制预算，美元预算只在 provider 提供可核验数据时启用。

## 11. 新文件规划

以下均为新文件；不覆盖旧文件。

```text
scripts/
├── codex_learning_workflow.ts
└── codex_learning_workflow/
    ├── app_server_client.ts
    ├── runtime_manager.ts
    ├── role_profiles.ts
    ├── protocol_parser.ts
    ├── protocol_state_machine.ts
    ├── task_scheduler.ts
    ├── canonical_store.ts
    ├── stable_ids.ts
    ├── source_validator.ts
    ├── domain_validators.ts
    ├── anchor_stage.ts
    ├── direction_stage.ts
    ├── renderer.ts
    ├── doctor.ts
    ├── types.ts
    ├── schemas/
    └── tests/

.codex/skills/
├── learning-anchor-stage-controller/
│   └── SKILL.md
├── learning-anchor-evidence-worker/
│   └── SKILL.md
├── learning-anchor-curator-worker/
│   └── SKILL.md
├── learning-direction-planner/
│   └── SKILL.md
├── learning-direction-reviewer/
│   ├── SKILL.md
│   └── references/
└── learning-review-evidence-worker/
    └── SKILL.md
```

每个 Skill：

- 只描述自己的输入、步骤、Marker、LOOP 和禁止事项；
- 不描述如何启动其他 Agent；
- 不包含全工作流状态机；
- 不写 canonical 文件；
- 不调用 Codex/Claude CLI；
- 不使用 Agent/Task/subagent 工具；
- 只加载该角色必需的 reference。

## 12. CLI 设计

```text
npx tsx scripts/codex_learning_workflow.ts doctor

npx tsx scripts/codex_learning_workflow.ts init \
  --topic "<topic>" \
  --work-dir "<run-dir>" \
  --model gpt-5.6-sol \
  --max-anchors 30

npx tsx scripts/codex_learning_workflow.ts run \
  --work-dir "<run-dir>"

npx tsx scripts/codex_learning_workflow.ts run \
  --work-dir "<run-dir>" \
  --resume

npx tsx scripts/codex_learning_workflow.ts status \
  --work-dir "<run-dir>"

npx tsx scripts/codex_learning_workflow.ts validate \
  --work-dir "<run-dir>"

npx tsx scripts/codex_learning_workflow.ts render \
  --work-dir "<run-dir>"
```

可调试停止点：

```text
--stop-after anchor-explore
--stop-after direction-plan
--stop-after direction-review
```

## 13. 实现阶段

### Phase 0：冻结协议和对象

产物：

- Domain types；
- 每个 role 的 Marker allowlist；
- LOOP transition table；
- canonical schemas；
- source quote validation rules；
- role model/effort config。

验收：

- 所有 Marker 都能唯一映射到一个合法下一状态；
- 每个 Agent 只有一个 Skill；
- Stage 1/2 完成条件可由纯代码计算。

### Phase 1：App Server Client

实现：

- process lifecycle；
- initialize handshake；
- request/response correlation；
- notification demux；
- thread start/resume；
- turn start/steer/interrupt；
- raw log；
- timeout；
- version/schema/model preflight。

测试：

- fake App Server 单元测试；
- 同一 thread 两轮；
- 两个 thread 并发事件交错；
- server crash 后 persistent thread resume；
- ephemeral task 重置；
- state DB backfill 延迟；
- malformed JSONL。

### Phase 2：协议与状态机

实现：

- Marker parser；
- payload boundary parser；
- LOOP parser；
- role allowlist；
- state transition validator；
- 一次 protocol repair；
- task/session checkpoint。

测试：

- 缺 Marker；
- 缺 LOOP；
- 多个主 Marker；
- round 不匹配；
- repair 改变语义；
- 非法 next state；
- raw 输出可恢复。

### Phase 3：六个单 Agent Skill

按顺序：

1. Anchor Evidence Worker；
2. Anchor Curator；
3. Anchor Controller；
4. Direction Planner；
5. Review Evidence Worker；
6. Direction Reviewer。

先实现短暂 Agent，确认输入/输出稳定，再实现持久 loop。

验收：

- 每个 Skill 独立阅读即可执行；
- 不引用其他 Agent 的内部逻辑；
- 不出现 Agent 启动命令；
- 控制输出只使用约定 Marker/LOOP。

### Phase 4：Stage 1

实现：

- Controller loop；
- evidence task pool；
- Obsidian read-only worker policy；
- claim ledger；
- quote validation；
- curator batch；
- Anchor/Baseline/Entry/Edge merge；
- Top-30 和停止条件；
- resume。

验收：

- 每层允许零到多个 entry；
- edge 连接 entry ID；
- baseline 低 exploration 仍保留；
- 连续两轮无新 Anchor 时停止；
- 30 Anchor 硬上限；
- 不生成 horizon/vertical summary。

### Phase 5：Stage 2

实现：

- per-Anchor Direction Planner；
- Direction subgraph validation；
- per-Direction Reviewer；
- ephemeral Review Evidence Worker；
- reference whitelist；
- 六维审阅覆盖；
- expert decision。

验收：

- 所有 Anchor 终止；
- 每个 Direction 单独审阅；
- conflict/substitute 不会被误拼；
- Reviewer 不直接检索；
- valid baseline-only bundle 进入最终输出。

### Phase 6：验证和渲染

实现纯代码：

- referential integrity；
- source quote；
- Anchor membership；
- graph connectivity/compatibility；
- baseline coverage；
- review enum/completion；
- deterministic Markdown render。

验收：

```text
final statement
→ review/direction/entry/baseline
→ EvidenceClaim
→ source path + line + quote
```

任何链条不可达则 validation 失败。

### Phase 7：小规模付费 Canary

按成本从低到高：

1. 一个 Evidence Worker；
2. 一个 Controller 两轮；
3. 一个 Anchor 的 Direction Planner；
4. 一个 Direction 的两轮 Review；
5. 一个小 topic 的 Stage 1 + Stage 2。

每步比较：

- 协议成功率；
- source 精度；
- tool violation；
- 重复率；
- input/output usage；
- effort 增益；
- 与人工 `review_draft`/`review_notes` 的一致性。

只有小规模 gold case 通过后才开放 30 Anchor。

## 14. 测试矩阵与最终验收

### 确定性测试

- stable ID；
- atomic write；
- event replay；
- state recovery；
- Anchor signature；
- claim quote；
- entry/edge refs；
- Direction graph；
- baseline registry；
- final render snapshot。

### Provider 测试

- version mismatch；
- model/effort unavailable；
- auth failure；
- App Server startup/backfill；
- thread resume；
- turn timeout/interrupt；
- event interleaving；
- token usage；
- server restart。

### AI 原生错误测试

- 叙述性输出无 Marker；
- JSON payload 局部损坏；
- hallucinated source；
- 错误 line/quote；
- 把 substitute 当 complement；
- 跨 Anchor edge；
- 重复 Direction；
- Reviewer 过早 complete；
- Controller 自报完成但脚本条件未满足；
- Agent 尝试使用工具或 subagent。

### 工作流最终验收标准

只有同时满足以下条件才可称为完成：

1. 旧脚本和旧 skill 未被修改、import 或调用；
2. 六个 Agent 各自只有一个 Skill；
3. 所有 Agent 生命周期由脚本管理；
4. `gpt-5.6-sol` 和 role effort 在 run config 中固定并经 doctor 验证；
5. 没有 `ultra` 或 subagent 行为；
6. Stage 1 满足 30 Anchor/两轮无新 Anchor/预算之一；
7. 每个 Anchor 保留多 entry 的 L1–L6 map；
8. 每个 Direction 是 entry-level 兼容子图；
9. 所有 Direction 都有 terminal review 或明确 pending reason；
10. baseline/reference 轨道完整进入最终输出；
11. canonical 数据可从 event log/checkpoint 恢复；
12. `validate` 成功后才允许生成正式 `final.md`。
