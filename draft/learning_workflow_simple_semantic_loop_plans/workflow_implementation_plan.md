# Simple Semantic Loop Workflow 需求闭包计划

## 1. 本计划的作用

本文件只记录：

- [`todo_draft.md`](../todo_draft.md) 第 19–35 行要求什么可观察行为；
- Controller 与四类 Turn 如何共同产生这些行为；
- 端到端数据如何流动；
- 最终需要哪些运行证据才能证明需求已闭合。

组件内部实现由各组件计划负责。本文件不重复完整 schema、Prompt 或数据库表。

## 2. 需求清单

| ID | 原始需求 |
|---|---|
| `REQ-19` | HPC 最新论文知识库包含多维度场景、知识、方法和实现；experiment、idea/baseline、knowledge、human 各有表达维度 |
| `REQ-20` | 不同维度笔记作为不同需求的查询目标，查询通过 Obsidian Omnisearch |
| `REQ-21` | 明确用什么关键词查询、什么时候触发查询 |
| `REQ-23` | 用户给一个 Topic，最终从知识库得到该 Topic 的进一步性能优化潜力 |
| `REQ-24` | 过程输出按实际需求组织，最终输出供人阅读理解 |
| `REQ-25` | 定义 Agent—脚本协议，以格式化数据给 Agent 思考 |
| `REQ-28` | L1–L6 是性能修改坐标，Topic 明确限定允许的层子集 |
| `REQ-29` | Anchor 用场景、baseline 和性能张力圈定局部搜索区域 |
| `REQ-30` | Direction 表达 Anchor 内候选修改、因果、实现、测量和反证，并独立审阅 |
| `REQ-31` | 以搜索—集成—审阅的 Loop 自动挖掘，而非一次性固定问答 |
| `REQ-34` | 调度必须把运行状态外部化，避免长上下文遗忘并支持重启恢复 |
| `REQ-35` | 脚本顺序调度 fresh Turn；Skill 保存稳定方法，运行产出由 Controller 保存 |

本阶段不以发现固定数量的 Anchor/Direction、执行研究实验或进行价值排名作为验收要求。

## 3. 组件计划

| 组件 | 计划 | 在闭包中的责任 |
|---|---|---|
| Shared Contracts | [共享契约规范](shared_contracts.md) | 定义过程对象和 Agent—脚本协议的唯一来源 |
| Controller | [Scheduler Script 实现计划](scheduler_script_implementation_plan.md) | 权威状态、动态 plan、Turn 编排、Gate、恢复、渲染和完成 |
| Workflow Decision Turn | [Workflow Turn Agent 实现计划](workflow_turn_agent_implementation_plan.md) | 从当前状态判断查询、集成、审阅、重排或闭包 |
| Evidence Reader Turn | [Evidence Reader 实现计划](evidence_reader_agent_implementation_plan.md) | 按维度使用 Omnisearch、深读并形成 Evidence |
| Direction Reviewer Turn | [Direction Reviewer 实现计划](direction_reviewer_agent_implementation_plan.md) | 独立审阅一个 Direction |
| Closure Reviewer Turn | [Closure Reviewer 实现计划](closure_reviewer_agent_implementation_plan.md) | 独立判断 Topic 是否闭环 |

Reasoning effort 是 workflow 级固定策略：

```text
Workflow Decision Turn = max
Evidence Reader / Direction Reviewer / Closure Reviewer = high
```

Controller、run config、CLI 和 Agent proposal 均不得覆盖或降级该映射。

Controller 只启动表中的四类 Turn。自检属于每个产出 Agent 的职责，但脚本
仍假定输出可能结构错误、绑定错误或结构合法而语义错误。pre-Gate 校验失败后
至多以同一角色创建两个 fresh replacement attempt，并把上一份 Controller
错误报告作为有界、哈希绑定的 task 输入。每条错误包含实际错误、权威
`requiredRule` 和 Controller 定义的 `validExamples`；确定性异常由脚本恢复，
需要业务语义取舍的异常以注册 trigger 交给 Workflow Decision Turn。

## 4. 端到端架构

```text
用户 Topic
   ↓
Controller 创建 run，冻结 objective / acceptance
   ↓
fresh Workflow Turn：TopicFrame
   ↓
Controller Gate + canonical commit
   ↓
fresh Workflow Turn：形成一个 SearchNeed
   ↓
Controller 冻结 EVIDENCE_READ Stage/Gate
   ↓
fresh Evidence Reader：Omnisearch + deep read
   ↓
Controller Gate + EvidencePacket commit
   ↓
fresh Workflow Turn：Evidence → SemanticDelta
   ↓
Controller commit Anchor / Direction revision
   ↓
fresh Direction Reviewer
   ↓
Controller commit ReviewDelta
   ↓
fresh Workflow Turn：查询 / 修订 / 终态 / replan
   ↺
fresh Workflow Turn：StopCandidate
   ↓
Controller preflight
   ↓
fresh Closure Reviewer
   ├─ reject → 新 Workflow Turn 重开 scope
   └─ accept → Controller validate/render/atomic completed
```

所有 Agent 之间均无直接通信；箭头中的转发均表示 Controller 从已提交状态构造新 task。

## 5. 业务语义 Loop

### 5.1 Topic 搜索空间

TopicFrame 保存：

- workload、phase、regime、stack；
- L1–L6 layer scope；
- target metrics；
- invariants、exclusions；
- seed terms、synonyms；
- scope audit。

未知范围保持未决，不允许 Agent 静默缩窄。

L1–L6 是固定修改坐标，不是必须依次覆盖的流水线：

| Layer | 可修改对象 |
|---|---|
| L1 Algorithm/Pipeline | 计算图、负载分解、动态参数、近似和并行性 |
| L2 Serving/Runtime | 请求、batch、stage、队列、放置、缓存和资源调度 |
| L3 Compiler | IR、依赖、pass、fusion、multiversion 和 codegen |
| L4 Kernel | tile、warp、指令流水、同步、数据移动和 kernel 组合 |
| L5 Architecture | 计算/控制单元、存储层次、调度器、NoC 和硬件原语 |
| L6 Chip/System | chiplet、PIM、wafer-scale、封装、互联和芯片资源边界 |

Direction 可使用一层或多层，但每个 `ModificationAtom.layer` 必须属于当前
Topic `layerScope`；不允许以探索需要为由扩大 Topic 层范围。

### 5.2 Anchor

```text
Anchor =
  场景边界
  × baseline 执行路径
  × 性能张力
```

Anchor 用于把 Topic 切成可理解、可检索的局部区域。

### 5.3 Direction

```text
Direction =
  Anchor
  + 最小修改集合
  + 性能因果链
  + 公平比较
  + 可推翻条件
```

Direction 是最终“性能优化潜力”的核心表达，但 `testable` 不等于 experimentally validated。

### 5.4 Loop

Evidence 可以随时：

- 创建或修订 Anchor；
- 创建或修订 Direction；
- 增加反证；
- 暴露实现/测量缺口；
- 将剩余问题归为 experiment-required。

Workflow 不冻结为“先发现完全部 Anchor，再开始 Direction”。

## 6. `REQ-19`–`REQ-21`：知识维度与查询闭包

### 6.1 维度

| Dimension | Vault path | 主要信息 |
|---|---|---|
| idea/baseline | `idea_notes/` | 场景、baseline、论文方法、报告收益、候选机会 |
| knowledge | `knowledge_notes/` | 机制、接口、约束、适用边界 |
| experiment | `experiment_notes/` | 已有实现、配置、工具、指标、历史实验和失败 |
| human | `human_notes/` | 本地经验、会议判断、环境限制 |
| paper | `paper_secs/` | 原文核验 |

### 6.2 查询何时触发

Workflow Turn 只在出现一个具体语义缺口时提出 SearchNeed：

- Topic 初始化或最终扩展需要发现 Anchor；
- baseline 缺执行路径或公平比较；
- Anchor 缺明确修改对象；
- Direction 因果机制或跨层接口不清；
- 缺实现入口；
- 缺测量/消融/falsifier；
- 需要反例或退化条件；
- 定量声明需要原文核验。

以下表达不能触发查询：

- “继续研究”；
- “寻找更多资料”；
- “把所有目录都搜索一遍”；
- “每个 L1–L6 都搜一次”。

### 6.3 查询关键词如何形成

Workflow Turn 在 SearchNeed 中提供：

- technical objects；
- exact scenario terms；
- performance relation/evidence intent；
- known terms/synonyms；
- success criteria；
- excluded sources。

Evidence Reader 形成：

```text
Q1：精确实体 + 精确场景 + intent-specific 精确词
Q2：保留场景，使用 SearchIntent 对应的 baseline / modification /
    mechanism / implementation / measurement / challenge / primary-source 关系轴
Q3：有来源的同义词 + 较宽但仍在 Topic 内的场景
```

每个 query 带 path filter、term provenance 和 sequence。Q1–Q3 是整个 task 的全局降级级别，不按 primary/auxiliary dimension 重置。Reader 只在上一级不足时降级，并在 success criteria 满足或 Topic 边界将丢失时停止。

### 6.4 查询闭包证据

一个 SearchNeed 的运行记录必须包含：

- trigger 和 owner；
- SearchIntent；
- primary/auxiliary dimension；
- success criteria；
- 每个 query 及 term 来源；
- hit selection；
- contexts actually read；
- findings/contradictions/unanswered；
- answered/partial/not_found；
- query/result artifact hashes。

## 7. `REQ-23`：Topic 到优化潜力

### 7.1 产物链

```text
Topic
→ TopicFrame
→ Anchor
→ Direction
→ Evidence / Review
→ Direction terminal state
→ human-readable performance opportunity
```

最终保留：

- `testable` Direction；
- `experiment_required` Direction 和不可执行 handoff；
- `rejected` Direction 的必要反证/原因；
- 未形成 Direction 但有保留价值的 Anchor 信息。

### 7.2 Direction 必须回答

- 性能问题发生在哪里；
- baseline 如何执行；
- 修改什么；
- 为什么可能改变目标指标；
- 公平比较是什么；
- 实现入口和未决接口是什么；
- 如何证伪；
- 什么条件下退化；
- Evidence 来自哪里；
- 剩余问题是否需要新实验。

### 7.3 终态

```text
testable
experiment_required
rejected
```

Anchor 终态：

```text
saturated
rejected
```

没有固定 Direction 数量。闭包基于可审计 frontier，而不是配额。

## 8. `REQ-24`：过程产物和最终人类输出

### 8.1 过程产物

过程使用结构化对象：

- WorkflowState、WorkflowPlan；
- StageContract、GateDefinition；
- tasks、attempts、events；
- TopicFrame、Anchor、Direction；
- SearchNeed、EvidencePacket；
- SemanticDelta、ReviewDelta；
- StopCandidate、ClosureReview；
- validation、usage、artifact manifest；
- raw Turn output。

这些对象服务于恢复、审计和 Agent 输入，不要求人直接阅读。

### 8.2 最终输出

Controller 从 committed canonical state 确定性渲染 `final.md`：

```markdown
# Topic 与搜索边界
# 性能机会概览
# Anchors
## A01：场景 × Baseline × 性能张力
### Baseline 执行路径
### 性能问题与证据
### L1–L6 候选修改区域
### Directions
#### D01：标题
- 修改对象
- 因果链
- 公平比较
- Evidence
- 反例和退化条件
- 实现与测量
- [EXPERIMENT_REQUIRED] 交接
# 未形成 Direction 的保留信息
# Evidence 索引
# Workflow 范围、限制与验证
```

Renderer 不能：

- 引用 uncommitted/raw proposal；
- 把 testable 写成已验证；
- 隐藏 blocking contradiction；
- 把历史 `experiment_notes` 误写成本 run 新实验；
- 省略来源和适用条件。

### 8.3 Coverage

final coverage 至少覆盖：

1. topic scope；
2. performance opportunity overview；
3. Anchor summaries；
4. Direction statuses 及 modification/causal/comparison/implementation/falsifier；
5. Evidence provenance；
6. contradictions/limits；
7. ExperimentHandoffs；
8. unresolved questions。

## 9. `REQ-25`：Agent—脚本协议闭包

### 9.1 输入

每个 Turn 收到：

- TurnIdentity；
- StateBinding；
- Skill name/version/hash；
- 当前任务目标和禁止事项；
- frozen StageContract；
- approved canonical/artifact projection；
- permission envelope；
- expected output schema；
- Turn 结束条件。

Workflow Turn 额外收到 trigger、immutable objective/acceptance 和 plan projection。

### 9.2 输出

每个 Turn 只允许一个消息族：

```text
WORKFLOW_TURN_TASK       → WORKFLOW_DECISION_PROPOSAL
EVIDENCE_READER_TASK     → EVIDENCE_PACKET
DIRECTION_REVIEW_TASK    → REVIEW_DELTA
CLOSURE_REVIEW_TASK      → CLOSURE_REVIEW
```

每个 Agent 在输出前依据 expected schema、TurnIdentity、StateBinding、角色约束和“唯一顶层 JSON”规则完成自检。脚本收到 raw response 后可以进行 BOM、换行、单一 Markdown fence 和唯一 JSON 提取等无语义规范化；仍不合法时只能按原 StageContract 重新启动同角色 fresh Turn，并附 validator report。脚本不得猜测或补写业务字段，也不得为此启动辅助 Agent。

### 9.3 提交

```text
raw response
→ deterministic normalization
→ parse/schema
→ identity/state/hash
→ role/message
→ domain/security
→ frozen Gate
→ SQLite transaction
→ next trigger/task
```

Agent 结果在 commit 前不能成为下一 Agent 的事实。

若失败存在唯一机械处理，Controller 直接重试、reconcile 或终止；若存在多个合法恢复方向，则构造 `GATE_FAILED_WITHOUT_RECOVERY_RULE`、`NO_PROGRESS_THRESHOLD_REACHED`、`NO_RUNNABLE_STAGE` 或其他已注册 trigger，启动新的 Workflow Decision Turn。Workflow Agent 仍只输出普通 `WorkflowDecisionProposal`，不直接修复状态。

## 10. 动态 Workflow 如何闭合需求

Workflow Agent 不保存 plan 状态。动态性由以下组合提供：

```text
当前 authoritative snapshot
+ Simple Semantic Loop Skill
+ registered WorkflowTrigger
+ allowed WorkflowDecisionAction
+ WorkflowPlanPatch / StageContractDraft / GateDefinitionDraft
```

当固定路径足够时，Controller 直接推进；遇到语义分支时才调用 Workflow Turn。

例：

```text
EvidencePacket 提交
→ trigger: COMMITTED_RESULT_REQUIRES_INTEGRATION
→ Workflow Turn 提出 SemanticDelta script stage
→ Controller Gate/commit
→ 新 Direction 需要审阅
→ Workflow Turn 提出 DirectionReview stage
→ Reviewer result
→ Workflow Turn 决定继续 Search 或终态
```

因此脚本无需理解具体性能机制，但仍拥有状态和执行权。

## 11. 失败和停止

### 11.1 非完成状态

- waiting user/external；
- paused budget/operator；
- failed retriable/terminal；
- blocked semantic/external；
- cancelled。

这些状态都不会生成成功 final output。

### 11.2 唯一完成路径

```text
Workflow Turn PROPOSE_COMPLETE
→ Controller mechanical preflight
→ fresh Closure Reviewer accept
→ unchanged canonical revision
→ full validators
→ final.md render
→ coverage validation
→ atomic completed
```

No runnable、budget exhausted、no-progress 和模型自然语言“done”均不等于完成。

## 12. 需求闭包矩阵

| Requirement | 实现路径 | 必须保存的证据 | 验收测试 |
|---|---|---|---|
| `REQ-19` | Shared dimension enum + Evidence routing | SearchNeed dimension、path、Evidence source family | 同一问题按 intent 路由到不同维度；未授权目录拒绝 |
| `REQ-20` | Evidence Reader Omnisearch/deep read | tool events、query ledger、contextsRead | 真实只读 canary 至少完成一次 Omnisearch 和 section read |
| `REQ-21` | Workflow Turn 触发 SearchNeed；Reader 形成 Q1–Q3 | trigger、gap rationale、terms/provenance、stop reason | baseline、mechanism、implementation、not_found 四类 fixture |
| `REQ-23` | TopicFrame → Anchor → Direction → Review → closure | canonical revisions、Evidence refs、ReviewDelta、StopProof | 小 Topic 形成至少一个可理解 Direction 或有证据的 rejected/experiment-required 结论 |
| `REQ-24` | structured process + deterministic renderer | workflow.db、exports、final.md、coverage report | 人工检查 final.md；每个主要主张可回溯 canonical Evidence |
| `REQ-25` | Shared Envelope + Stage/Gate + Controller CAS | task/prompt/result/schema/hash/validation/event | 删除全部 provider history 后，仅用 workflow.db 恢复并完成下一 Turn |
| `REQ-28` | 固定 L1–L6 registry + Topic layerScope + Direction subset validator | Topic revision、ModificationAtom、validator report | Topic 之外的修改层在 pre-Gate 阶段被拒绝 |
| `REQ-29` | Anchor schema/validator/renderer | scenario、baseline path/config、performance tension、constraints、Evidence | 缺任一 Anchor 不变量即拒绝；final.md 完整呈现 |
| `REQ-30` | Direction schema + independent Reviewer + renderer | ModificationAtom、causalLinks、comparison、implementation、falsifiers、ReviewDelta | 四分支审阅矩阵和可读 Direction bundle |
| `REQ-31` | semantic trigger + dynamic plan + result consumption loop | event、plan revision、SearchNeed/Review integration | E2E 至少经历查询、集成、审阅、重开/闭包 |
| `REQ-34` | SQLite/CAS/event log + captured-raw local replay | workflow.db、raw hash、attempt、recovery event | 杀进程后不依赖 provider history 恢复；已落盘 raw 不重复调用 |
| `REQ-35` | 四类 fresh Turn 顺序调度 + immutable Skill hash binding | task/attempt/Skill hash/result artifact | 无持久 Agent、无 Skill 状态写入、每次新 thread |

## 13. 端到端验收场景

### E2E-1：Baseline 缺口

1. 输入一个有明确 workload 但 baseline 路径不清的 Topic。
2. Workflow Turn 创建 `define_baseline` SearchNeed。
3. Evidence Reader 路由 idea + experiment。
4. Evidence 形成 Anchor。
5. 验证 query terms、contexts 和 sources。

覆盖：`REQ-19`、`REQ-20`、`REQ-21`、`REQ-25`。

### E2E-2：Direction 形成与审阅

1. Evidence 支持一个修改对象和机制。
2. Workflow Turn 形成 Direction。
3. Direction Reviewer 找到一个可知识回答的因果缺口。
4. 新 SearchNeed 只查询 knowledge。
5. 新 Evidence 后 Direction 进入 testable 或 experiment-required。

覆盖：`REQ-21`、`REQ-23`、`REQ-25`、`REQ-28`、`REQ-29`、
`REQ-30`、`REQ-31`。

### E2E-3：Closure reject

1. 构造仍有 critical Need 的 StopCandidate。
2. preflight 或 Closure Reviewer reject。
3. Controller 触发新的 Workflow Turn。
4. 旧候选作废，run 不 completed。

覆盖：`REQ-23`、`REQ-25`、`REQ-31`。

### E2E-4：最终输出

1. 构造一个已闭环的小 Topic。
2. Closure Reviewer accept。
3. Controller render final.md。
4. Coverage validator 逐项解析 canonical refs。
5. workflow.db 在删除 provider history 后仍可重放。

覆盖：`REQ-23`、`REQ-24`、`REQ-25`、`REQ-29`、`REQ-30`、
`REQ-34`、`REQ-35`。

### E2E-5：No Experiment

1. Direction Reviewer 输出 experiment-required。
2. Controller 提交不可执行 handoff。
3. 验证 task queue 无实验 Stage。
4. 验证 tool event 无 shell/build/benchmark/profile/GPU。

覆盖整个 workflow 的安全边界。

## 14. 实现顺序

```text
WF-0 Shared contracts 完成
WF-1 Controller store/state/plan/runtime
WF-2 Workflow Turn
WF-3 Evidence Reader
WF-4 Direction Reviewer
WF-5 Closure Reviewer
WF-6 Renderer / coverage
WF-7 E2E fixtures
WF-8 read-only Obsidian canary
```

每一步以对应组件计划的 Definition of Done 为进入下一步的门。

## 15. Workflow 完成证据

整套 workflow 实现完成时，必须提供：

1. shared schema manifest 和全部 validator 测试报告；
2. Controller database migration/CAS/replay 测试报告；
3. 四类 Turn 的 role-level fixture 报告；
4. E2E-1 至 E2E-5 的 run directories；
5. 至少一个真实只读 Obsidian canary；
6. 每个 requirement 的 artifact/ref/test 映射；
7. final.md 人工可读性检查记录；
8. provider history 删除后的恢复证据；
9. Closure reject 与 accept 两条路径证据；
10. No Experiment runtime event audit。

只有 `REQ-19`、`REQ-20`、`REQ-21`、`REQ-23`、`REQ-24`、
`REQ-25`、`REQ-28`、`REQ-29`、`REQ-30`、`REQ-31`、`REQ-34`
和 `REQ-35` 均在闭包矩阵中有通过证据，才可以声明 workflow 完成。
