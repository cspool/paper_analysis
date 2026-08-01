# Simple Semantic Loop 实现计划入口

> 本文件是计划包索引，不再承载可直接编码的全部细节。实现任务、接口、测试和完成证据均拆分到 `learning_workflow_simple_semantic_loop_plans/`。

## 1. 需求基线

本计划包把 [`todo_draft.md`](todo_draft.md) 第 19–35 行中的有效需求作为必须闭合的用户需求：

| ID | 来源行 | 必须闭合的需求 |
|---|---:|---|
| `REQ-19` | 19 | 将 HPC 论文知识库视为具有 experiment、idea/baseline、knowledge、human 等不同表达维度的仓库 |
| `REQ-20` | 20 | 不同维度按不同查询需求使用，并通过 Obsidian Omnisearch 查询 |
| `REQ-21` | 21 | 明确查询关键词如何形成、查询何时触发 |
| `REQ-23` | 23 | 输入一个 Topic，最终输出该 Topic 相关的进一步性能优化潜力 |
| `REQ-24` | 24 | 过程产物按工作需要组织，最终产物必须便于人阅读理解 |
| `REQ-25` | 25 | 定义 Agent—脚本协议，并把格式化状态作为 Agent 的思考输入 |
| `REQ-28` | 28 | L1–L6 是可修改性能对象的坐标；Topic 必须限定允许搜索与组合的层子集 |
| `REQ-29` | 29 | Anchor 以场景、baseline 和性能张力圈定一个可延伸的局部搜索区域 |
| `REQ-30` | 30 | Direction 是 Anchor 区域内经探索、反证和独立审阅的候选优化方向 |
| `REQ-31` | 31 | 工作流使用可重复的搜索—集成—审阅语义 Loop，而非一次固定串行问答 |
| `REQ-34` | 34 | 调度机制必须解决长上下文中的运行状态丢失，并让状态可恢复、可审计 |
| `REQ-35` | 35 | 脚本按权威状态顺序调度单次 Agent；稳定 Skill 与运行期输出/状态分离 |

本实现不保留长期存活 Agent；它以“Controller 持久状态 + fresh Turn”的方式
闭合 `REQ-34` 的底层问题。Skill 保存稳定方法，运行状态、事件和产出由
Controller 保存，不能让 Skill 自行变成运行期数据库。

## 2. 最终调度架构

```text
持久存在
  Deterministic Workflow Controller
  + transactional state / event log
  + revisioned WorkflowPlan
  + frozen StageContract / GateDefinition
  + canonical domain objects / artifact manifests

按需存在
  Workflow Decision Turn
  Evidence Reader Turn
  Direction Reviewer Turn
  Closure Reviewer Turn
```

全局规则：

1. 脚本是唯一状态写入者、任务启动者和完成提交者。
2. 所有 LLM Agent 都只执行一个 fresh Turn，输出一个结构化结果后退出。
3. Workflow Turn 通过领域 Skill 理解当前状态，只提出 `WorkflowDecisionProposal`。
4. 动态工作流通过 revisioned plan、StageContractDraft 和 GateDefinitionDraft 表达，不通过持久 Agent 上下文表达。
5. Worker 不定义或判断自己的成功条件。
6. 只有 Closure Reviewer 接受且脚本最终校验通过，run 才能进入 `completed`。
7. 不使用 provider Session、聊天历史、Agent checkpoint 或 active Codex Goal 作为 workflow 状态。
8. 本 workflow 只读取已有实验知识，不执行任何新研究实验。
9. reasoning effort 固定为 `workflow_decision = max`，其他三类 Agent Turn 均为 `high`，不得由 run config 或 proposal 降级。
10. Controller 只调度上述四类 Turn；不存在独立的格式修复或异常恢复 Agent。
11. 每类 Agent 在结束前自检输出；Controller 仍假设结构、绑定或语义都可能
    错误。pre-Gate 校验失败时，同一 logical task 以 fresh attempt 重试，并
    注入上一份 Controller ValidationReport 的有界错误包。
12. Workflow Agent 只提出 Stage 特定 Gate 条件；Controller 编译并注入强制
    Gate、冻结和执行。合法结果的 Gate failure 不进入同任务纠错。
13. 确定性恢复由脚本执行；需要语义取舍的异常统一触发新的 Workflow Decision Turn。

## 3. 可直接执行的计划包

目录中只有六份 implementation plan：

```text
1 份 Workflow 需求闭包计划
+ 1 份 Scheduler 实现计划
+ 4 份脚本可调用的 Turn Agent 实现计划
```

`shared_contracts.md` 是这六份计划共用的接口规范，不是第七个实施项目。

### 3.1 共享基础

| 文档 | 实现责任 |
|---|---|
| [共享契约规范](learning_workflow_simple_semantic_loop_plans/shared_contracts.md) | 领域对象、控制对象、Turn Envelope、JSON Schema、validator、fixture 和版本/hash 规则的唯一来源 |
| [Workflow 需求闭包计划](learning_workflow_simple_semantic_loop_plans/workflow_implementation_plan.md) | 记录脚本和各 Turn 如何端到端闭合上述十二项需求 |
| [Scheduler Script 实现计划](learning_workflow_simple_semantic_loop_plans/scheduler_script_implementation_plan.md) | 确定性状态内核、动态 workflow、Workflow Agent 设置、Turn 调度、Gate、恢复和完成提交 |

### 3.2 每类具体 Turn Agent

| Turn | 实现计划 | 类型 |
|---|---|---|
| Workflow Decision Turn | [Workflow Turn Agent 实现计划](learning_workflow_simple_semantic_loop_plans/workflow_turn_agent_implementation_plan.md) | 语义决策 |
| Evidence Reader Turn | [Evidence Reader Agent 实现计划](learning_workflow_simple_semantic_loop_plans/evidence_reader_agent_implementation_plan.md) | 只读知识检索 Worker |
| Direction Reviewer Turn | [Direction Reviewer Agent 实现计划](learning_workflow_simple_semantic_loop_plans/direction_reviewer_agent_implementation_plan.md) | 单 Direction Semantic Evaluator |
| Closure Reviewer Turn | [Closure Reviewer Agent 实现计划](learning_workflow_simple_semantic_loop_plans/closure_reviewer_agent_implementation_plan.md) | Topic 闭包 Semantic Evaluator |

Direction Reviewer 和 Closure Reviewer 是当前 workflow 仅有的两个具体 Semantic Evaluator。不得实现一个可接受任意 rubric、任意权限和任意业务对象的“通用 Evaluator Agent”。新增 evaluator 类型前必须增加独立计划、Skill、schema、role profile 和测试。

四份 Agent 实现计划对应脚本唯一可启动的四个角色。Agent 执行中所需的内部分析、格式自检或问题求解是该角色自身职责，不形成新的可调度角色或实施计划。

## 4. 文档权威顺序

发生冲突时按以下顺序处理：

```text
todo_draft.md 第 19–35 行的用户需求
→ Workflow 需求闭包计划中的验收行为
→ 共享契约规范
→ Scheduler 与具体 Turn 实现计划
→ Skill 内部提示词和实现注释
```

Agent Skill 不能修改协议；Scheduler 不能静默改变领域对象；Workflow 计划不能用新的隐藏需求替换上述十二项需求。

## 5. 推荐实施顺序

```text
P0 共享契约
  ↓
P1 Controller store / events / snapshot / CAS / runtime
  ↓
P2 Workflow Turn + Evidence Reader
  ↓
P3 Direction Reviewer + Closure Reviewer
  ↓
P4 端到端 Loop、Renderer 和需求闭包测试
```

每一步都必须先完成对应计划中的自动化测试和完成证据，再进入下一个依赖步骤。不得先实现自由文本 Agent 调度，再事后补 schema 或 Gate。

## 6. 第一版允许的动态性

Workflow Turn 可以提出：

- 新增、替换或废弃 Stage；
- 选择一个非等价 runnable Stage；
- 为一个 Stage 提出冻结前的合同和 Gate 草案；
- 在 Gate 失败、证据冲突或 Closure reject 后重排计划；
- 请求用户、报告阻塞、建议暂停或提出闭包候选。

第一版不允许：

- Agent 创建未知 Stage 类型或未知角色；
- Agent 注入任意代码、shell、正则执行环境或 Gate 表达式；
- Agent 修改 immutable objective、acceptance criteria 或权限边界；
- 动态创建没有独立计划的 Agent 类型；
- 以“没有 runnable stage”“预算耗尽”或“模型说完成”提交成功。

## 7. 计划包完成标准

只有以下条件全部成立，才可以开始宣称“计划已经可直接实行”：

1. 每个计划都有明确输入、输出、依赖、交付文件、工作包、测试和 Definition of Done。
2. 所有 Turn 均是 fresh one-turn，且没有 checkpoint、resume point 或 provider Session 恢复语义。
3. Scheduler 计划完整说明 Workflow Agent 的 Skill、输入快照、权限、输出 schema 和动态 plan patch 校验。
4. 共享 schema 对所有计划只有一个定义来源。
5. Workflow 计划为上述十二项需求逐一指定组件、运行路径和验收证据。
6. 所有本地 Markdown 链接有效。
7. 端到端测试能够证明从 Topic 输入到 `final.md`，中间经过按需查询、证据集成、Direction 审阅和 Closure Review。
8. 整个路径没有实际实验、目标代码修改或 Agent 直接调度事件。
9. role、Stage、schema 和 dispatch registry 中只有四类 Agent；结构、绑定和
   pre-Gate 语义错误走同角色 fresh retry，Gate failure 或其他需要取舍的异常
   走 Workflow trigger。
10. L1–L6 语义固定，Direction 修改层必须是 Topic `layerScope` 的子集；最终
    Markdown 展示 Anchor baseline/张力和 Direction 修改、因果、公平比较、
    实现、反证、退化条件及来源。
