# Learning Workflow 调度控制型 Agent 重构策略

> 状态：重构设计总策略，尚未进入协议和实现修改。  
> 范围：明确 Controller、调度控制型 Agent、内容产出型 Agent、Skill、Ref 和 Prompt 之间的职责边界。  
> 原则：先冻结职责与信息流，再设计精简 JSON；本文件不定义最终 JSON Schema。

## 1. 核心结论

调度控制型 Agent 的唯一职责应当是：

> 以最终需求为判断基准，根据 Controller 提供的权威脚本状态和其他 Agent 已校验的结论，理解当前工作流处境，并从脚本本次给出的有限决策选项中选择一个决策。

它不是内容生产者，也不是状态库、任务执行器或任意工作流规划器。

目标架构是：

```text
最终需求 + Controller 权威状态 + 已校验的 Agent 结论
                         │
                         ▼
              Controller 先执行确定性判断
                         │
              需要语义决策时才调用
                         ▼
                调度控制型 Agent
              从脚本给定选项中选一个
                         │
                         ▼
              Controller 校验并落实决策
                         │
                         ▼
       内容 Agent / Reviewer 执行一个具体 Turn
                         │
                         ▼
                返回可校验的内容结论
                         │
                         └──────────→ 写入状态并进入下一轮
```

动态工作流来自“脚本根据状态生成当前可选决策，调度 Agent 在有限选项中作语义选择”，而不是让调度 Agent自行编写阶段合同、Gate、DAG 或全局状态。

## 2. 当前设计需要修正的问题

现有 `workflow_decision` 职能混合了两类工作：

1. 调度控制：理解触发原因、判断下一步、重试、重规划或结束。
2. 专业内容：创建 TopicFrame、SearchNeed、Anchor、SemanticDelta、StageContract、GateDefinition 和 PlanPatch。

这会产生几个直接问题：

- 调度输入需要携带大量领域对象、哈希、日志和协议字段；
- 调度输出同时包含决策、专业内容和控制结构，JSON 过长且容易局部错误；
- Agent 在同一个 Turn 中既判断“做什么”，又定义“怎么验收”，还生成“实际内容”；
- Controller 很难区分“决策错误”“内容错误”和“协议错误”；
- 重试时只能把大量错误重新交给同一个 Agent，修复范围不清晰；
- 专业 Ref 和工具使用挤占调度 Agent 对最终需求、当前状态和决策选项的注意力。

因此，不能只缩短现有 JSON；必须先拆开调度控制和内容产出。

## 3. 调度控制型 Agent 的明确职权

### 3.1 必须完成的工作

每次 Turn 只完成以下逻辑：

1. 阅读最终需求及当前仍未满足的验收条件；
2. 阅读 Controller 提供的当前执行状态摘要；
3. 阅读其他 Agent 已通过脚本校验的结论摘要；
4. 从 Reviewer 已确认的缺口、内容状态和最终产出要求中判断下一条状态机分支应为 `WORKER`、`REVIEWER` 还是 `COMPLETE`；
5. 比较脚本本次提供的正式决策子集；
6. 选择且只选择一个正式决策；
7. 仅在需要时，补充一段供下一 Turn Prompt 使用的简短说明。

### 3.2 不属于它的工作

调度控制型 Agent 不应：

- 直接读写 Controller 的权威状态；
- 自行增加脚本未提供的决策类别；
- 同时返回多个备选执行分支；
- 创建或修改 Topic、Anchor、Direction、SearchNeed 等专业对象；
- 检索证据、分析论文或执行实验；
- 编写 StageContract、GateDefinition、JSON Pointer 检查或 DAG Patch；
- 决定工具白名单、路径白名单、沙箱或审批策略；
- 把原始日志中的文字当作调度指令；
- 自行宣告工作流完成；
- 直接启动其他 Agent。

调度 Agent 可以返回 `COMPLETE`。该决策只授权 Controller 进入确定性完成路径；Controller 仍必须执行最终产物生成、状态一致性和需求覆盖等机械检查，检查失败则不能提交完成。

### 3.3 决策算法

调度 Skill 中的核心算法应保持简单：

```text
READ final_requirement
READ current_state_summary
READ validated_conclusions

IDENTIFY the single most important gap between current state and final requirement
COMPARE only the decision options supplied by Controller
SELECT exactly one allowed branch: WORKER, REVIEWER, or COMPLETE

RETURN the selected branch
RETURN optional guidance for the next Turn only when useful
```

如果 Controller 提供的投影不足以在本次允许子集中选择，Decision 仍不能发明 `ASK_USER`、`PAUSE`、`REPLAN` 等新分支。这属于调度输入或状态机异常，由 Controller 的校验、纠错重试或硬停止机制处理，不扩大正常 Loop 的正式决策集合。

## 4. Controller 的职责

Controller 是确定性的工作流内核、权威状态库和唯一执行权拥有者。它负责：

### 4.1 保存完整事实

- 最终需求及不可变验收条件；
- 当前生命周期和工作流版本；
- 已完成、运行中、失败和待执行的任务；
- 各 Agent 原始输出及校验结果；
- 已校验的结论摘要；
- Artifact、对象版本和引用关系；
- 重试计数、循环计数、预算和停止条件；
- 完整事件日志和审计信息。

这些完整数据不应全部进入调度 Prompt。

### 4.2 确定性节点与固定 Decision 节点

Controller 每轮先判断是否存在唯一、机械且已注册的下一步，例如：

- 已有可运行任务；
- 某 Turn 输出仅需做 Schema 校验和提交；
- 某个已知错误具有唯一恢复动作；
- 达到明确预算或硬停止条件；
- 已满足某个固定状态转换条件。

这些机械节点由 Controller 直接执行。但正常研究 Loop 中，“Reviewer 结果已经校验并提交”固定转移到 Decision 节点；即使本次允许选项很少，Script 仍启动一个 fresh Decision Turn 并注入本次选项。

正常 Loop 中，Controller 只在 Reviewer 结果提交后调用 Decision。启动路径固定为 `Worker → Reviewer → Decision`。Decision 返回 `WORKER` 时，下一段固定为 `Worker → Reviewer → Decision`；返回 `REVIEWER` 时，下一段固定为 `Reviewer → Worker → Reviewer → Decision`；返回 `COMPLETE` 时进入 Controller 的确定性完成路径。

### 4.3 生成本次决策选项

脚本根据当前状态，从以下三个正式决策中生成一个很小的本次允许子集：

- `WORKER`：进入 `Worker → Reviewer → Decision` 分支；
- `REVIEWER`：进入 `Reviewer → Worker → Reviewer → Decision` 分支；
- `COMPLETE`：进入 Controller 的确定性完成路径。

“增加 Anchor”“增加 Direction”“继续深入当前对象”“从新角度审阅”等不是正式决策，也不参与状态机分支匹配。它们只能作为可选 guidance，帮助下一 Worker 或 Reviewer 理解本轮关注点。即使没有 guidance，Controller 也必须能根据权威状态为所选分支实例化一个合法的默认任务。

Decision 选择的是状态机下一条边，而不是研究内容。其正式产出只需要给出 `WORKER`、`REVIEWER` 或 `COMPLETE` 中本次允许的一个；还可以附带一段可选 guidance。具体 Anchor、Direction、查询词、证据和专业判断由下一 Worker 或 Reviewer 形成；Controller 根据当前状态自动绑定对象并实例化正式任务。

每次调用 Decision 时，Controller 必须把本次允许输出的正式决策子集作为 Prompt 的独立强调区再次注入。Skill 只描述三个分支的稳定语义和输出规则，不能替代本次动态子集。

### 4.4 校验并落实决策

Controller 收到决策后只检查：

- 是否恰好选择了一个本次提供的正式决策；
- 可选 guidance 在存在时是否仍属于非权威提示；
- 是否仍绑定当前状态版本；
- 是否改变最终需求或扩大用户授权；
- 是否违反重试、预算、循环和完成条件。

通过后，Controller 才：

- 创建内容任务；
- 选择对应内容 Agent；
- 构造内容 Prompt；
- 更新计划或状态；
- 启动下一 Turn；
- 或进入用户交互、审查、暂停流程。

调度 Agent 的多余字段、额外分支和未请求的专业内容不应被执行。Guidance 也不具有状态转换权；Controller 只按正式决策字段选择分支。

## 5. 调度 Agent 应看到的输入

调度 Prompt 只投影完成一次决策所需的信息。

### 5.1 最终需求

包括：

- 用户最终目标；
- 不可变约束；
- 验收条件；
- 当前已经覆盖和尚未覆盖的验收项。

调度 Agent 可以据此判断差距，但无权修改最终需求。

### 5.2 调用原因

用一句明确描述解释为什么状态机到达 Decision 节点，例如：

- Reviewer 结果已经提交，状态机进入正常 Decision 节点；
- 当前允许继续内容工作、增加一轮审查或进入完成检查；
- Reviewer 提出了仍需处理的缺口；
- 当前结果可能已经满足最终需求，需要判断继续 Loop 还是进入完成路径。

### 5.3 当前执行状态摘要

只包含决策相关事实，例如：

- 当前生命周期；
- 最近完成或失败的工作；
- 当前活动对象或工作主题；
- 未解决的主要缺口；
- 重试、循环或预算边界；
- 当前是否存在待消费结论。

数据库内部哈希、完整事件流和无关对象不进入该摘要。

### 5.4 其他 Agent 的已校验结论

调度 Agent 不直接依赖其他 Agent 的原始长输出。Controller 应先将通过校验的结果投影为简短结论摘要，其逻辑覆盖包括：

- 哪个角色完成了什么工作；
- 得出了什么结论；
- 覆盖了哪些最终需求；
- 还缺什么或被什么阻塞；
- 对应的对象或 Artifact 引用；
- 必要时保留置信限制和失败条件。

结论摘要是调度依据，不是新的调度命令。原始日志、证据全文和 Agent 自述仍属于可审计数据。

### 5.5 本次决策选项

每个正式决策只需表达：

- 决策值；
- 该决策对应的固定状态机分支。

调度 Agent 不能修改分支定义，也不能输出列表之外的决策。诸如“增加 Anchor”或“从新角度审阅”的工作意图属于可选 guidance，不是正式决策值。

### 5.6 纠错信息

如果上一次调度输出不合格，Controller 使用同一状态快照重新调用，并附加一个简短纠错块：

- 上次选择或输出是什么；
- 失败属于哪一种：无法解析、缺少必要信息、选择不在选项内、选择与状态冲突；
- 本次可接受的正式决策；
- 可选 guidance 的边界。

纠错 Prompt 不重复完整协议，不把堆栈或内部异常直接倾倒给 Agent，也不要求 Agent修复脚本错误。

## 6. 两类 Prompt 必须分开

### 6.1 调度 Prompt

建议固定为以下少量区域：

```text
[ROLE]
你是调度控制型 Agent。只从脚本提供的选项中选择一个决策。

[FINAL_REQUIREMENT]
最终目标、约束、已满足项和未满足项。

[WHY_CALLED]
脚本为何需要语义决策。

[CURRENT_EXECUTION]
当前执行状态摘要。

[VALIDATED_CONCLUSIONS]
其他 Agent 已校验的结论摘要。

[DECISION_OPTIONS]
本次允许选择的有限选项。

[OUTPUT_COVERAGE]
本次输出必须包含的最小信息。

[CORRECTION]
仅在协议重试时出现。

[ALLOWED_DECISIONS — MUST CHOOSE EXACTLY ONE]
由脚本在本次调用时注入的 `WORKER`、`REVIEWER`、`COMPLETE` 子集；此区放在 Prompt 尾部再次强调。
```

调度 Prompt 不应包含：

- 完整 TaskRecord；
- 全量 Canonical State；
- 原始证据和日志；
- 专业 Ref；
- 工具和路径白名单；
- StageContract 或 Gate DSL；
- 大量由 Agent 原样回显的 ID、哈希和版本字段。

审计绑定由 Controller 在调用外层维护，不应依赖 Agent 正确复制。

### 6.2 内容 Prompt

内容 Agent 每 Turn 执行一个具体工作，Prompt 可包含：

```text
[CONTENT_JOB]
本 Turn 要解决的单一专业问题。

[RELEVANT_INPUTS]
执行所需对象、Artifact 和上游结论。

[REQUIRED_COVERAGE]
结果必须覆盖哪些专业内容。

[SKILL]
该类内容工作的核心方法。

[AVAILABLE_REFS]
按需读取的专业参考资料。

[PRIOR_CONCLUSIONS]
与本工作直接相关的既有结论。
```

内容 Prompt 可以更丰富，因为其失败只影响本项内容工作，不直接承担全局调度推进。

## 7. Skill 与 Ref 的分层策略

### 7.1 调度控制型 Agent

使用一个短小、稳定的调度 Skill：

- 明确唯一职责；
- 给出固定决策算法；
- 说明 `WORKER`、`REVIEWER`、`COMPLETE` 三个正式决策的稳定含义；
- 说明产出只包含一个正式决策和可选 guidance；
- 明确只能选择脚本选项；
- 明确不得生产专业内容或修改状态；
- 给出输出前自检；
- 不配置专业 Ref。

调度 Skill 不应复制完整工作流状态机或最终协议 Schema。当前状态、本次允许选项和输出覆盖由脚本每次直接注入并在 Prompt 尾部强调。

### 7.2 内容产出型 Agent

内容 Agent 的 Skill 描述稳定的专业方法，Ref 提供按任务类型拆分的专业知识、例子或工具使用说明：

- 思考型、检索型和专业分析型 Agent 可使用 Ref；
- 每个 Turn 只注入当前工作需要的 Ref 索引或路径；
- Agent 按需读取 Ref，不把所有参考资料预先塞进 Prompt；
- 内容 Agent 输出专业结论，不输出下一步调度命令。

### 7.3 Reviewer

研究 Loop 内的 Anchor/Direction Reviewer 属于独立专业评估角色，不属于调度控制角色：

- 可以有针对审查标准的 Skill 和 Ref；
- 研究 Loop 内每次只读评估一个指定 Anchor 或 Direction；
- 返回 verdict、依据、缺口和阻塞项；
- 不修改状态，不启动下一工作。

### 7.4 工具策略

在 YOLO 模式下，不再把工具名或路径白名单当作调度协议的核心组成，也不依赖 Prompt 声称的工具限制来提供安全性。

- Agent 的职能和结果覆盖由 Prompt、Skill 和结果校验区分；
- 内容 Agent 根据任务和 Skill 自主使用所需工具；
- 调度 Agent 通常没有使用工具的必要，但不为此增加复杂的逻辑白名单；
- Controller 可保留工具事件用于审计和诊断；
- 如果未来必须强制禁止某类副作用，应由实际沙箱或权限机制执行，而不是依赖 JSON 字段。

## 8. 内容职责从旧 Workflow Agent 迁出

缩减调度 Agent 之前，必须为旧 Workflow Agent 当前承担的专业内容找到明确接收者：

| 旧职责 | 新职责归属 |
|---|---|
| 判断下一步做什么 | 调度控制型 Agent |
| 从脚本选项中选择 Worker、Reviewer 或完成分支 | 调度控制型 Agent |
| 维护状态、计划、版本和循环限制 | Controller |
| 生成本次合法决策选项 | Controller |
| 理解最终需求、全局状态和所有 Agent 结论并选择 `WORKER`、`REVIEWER` 或 `COMPLETE` 分支 | Decision |
| 围绕一个指定 Anchor 深度调研并形成内容结论 | 内容 Agent 的 anchor research work |
| 审阅一个指定 Anchor 并提出该对象的局部缺口 | Reviewer |
| 围绕一个指定 Direction 深度调研并形成内容结论 | 内容 Agent 的 direction research work |
| 审阅一个指定 Direction 并提出该对象的局部缺口 | Reviewer |
| 登记结论、缺口并触发下一次全局决策 | Controller |
| 定义机械 Gate | Controller 中注册的任务类型 |
| 提交对象和生命周期转换 | Controller |

推荐优先使用一个通用 `Learning Worker`，通过单一 `job kind` 区分 anchor research 和 direction research，避免为每个细步骤都增加一个调度角色。每个 Worker Turn 必须绑定一个 Anchor 或 Direction 工作；Reviewer 的缺口也必须绑定其审阅的同一对象。若某项专业方法明显不同，再拆分专门 Worker。

无论采用一个还是多个内容 Agent，都必须保持：

- 每个 Turn 只有一个内容目标；
- Skill/Ref 与当前 `job kind` 对应；
- 输出只覆盖本项工作的专业结论；
- 是否继续、重试或转向仍由 Controller 与调度 Agent决定。

## 9. 先定义结果覆盖，再定义 JSON

当前阶段只冻结逻辑覆盖，不冻结字段名。

### 9.1 调度结果的最小覆盖

调度结果只需表达：

1. 从本次允许子集中选择的一个正式决策：`WORKER`、`REVIEWER` 或 `COMPLETE`；
2. 一段可省略的 guidance，用于提示下一内容 Turn 或审查 Turn 的关注点。

不要求 Agent 回显脚本已经知道的 workflow ID、run ID、task ID、版本、哈希、状态游标或完整选项内容。

### 9.2 内容结果的最小覆盖

由 `job kind` 决定，例如：

- framing：范围、目标、约束和未决问题；
- anchor research：指定 Anchor 的证据、场景、Baseline、性能矛盾、适用条件和未回答项；
- direction research：指定 Direction 的机制、证据、适用条件、冲突、代价和未回答项；
- anchor/direction review：指定对象的 verdict、依据和局部查询缺口；
- synthesis：整合结论、依据、限制和后续缺口；
- review：verdict、逐项审查结果、阻塞项和可修复建议。

### 9.3 JSON 设计的后置条件

只有以下内容冻结后，才设计新 JSON：

- Agent 类型和职责矩阵；
- Controller 的状态投影；
- 三个正式决策及可选 guidance 的边界；
- 内容 `job kind` 及结果覆盖矩阵；
- 重试和纠错分类；
- 完成性闭环。

届时应分别设计：

- 简单的调度输入与决策输出；
- 按 `job kind` 区分的内容输入与结果输出；
- Controller 内部的完整持久化记录。

三者不再共用一个巨大的万能协议。

## 10. 修改实施顺序

### 阶段 A：冻结职责

1. 确认调度控制型 Agent 的职权和禁止职责；
2. 确认内容 `job kind`；
3. 确认 Reviewer 是否保持独立；
4. 确认 Controller 对三个正式决策的按状态裁剪条件；
5. 为每种输入和输出定义逻辑覆盖。

此阶段不改 JSON 和运行代码。

### 阶段 B：重写设计文档

按本策略依次修改：

1. 根实现计划：改成“确定性 Controller + 有限选项调度 Agent + 内容 Agent”；
2. Workflow Turn 计划：重写为纯调度控制 Agent 计划；
3. Scheduler Script 计划：增加状态投影、选项生成、决策落实和纠错职责；
4. 内容 Agent 计划：接收 TopicFrame、SearchNeed、Evidence、Synthesis 等旧 Workflow 内容职责；
5. Reviewer 计划：保持独立专业评估，移除调度职能；
6. Shared Contracts 计划：最后根据覆盖矩阵设计精简协议。

### 阶段 C：设计协议

1. 先设计调度 Decision Context 和单决策结果；
2. 再设计内容 Job 和结论结果；
3. 最后设计 Controller 内部持久化记录与二者的映射；
4. 为结构错误和语义错误分别定义有限重试；
5. 对旧协议保留明确版本边界。

### 阶段 D：迁移 Skill 与 Prompt

1. 将 `learning-semantic-loop-workflow-turn` 缩减为简单调度 Skill；
2. 将其专业工作流知识迁移到内容 Skill 或按需 Ref；
3. 为不同 `job kind` 建立内容 Prompt 构造器；
4. 为调度 Turn 建立独立的最小 Prompt 构造器；
5. 删除 Prompt 中重复状态、复杂 Gate 和工具白名单描述。

### 阶段 E：修改 Controller

1. 增加确定性快速路径；
2. 增加调度触发判定；
3. 增加决策选项注册与按状态裁剪；
4. 增加结论摘要投影；
5. 增加单决策校验和任务实例化；
6. 增加针对结构错误、语义错误的纠错重试；
7. 保持 Controller 对状态提交和最终停止的唯一权威。

### 阶段 F：验证和迁移

至少验证：

- Worker/Reviewer 之间的确定性链由 Script 执行，而每次 Reviewer 结果提交后都触发 fresh Decision Turn；
- 调度 Agent 不能选择未提供的选项；
- 调度输出错误会收到短小、针对性的纠错输入；
- 内容 Agent 的失败不会破坏全局状态；
- 原始结论与调度摘要可以互相追溯；
- 调度 Prompt 不包含专业 Ref、完整日志和大对象；
- 工具事件不再作为内容正确性的替代条件；
- Decision 返回 `COMPLETE` 后，只有 Controller 的最终产物、状态一致性和需求覆盖检查全部通过才能提交；
- 中断恢复只依赖持久状态，不依赖任何 Agent 历史上下文。

现有运行应保留为旧协议审计数据。新架构应使用新的协议或运行版本，除非另行设计显式迁移，不应让旧运行在无提示情况下继续套用新语义。

## 11. 对现有文件的预期影响

本策略通过讨论后，后续需要重点重构：

- `draft/learning_workflow_simple_semantic_loop_implementation_plan.md`
- `draft/learning_workflow_simple_semantic_loop_plans/workflow_turn_agent_implementation_plan.md`
- `draft/learning_workflow_simple_semantic_loop_plans/scheduler_script_implementation_plan.md`
- 各内容 Turn 和 Reviewer 实现计划；
- `.codex/skills/learning-semantic-loop-workflow-turn/`
- 内容 Agent 对应 Skill 与 Ref；
- `scripts/simple_semantic_loop.ts` 及协议 Schema。

本文件只定义修改方向，不表示上述实现已经完成。

## 12. 策略验收标准

进入 JSON 和代码修改前，本策略应满足：

- 调度 Agent 的一句话职责没有内容生产和状态提交含义；
- 每个调度 Turn 只回答一个决策问题；
- 所有可选决策均由脚本本次明确提供；
- Agent 通过最终需求、脚本状态和已校验结论理解当前处境；
- 调度 Agent 不需要专业 Ref；
- 专业 Skill、Ref 和工具留在内容 Agent；
- Controller 仍保存完整状态、审计数据和执行权；
- 机械节点由 Controller 直接执行，但每次 Reviewer 结果提交后固定进入 Decision；
- 调度结果只保留一个正式分支决策和可选 guidance；
- 最终 JSON 在职责、选项和覆盖矩阵冻结之后设计。

## 13. 协议设计前仍需确认的事项

1. 是否采用一个通用 `Learning Worker + job kind`，还是保留独立 Evidence Reader 并新增 Synthesis Worker；
2. 内容 `job kind` 的最小集合；
3. `WORKER`、`REVIEWER`、`COMPLETE` 各自在什么状态下可被加入本次允许子集；
4. 每种内容结论的必需覆盖项；
5. 旧运行采用只读保留、显式迁移，还是重新初始化；
6. 可选 Prompt 说明的长度、允许内容和校验边界。

建议先确认前四项，再开始精简 JSON。否则只是把当前复杂职责换成另一套字段，无法真正降低调度失败率。
