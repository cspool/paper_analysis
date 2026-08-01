# 最小 Controller 校验与语义判断移交设计

> 状态：已实施；活动实现使用 format version 5。06 增加的运输状态、观察层和
> recovery 仍只做机械判断，不改变本文“Script 只校验安全调度核心”的边界。  
> 适用范围：`scripts/simple_semantic_loop/refactor/`、三类活动 Skill/Ref、
> 通信契约、持久化审计、最终 Renderer 和测试。  
> 前置设计：`01_agent_types_and_scheduling_responsibilities.md`、
> `02_loop_requirement_closure_design.md`、
> `03_script_agent_message_and_storage_contract_design.md`、
> `04_script_agent_json_communication_contract_inventory.md`。  
> 本文修正 03、04 中“由 Script 对完整内容 Schema 和跨字段内容关系进行
> 阻断性校验”的设计。后续实施时必须同步修改 03、04，不能让旧描述继续作为
> 权威契约。

## 1. 已确认的核心原则

Controller Script 的主要职责只有：

1. 根据当前状态、Agent 输出中的核心控制字面量和已定义状态机启动下一个
   Agent Turn；
2. 完整保存 Agent 原始输出、协议结果、Turn、Round、对象引用、状态变更和
   事件，并建立可恢复、可追踪的索引；
3. 只校验安全推进状态机直接需要的信息，不承担专业内容或流程语义判断。

判断某项校验是否属于 Script 的标准不是：

> Script 能否用代码写出这个检查。

而是：

> 如果不执行这个检查，Script 是否无法安全、确定地选择状态转换或保存状态。

若一个检查需要理解 Goal、Task、Anchor、Direction、证据、finding、
query gap 或自然语言之间的关系，它就属于 Reviewer 或 Decision，而不是
Script。

## 2. 为什么必须削弱强校验

完整 Schema 和预定义跨字段规则有三个根本问题：

1. 它们只能覆盖实现者预先想到的情况，无法穷尽动态研究工作流中的合法表达；
2. 它们可以保证固定结构，却不能保证 baseline、机制、证据、结论或 verdict
   在语义上正确；
3. 它们会把可由 Reviewer/Decision 理解和修复的内容问题提前误判为通信错误，
   消耗格式重试预算，甚至错误终止工作流。

因此：

- “Agent 输出不可信”不等于“Script 必须验证全部正文”；
- “Agent 输出不可信”应当表示 Agent 不能直接修改状态，只能提交内容结果或
  决策建议；
- Script 从结果中提取少量控制字段；
- Reviewer 和 Decision 对正文进行智能判断；
- 只有 Script 执行最终状态转换。

## 3. 三层错误分类

### 3.1 传输与控制错误

负责者：Script。

包括：

- Codex Turn 没有唯一、明确的协议结果消息；
- 最终结果不是可解析 JSON；
- JSON 顶层不是对象；
- Worker 缺少合法 `workOutcome`；
- Reviewer 缺少合法 `reviewVerdict`；
- Decision 缺少唯一、合法且在本次允许集合内的 `decision`；
- 状态转换所需的 TaskBinding、pending pair 或引用不存在。

这类错误不能安全进入下一个状态，由 Script 生成精简错误报告并重做同一 Turn。

### 3.2 内容与流程语义错误

负责者：Reviewer 和 Decision。

包括：

- Worker 改变 Topic、目标、Task 或对象绑定；
- Anchor 的场景、baseline、性能矛盾或 6L 范围不成立；
- Direction 越出 Anchor、机制不可证伪或 baseline change 不清楚；
- evidence 不充分、事实与假设混淆；
- 关键指标、guardrail、反例、失败条件或测量方法缺失；
- `workOutcome` 与正文的实际完成程度不一致；
- Reviewer findings、query gaps 和 verdict 的语义不一致；
- Reviewer 漏掉会阻止需求闭合的问题；
- Worker/Reviewer 的结果会使 workflow 走向错误分支；
- 现有结论是否已经真正满足最终需求。

这类问题不得触发 Script 的格式纠错。结果先被保存并进入正常
Worker → Reviewer → Decision 链路，再由 Decision 选择正常分支或
`RETRY_WORKER` / `RETRY_REVIEWER`。

### 3.3 运行与持久状态错误

负责者：Script。

包括：

- Provider Turn 超时、失败或中断；
- 状态 revision 冲突；
- 路径或引用不存在；
- 对象归属、revision 或 pending 记录与 TaskBinding 冲突；
- 重试、轮次或运行预算耗尽；
- 持久化文件损坏；
- 同一 revision 被不同结果重复提交。

这类错误按确定性恢复、暂停或失败策略处理，不交给 Agent 猜测。

## 4. Script 的阻断性校验白名单

后续实现中，只有本节列出的检查可以直接阻止状态机推进。

### 4.1 Codex 传输边界

- 保留全部 Provider 事件和 Agent commentary；
- 只将唯一的 `phase=final_answer` 消息作为协议 payload；
- 旧 Provider 兼容仅允许整个 Turn 恰好一个非空、phase-unknown 消息；
- 不拼接多个 Agent 消息；
- 多个 final answer 或无法确定协议消息时按传输错误处理。

### 4.2 Worker 核心字段

Script 只要求：

```json
{
  "workOutcome": "READY_FOR_REVIEW | PARTIAL_RESULT | BLOCKED_NO_RESULT"
}
```

完整输出仍可包含：

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {},
  "evidence": [],
  "unresolved": []
}
```

但除 `workOutcome` 外，其他字段均作为 Agent 内容保存，不作为 Script
状态转换的阻断性 Schema。

Script 不再判断：

- `content` 是否符合固定 Anchor/Direction 字段集合；
- `content` 与 `workOutcome` 是否语义一致；
- `unresolved` 是否必须为空或非空；
- evidence、数组或字符串的数量、长度；
- 6L 是否至少一个非空层；
- 是否存在未知内容字段。

### 4.3 Reviewer 核心字段

Script 只要求：

```json
{
  "reviewVerdict": "PASS | REVISE | REJECT"
}
```

`summary`、`findings` 和 `queryGaps` 仍由 Reviewer Ref 定义和推荐，并被
原样保存，但不由 Script 对其完整结构和语义关系进行阻断性校验。

Script 不再判断：

- `PASS` 是否包含 blocking finding；
- `REVISE` / `REJECT` 是否至少有一个 blocking finding；
- PARTIAL/BLOCKED Worker Result 是否可以得到 PASS；
- query gap 的维度、内容或与 finding 的对应关系；
- REVISE 与 REJECT 的可修复性差异。

这些都是 Decision 在读取 Reviewer 结果时必须复核的语义。

### 4.4 Decision 核心字段

Script 只提取：

```text
decision = RUN_WORKER | RUN_REVIEWER | RETRY_WORKER |
           RETRY_REVIEWER | FINISH_WORKFLOW
guidance = 可选自然语言
```

阻断性检查只有：

- 能提取唯一 decision；
- decision 在本次 Script 注入的允许集合内；
- 目标分支具备对应状态前提；
- 重试和轮次预算允许执行；
- `FINISH_WORKFLOW` 满足最小机械闭环。

Script 不再：

- 要求 guidance 包含某个 Ref 名称；
- 从 guidance 搜索 Anchor、Direction、create、deepen、review 等关键词；
- 根据 guidance 内容选择 Task；
- 判断 guidance 是否提出了正确语义；
- 用任意固定字符数表达专业内容是否“足够精简”。

guidance 原样保存并原样附加到后续 Agent Prompt。

## 5. 最小机械闭环

`FINISH_WORKFLOW` 仍需要 Script 防止明显不完整的状态被提交。该检查只使用
Agent 已给出的控制字面量和 Script 自身索引，不读取正文语义。

建议保留：

- 至少存在一个未被 `REJECT` 的 Anchor；
- 每个最终 Anchor 当前 Work Result 的 `workOutcome` 为
  `READY_FOR_REVIEW`；
- 每个最终 Anchor 当前 Reviewer `reviewVerdict` 为 `PASS`；
- 每个最终 Anchor 至少绑定一个未被 `REJECT` 的 Direction；
- 每个最终 Direction 当前 Work Result 的 `workOutcome` 为
  `READY_FOR_REVIEW`；
- 每个最终 Direction 当前 Reviewer `reviewVerdict` 为 `PASS`；
- Direction 仍绑定当前 Anchor revision；
- 没有未处理 pending pair；
- Decision 明确选择 `FINISH_WORKFLOW`。

这些条件只说明：

> 状态机的最低结构性闭环已经成立。

它们不说明：

> 研究内容在语义上已经满足用户需求。

后一个判断只由 Decision 完成。

## 6. Reviewer 和 Decision 的语义接管

### 6.1 Reviewer

Reviewer 继续独立审阅一个 Script 绑定的 Work Result，负责：

- 按 Worker Result Ref 理解预期正文；
- 判断正文是否遵循 Goal、Task 和对象范围；
- 判断 `workOutcome` 是否诚实反映结果；
- 判断专业正确性、证据、深度、反例、失败条件和测量方法；
- 记录 findings 和对象局部 query gaps；
- 给出 `PASS`、`REVISE` 或 `REJECT`。

Script 不预先拒绝“字段不完整但仍可理解”的 Worker 结果；Reviewer 必须看到
它并作出语义判断。

### 6.2 Decision

Decision 必须读取：

- Workflow Goal；
- 当前 pending Worker Task；
- pending Worker Result；
- pending Reviewer Result；
- 所有当前已提交的 Agent 结论；
- Script 计算的最小机械 requirement；
- 本次允许决策集合。

Decision 负责复核：

- Worker 是否遵循 Result Ref 和 Task；
- Reviewer 是否审阅了正确对象；
- Reviewer 的 verdict 是否与其结论及实际内容一致；
- 任一语义错误是否会阻止需求闭合或导致错误 workflow 分支；
- 是正常进入下一轮、增加审阅、重试 Worker、重试 Reviewer，还是完成需求。

当发现语义错误时：

```text
Worker 语义错误
→ RETRY_WORKER
→ guidance 说明错误、期望和正确 Result Ref

Reviewer 语义错误
→ RETRY_REVIEWER
→ guidance 说明漏审、误判或正确 verdict 逻辑
```

Script 只执行上述决策，不验证 guidance 的自然语言内容。

## 7. Provider outputSchema 策略

完整 Anchor、Direction、Review Schema 不应继续作为 Provider 的强制
`outputSchema`，因为这会在 Agent 结果进入 Reviewer/Decision 前阻断合法但
非预期的表达。

后续实现优先采用：

1. Worker/Reviewer 仍按 Skill + Ref 直接输出一个 JSON 对象；
2. 不向 Provider 发送完整正文 `outputSchema`；
3. Script 对最终消息执行 JSON parse；
4. Script 只提取并校验本角色的核心控制字段；
5. 完整对象原样保存。

若后续确认 Codex Provider 可以表达“不限制 content 正文”的最小 envelope，
可使用只约束核心控制字段的 Provider Schema；不得为了使用 Structured
Outputs 再次封闭正文结构。

## 8. Skill 和 Ref 的定位

完整模板仍然必要，但其作用改变为：

```text
Skill
  = 角色、目标、方法、约束和执行过程

Ref
  = 推荐输出结构、字段语义和专业内容合同

Script gate
  = 状态机直接需要的核心控制字段
```

后续必须删除或改写 Skill/Ref 中以下旧表述：

- “Script validates every cross-field rule”；
- “unknown fields are rejected by Script”；
- “Provider Schema enforces the complete content shape”；
- “verdict/blocking consistency is a Script mechanical gate”。

替换为：

- Agent 应尽量完整遵循 Ref；
- Reviewer 检查 Worker 是否遵循 Work Result Ref；
- Decision 检查 Worker/Reviewer 的语义 follow 能力；
- Script 只校验可安全调度所需的控制字段。

## 9. 持久化和索引

削弱校验不能削弱审计能力。每个 Turn 仍必须保存：

- Prompt；
- 全部 Runtime/Provider 事件；
- commentary 和工具事件；
- 唯一协议 payload；
- 原始 JSON 文本；
- 解析后的完整 JSON 对象；
- 提取出的核心控制字段；
- 格式/控制校验结果；
- TaskBinding；
- Turn 状态和 retry 关系；
- Round 和事件索引。

对象索引只复制状态机直接使用的信息：

- 对象 ID 和 revision；
- Work Result 引用；
- `workOutcome`；
- Review Result 引用；
- `reviewVerdict`；
- 父 Anchor；
- 提交 Decision Turn；
- rejected/active 状态。

Anchor、Direction、finding、query gap 和 evidence 正文只通过结果引用读取，不
复制进状态机索引。

完整 Schema 检查若为了诊断保留，只能写成非阻断 audit：

```text
advisory / warning
```

它不能：

- 把 Turn 标记为 `INVALID_OUTPUT`；
- 消耗 output-correction retry；
- 阻止 Reviewer 或 Decision 读取结果；
- 改变对象状态；
- 影响 Decision 允许集合。

## 10. Renderer 解耦

当前最终 Renderer 假定每个结果都严格具有固定正文结构。削弱在线校验后，
Renderer 必须同步改成容错读取，否则工作流可能在 Decision 选择完成后才失败。

目标：

- Renderer 不重新判断专业正确性；
- 已提交结果即使包含附加字段也能渲染；
- 可识别字段按标准章节渲染；
- 未识别或非标准正文以保真的 JSON/文本附录保存；
- 缺少可选字段不能使状态机完成阶段崩溃；
- 最终 manifest 继续记录全部来源结果引用。

若某个结果的内容严重不足以形成最终报告，应由 Reviewer/Decision 在
`FINISH_WORKFLOW` 前识别，而不是由 Renderer 充当最后一个隐式 Gate。

## 11. 后续代码修改范围

### 11.1 `schemas.ts`

- 删除完整正文的阻断性 Provider Schema；
- 定义 Worker/Reviewer 核心控制字段提取器；
- 删除 `validateWorkMechanics` 和 `validateReviewMechanics` 的在线阻断作用；
- 可选保留完整模板 lint，但输出只能是 advisory。

### 11.2 `controller.ts`

- Worker/Reviewer Turn 不再传入完整正文 `outputSchema`；
- JSON parse 后只校验角色核心字段；
- 核心字段合法即保存结果并进入下一节点；
- 只有传输、JSON 和核心字段错误进入 output-correction retry；
- 删除 retry guidance 的 Ref 子串检查；
- 保持语义重试只由 Decision 的 `RETRY_*` 触发。

### 11.3 `protocol.ts`

- 保留唯一 Decision 和 allowed-set 检查；
- guidance 只作为不透明文本；
- 不基于 guidance 内容或长度改变分支；
- 对非控制性附加文本采用“提取核心字段并保存其余文本”的鲁棒策略，不能因
  与状态机无关的表达直接失败。

### 11.4 `workflow.ts`

- 继续只使用 `workOutcome`、`reviewVerdict`、TaskBinding 和对象索引；
- 不读取正文来选择下一 Task；
- 保留最小机械 completion guard；
- 检查 requirement 计算中没有隐含完整正文 Schema 假设。

### 11.5 `validation.ts`

- 将运行目录一致性校验与 Agent 内容校验分开；
- 保留引用、状态、revision、Turn、Round、事件和对象索引一致性；
- 不因正文未匹配完整模板而把运行目录判为损坏；
- 完整内容 lint 若保留，单列为非阻断 advisory。

### 11.6 `renderer.ts`

- 改为容错读取；
- 不能依靠在线强 Schema 保证字段存在；
- 不承担隐藏的完成 Gate。

### 11.7 `types.ts` 和发布 Schema

- 明确区分：
  - Agent 原始结果；
  - Script 提取的核心控制投影；
  - 推荐内容模板；
  - Controller 内部索引。
- 清理不再作为 wire gate 的强类型假设；
- 发布 Schema 若保留，应标记为 Agent Ref 模板，而不是 Controller 权威
  阻断协议。

## 12. 后续 Skill/Ref 修改范围

### Worker

- 保留完整 Anchor/Direction 方法和推荐 JSON 模板；
- 不再声称 Script 会替它检查全部跨字段规则；
- 明确内容会由 Reviewer 和 Decision 语义审查；
- 输出不完整或 follow 错误可能导致 Decision 语义重试。

### Reviewer

- 显式接管 Worker Result Ref、`workOutcome` 和正文一致性的审阅；
- 对非标准但可理解的 JSON 正文作语义判断，而不是要求 Script 先拒绝；
- 继续记录 query gaps；
- 不承担调度决策。

### Decision

- 显式复核 Worker/Reviewer 是否遵循各自 Ref；
- 识别结构可解析但语义错误、字段间矛盾、漏审和错误 verdict；
- 只有会破坏闭环或 workflow 的错误才选择 `RETRY_*`；
- 正常的 REVISE、REJECT、证据缺口和未完成结论继续通过正常分支处理；
- 决策仍只从 Script 注入的允许集合选择。

## 13. 测试修改

必须新增或改写以下测试：

1. Worker JSON 可解析且 `workOutcome` 合法，即使正文缺少推荐字段，也会进入
   Reviewer，而不是 Script output correction；
2. Reviewer JSON 可解析且 `reviewVerdict` 合法，即使 findings 结构非标准，
   也会进入 Decision；
3. Worker 语义错误由 Decision 选择 `RETRY_WORKER`；
4. Reviewer 漏审或 verdict 错误由 Decision 选择 `RETRY_REVIEWER`；
5. JSON 无法解析仍重做同一 Agent Task；
6. 缺少或使用未知核心控制字面量仍触发格式纠错；
7. Decision 不在 allowed set 时仍被拒绝；
8. guidance 不包含 Ref 名称时不被 Script 拒绝；
9. unknown content fields 不阻止流程；
10. `FINISH_WORKFLOW` 仍受最小机械闭环限制；
11. Renderer 能处理附加字段和缺少非核心字段；
12. Codex commentary 与 final answer 的 phase 分离测试继续保留；
13. 持久化验证确认完整原始结果和核心投影均可追踪；
14. 旧的“完整 Schema 缺字段必然 INVALID_OUTPUT”和
    “PARTIAL + PASS 必然由 Script 重试”测试应删除或改为 Decision 语义重试。

## 14. 兼容和迁移

这一修改改变了 Agent wire gate 和运行目录的有效性定义，不能静默用新逻辑
重新解释旧运行。

实施前需确定：

- 推荐将新契约提升到新的 format version；
- 新运行使用新 work directory；
- 旧 format version 继续只读检查或明确拒绝恢复；
- 不自动把旧 `INVALID_OUTPUT` 改写成新有效结果；
- 如需恢复旧失败运行，必须使用独立、显式、可审计的迁移命令。

## 15. 实施完成判据

只有同时满足以下条件，才算完成本次重构：

- Script 在线 Gate 只剩传输、JSON、核心控制字面量、状态和引用完整性；
- 完整专业正文不再由 Provider/Script 强 Schema 阻断；
- Reviewer 能看到所有核心字段合法的 Worker 结果；
- Decision 能看到 pending Worker/Reviewer 原始结论并决定语义重试；
- guidance 完全不透明地保存和转发；
- 状态机仍只有 Script 可以修改；
- 最小机械 completion guard 仍有效；
- Renderer 不再隐式依赖完整强 Schema；
- 原始输出、核心投影、状态、索引和事件仍完整可审计；
- 03、04、Skill、Ref、README、发布 Schema 和测试与本文保持一致。

最终职责边界为：

> Script 校验“能否安全调度”；Reviewer 和 Decision 判断“内容是否正确、
> 是否需要修复、是否真正完成”。

## 16. 实施记录

format version 4 完成了最小 Gate；format version 5 在不改变该 Gate 的前提下
进一步完成以下 06 增量：

- Worker/Reviewer Provider dispatch 不再携带完整正文 `outputSchema`；
- Script 在线只校验协议消息、JSON object 和角色核心控制字面量；
- 完整 Ref-template lint 只写 `advisories`；
- 每个成功 Turn 保存完整结果和独立 `control.json`；
- pending、pre-review 和 ObjectIndex 索引 `workOutcome` /
  `reviewVerdict`；
- Reviewer/Decision Skill 显式接管 Ref、跨字段和 workflow 语义判断；
- Decision guidance 作为不透明文本保存与转发；
- Renderer 容错读取，并为非标准结果写保真 JSON 附录；
- 01–04 设计、三类 Skill/Ref、README 和测试已同步；
- D01 只增加一个 `observationRef`；W01/R01/Decision 输出协议不扩张；
- partial 永不进入 JSON/core-control Gate，完整唯一消息才可重放；
- runtime timeout、trajectory、memory 和 recovery 均为机械记录，不成为专业
  内容 Gate；
- v4 及更旧运行目录明确不可由 v5 Controller 恢复。

本地验证包括协议/端到端、流式 timeout 和 recovery 测试，三个活动 Skill 的
结构校验，以及新建 v5 run 的 `init → validate → status` 检查。
