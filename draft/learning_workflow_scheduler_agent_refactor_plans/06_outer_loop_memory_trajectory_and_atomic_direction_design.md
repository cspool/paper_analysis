# 外循环观察、压缩记忆、进展轨迹、原子 Direction 与运行时恢复设计

> 状态：已于 2026-07-31 按 format version 5 实施；Agent wire 输出协议保持
> 不变。旧 format 目录继续只读保留，v5 Controller 明确拒绝原地重解释。  
> 前置设计：`01_agent_types_and_scheduling_responsibilities.md`、
> `02_loop_requirement_closure_design.md`、
> `03_script_agent_message_and_storage_contract_design.md`、
> `04_script_agent_json_communication_contract_inventory.md`、
> `05_minimal_controller_validation_and_semantic_handoff_design.md`。  
> 适用范围：`scripts/simple_semantic_loop/refactor/`、Decision/Worker/Reviewer
> Skill 与 Ref、Agent 可见观察输入、Turn 流式留痕、超时恢复、过程报告和测试。  
> 本文既保留设计依据，也记录 format version 5 的实际落地；第 19 节是实现索引
> 与验证结果，若早期“建议”措辞和第 19 节存在差异，以已实施契约为准。

## 1. 本轮确认采用的建议

采用 Orchestra autoresearch 的以下思想，并按当前“确定性 Script + fresh
one-turn Agent”架构改写：

1. Decision 在每轮局部结果判断之外执行一次全局外循环反思；
2. 为长期运行提供压缩的研究结论记忆，同时保留完整 W01/R01 为可回读证据；
3. Worker 和 Reviewer 按当前 Task 动态选择领域专家 Skill，Decision 保持简单
   的调度控制 Skill；
4. 明确语义完成标准，不把最小机械闭合误当成需求已经充分闭合；
5. 使用非权威、可重建的过程观察和阶段性人类报告降低长期运行的认知成本。

采用 Karpathy autoresearch 的以下两项思想：

1. 建立简洁、可比较的跨轮进展轨迹；
2. 将 Direction 收紧为一个最小可检验的主要变化。

其中第 1 项有一个不可违反的边界：

> 进展轨迹只能由 Script 从已有权威运行记录中机械生成，供 Decision Agent
> 和人类观察；它不是任何 Agent 需要输出、补全或回显的协议消息。

不采用 Karpathy autoresearch 的单指标自动保留/回退、Agent 直接修改权威
状态、无限运行直到人工中断，也不采用“由 Agent 自行修改 Controller/Skill”
的运行内自修改方式。

同时吸收实际长 Worker Turn 暴露出的运行时可靠性问题：Agent 已经执行工具并
开始流式输出，不代表 Provider Turn 已完成，更不代表 Controller 已取得可提交
的完整协议消息。后续设计必须把流式片段、完整 Agent 消息和 Provider Turn
完成三个状态分开记录；超时现场必须可审计、可重试，但不允许把半截输出当成
状态机输入。

## 2. 不改变的架构边界

本设计不改变顶层角色和正常 Loop：

```text
Worker → Reviewer → Decision
  ↑                    │
  └──── Script branch ─┘
```

继续保持：

- Controller Script 是唯一持久状态库和状态转换执行者；
- Worker、Reviewer、Decision 都是 fresh one-turn Agent；
- Worker 只创建或深化一个 Anchor/Direction；
- Reviewer 只独立审阅一个 Script 绑定的 Work Result；
- Decision 只从 Script 本次允许集合选择下一条分支；
- Agent 不直接写 Controller 权威状态；
- Script 在线只校验传输、JSON、核心控制字面量、状态和引用完整性；
- 专业语义继续由 Reviewer 和 Decision 判断；
- 新实验仍然只形成测量计划，不在当前 Learning Loop 中执行；
- `maxRounds` 仍然触发可恢复暂停，而不是语义完成；
- 最终完成仍需要 Decision 选择 `FINISH_WORKFLOW` 和 Script 的最小机械闭合。
- runtime partial output 只属于审计证据；只有完整、可提取的 Agent 消息才进入
  JSON 核心控制校验；
- timeout、retry 和显式 recovery 只能新建 Turn，不得改写旧 Turn 或续写半截
  JSON；
- Script 不以字符数、数组长度或专业内容完整度替代 Reviewer/Decision 的语义
  判断。

Agent 输出协议保持不变：

```text
Worker:   一个 JSON object，核心字段 workOutcome
Reviewer: 一个 JSON object，核心字段 reviewVerdict
Decision: decision = <allowed literal>
          guidance = <optional opaque text>
```

本文不增加 `progress`、`trajectory`、`selectedSkills`、`outerLoopState`、
`memoryUpdate` 等 Agent 输出字段。

## 3. 当前缺口

### 3.1 最小机械闭合不等于语义完成

当前 Script 在至少一个 Anchor 和其至少一个 Direction 均为
`READY_FOR_REVIEW + PASS` 后即可向 Decision 提供 `FINISH_WORKFLOW`。这只能
证明最小结构闭合，不能证明：

- Topic 已获得适当的场景和 6L 覆盖；
- 尚无会改变主要结论的关键缺口；
- 已通过结论构成连贯、可解释的最终答案；
- 继续增加 Anchor、Direction 或审阅角度的信息增益已经较低。

这些只能由理解 Goal 和所有结论的 Decision 判断。

### 3.2 Decision 的长期输入会随对象数量增长

当前 D01 提供全部 committed Work/Review Ref，并要求 Decision 读取所有结果。
随着 Anchor、Direction、revision 和拒绝历史增长，完整正文会反复进入新的
Decision Turn，成本和干扰持续增加。

完整结果不能删除或被摘要替代，但可以增加一个非权威、可回读的压缩观察层。

### 3.3 Decision 只看到允许字面量，分支后果不够显式

Script 实际能够确定：

- `RUN_WORKER` 当前会绑定什么 `CREATE_*` / `DEEPEN_*` action；
- `RUN_REVIEWER` 会审阅哪个当前对象；
- retry 会复用哪个冻结 TaskBinding；
- 每条分支之后的固定 Agent 序列。

这些都是确定性事实，但当前 Decision 主要通过 requirements 和 Skill 自行推断。
若 Decision 不知道一个允许字面量将产生的确切机械后果，其全局判断可能正确，
分支选择却无法实现预期。

### 3.4 EventLog 很完整，但不等于进展轨迹

当前 Event、Turn、Runtime 和 ObjectIndex 在正常封口路径上适合审计和恢复，
但不适合快速观察；超时路径的数据缺口另见 3.6：

- 每轮做了什么；
- 是否产生新的通过对象；
- requirement 是否发生变化；
- 格式/语义重试是否增多；
- token 和时间成本如何变化。

需要一个 Script 派生的紧凑观察视图，而不是要求 Agent 再次总结这些运行事实。

### 3.5 Direction 的“一个主要变化”需要进一步操作化

当前 `work-result-direction-v2` 已有 `single primary change`，Reviewer Rubric
也要求主要变化最小且与 enabler 区分。但仍需明确：

- 什么叫最小可检验，而不是最小代码改动；
- 联合配置何时可作为一个不可分 Direction；
- 多项独立变化何时必须拆分；
- 不能拆分归因时允许声明什么层级的结论；
- measurement plan 如何先验证 baseline，再证伪主要变化。

这些是 Worker/Reviewer 的语义规则，不是 Script Schema。

### 3.6 流式输出已可见，但超时现场没有完整持久化

实际运行出现过以下序列：Worker 已完成多次工具调用，随后在控制台连续输出
大段 Direction JSON，但 300 秒硬超时发生时该 JSON 仍停在字符串中间。该次
失败不是 JSON Schema 或核心字段错误，而是 Provider Turn 尚未完成：

- 控制台的 `agent:output` 只是第一个 output delta 到达；
- `turn/completed` 尚未到达，完整 final-answer item 也未被 Controller 提交；
- validation audit 仍为空，说明 JSON Gate 尚未执行；
- timeout 路径把内存中的 provider IDs、raw events、tool events、usage 和半截
  文本降级为空失败结果，现场无法从工作目录完整复盘；
- 连续 runtime retry 复用了同一冻结 Task，但没有把“上次因输出过长而超时”
  作为明确的 runtime retry 事实反馈给新 Turn；
- runtime retry budget 用尽后 lifecycle 进入 `FAILED`，现有 `resume` 不能在保留
  审计历史的前提下显式恢复该类运行。

因此 06 的观察层不能只覆盖已完成轮次，还必须覆盖 Turn 进行中的运输事实和
可恢复失败。解决方案仍遵守最小 Controller 原则：Script 只判断运输状态、
超时、完整消息边界和重试资格，不判断半截研究内容是否“已经足够好”。

## 4. 总体修改结构

增加一个完全由 Script 生成的观察层：

```text
权威层
  workflow_goal / tasks / bindings / turns / results /
  objects / rounds / events / runtime
        │                         ├─ runtime.jsonl / partial_output.txt
        │                         │  增量审计，不是状态机消息
        │ Script 机械投影，不产生专业判断
        ▼
观察层
  progress_trajectory.jsonl
  research_memory.json
  decision_observation.json
  checkpoint.md
        │
        ├─ Decision 读取后执行外循环判断
        └─ 人类查看过程进展
```

观察层具有以下性质：

- 非权威；
- 可从权威层重建；
- 不允许反向覆盖权威状态；
- 不参与 Worker/Reviewer 输出 Gate；
- 不参与 Script 的专业语义判断；
- 缺少某个非核心正文摘要时必须退化为 Ref 导航，不能失败；
- 所有压缩条目都保留原始 W01/R01/Decision Ref；
- Agent 可按需回读完整结果。

## 5. Script 生成的进展轨迹

### 5.1 目的

进展轨迹回答的是机械事实：

```text
这一轮执行了什么？
产生了什么核心 outcome/verdict/decision？
通过对象数量和 remaining requirements 如何变化？
付出了多少运行成本？
是否发生重试？
```

它不回答：

```text
研究方向是否专业正确？
某项证据是否可信？
Topic 是否已经语义饱和？
是否应当完成？
```

后一组问题仍由 Reviewer/Decision 处理。

### 5.2 存储位置

建议新增：

```text
<work-dir>/observations/progress_trajectory.jsonl
```

每个完成的 Decision 周期追加一条记录；若运行在到达 Decision 前发生终止性
runtime failure，可追加一条 terminal observation。追加必须幂等，中断恢复
不得复制同一 Round/Decision 记录。

terminal observation 只引用结构化 runtime failure、最后 Turn、capture state
及 partial/complete output Ref；不得摘录或总结 partial 的研究语义。

### 5.3 最小记录格式

```json
{
  "round": 3,
  "action": "CREATE_DIRECTION",
  "workRef": "results/work-....json",
  "workOutcome": "READY_FOR_REVIEW",
  "reviewRef": "results/review-....json",
  "reviewVerdict": "PASS",
  "decision": "RUN_WORKER",
  "accepted": {
    "anchors": 1,
    "directions": 1
  },
  "remainingRequirements": [],
  "retries": {
    "outputCorrection": 0,
    "semantic": 0,
    "runtime": 0
  },
  "usage": {
    "elapsedMs": 214000,
    "inputTokens": 42000,
    "outputTokens": 6800
  }
}
```

字段只来自 Script 已知事实：TaskBinding、核心控制投影、ObjectIndex、
requirements、Turn/Runtime records 和已提取 Decision。

允许实现时进一步删减可由其他字段无损推导的元素；不应增加 Script 无法机械
确定的 `qualityScore`、`noveltyScore`、`saturation`、`researchValue` 等字段。

### 5.4 Agent 可见性

进展轨迹主要提供给负责全局进度判断的 Decision：

- Decision 通过一个 Ref 读取；
- Prompt 不内联完整轨迹；
- 默认提供最近若干条记录和完整轨迹路径；
- Decision 可以在判断重复、停滞或成本异常时读取更早记录；
- Worker/Reviewer 不需要全局轨迹，继续只读取当前局部 T01 和具名输入；
- Agent 不得被要求输出或回显任何 trajectory 字段。

`status` 或新增的只读查看方式可以把同一轨迹渲染给人类，但不能建立第二套
状态来源。

## 6. 压缩研究记忆

### 6.1 定位

Orchestra 的 `findings.md` 思想不能直接照搬为“让持久 Agent 修改权威研究
状态”，因为当前架构明确取消持久 Agent。适配后的定义是：

> Script 从当前 Agent 结论中机械生成一个非权威的研究结论目录；Decision
> 在每个 fresh Turn 中利用该目录恢复全局认知，并按需回读原始结果。

建议新增：

```text
<work-dir>/observations/research_memory.json
```

### 6.2 内容分区

研究记忆至少包含：

```text
accepted
  当前 active 且通过的 Anchor/Direction

needs_revision
  当前 verdict 为 REVISE 的对象

rejected_lessons
  最新 REJECT 对象、Reviewer summary/findings 和原始 Ref

open_query_gaps
  Reviewer 已提出、尚可能影响结论的对象局部问题

coverage
  从可识别 Anchor scope6L 字段机械投影的动态覆盖；缺失时只保留 Ref

decision_trail
  历次正式 decision 和其可选 guidance 的 Ref/原文

requirements
  当前 Script 机械 remaining requirements
```

### 6.3 压缩规则

Script 只能：

- 复制 Agent 已经写出的简短 `name`、`summary`、scenario、baseline、
  performance tension、primary change、Reviewer finding/query gap；
- 复制核心 `workOutcome`、`reviewVerdict` 和正式 decision；
- 复制或引用 Decision guidance；
- 按 ObjectIndex 状态分组；
- 保留原始结果路径；
- 对已知 Ref 字段进行容错读取。

Script 不得：

- 自己生成专业总结；
- 判断哪个机制更重要；
- 合并互相冲突的专业结论；
- 把缺少摘要解释为内容无效；
- 根据自然语言标记“已饱和”或“应当完成”；
- 用研究记忆覆盖 W01/R01。
- 把 `partial_output.txt`、未完成 provider item 或 runtime-failed Turn 当作 Worker、
  Reviewer 或 Decision 的结论；

非标准但 core-valid 的 Agent 结果必须进入记忆目录。无法识别正文时使用：

```json
{
  "workRef": "results/work-....json",
  "reviewRef": "results/review-....json",
  "workOutcome": "READY_FOR_REVIEW",
  "reviewVerdict": "PASS",
  "summaryAvailable": false
}
```

Decision 看到 `summaryAvailable=false` 后自行回读原始结果；Script 不补写语义。

### 6.4 权威关系

发生冲突时：

```text
Workflow Goal / T01 / 原始 W01-R01 / Controller 索引
  > research_memory.json
  > checkpoint.md
```

研究记忆是导航和压缩层，不是新的 canonical state。

## 7. 冻结的 Decision 观察快照

### 7.1 单一入口

为避免继续向 D01 平铺大量元素，建议 D01 只新增一个 Ref：

```json
{
  "goalRef": "workflow_goal.json",
  "committedResults": [],
  "pendingResults": {},
  "remainingRequirementsAfterPendingCommit": [],
  "observationRef": "contexts/<decision-id>/decision_observation.json"
}
```

`decision_observation.json` 是本次 Decision Turn 的不可变输入快照，包含：

- `researchMemoryRef`；
- 完整 trajectory 路径和最近若干条 `trajectoryTail`；
- 本次允许 Decision 的机械分支效果；
- 当前结果计数、requirements 变化和重试计数等纯机械观察。
- 最近 runtime failure 的结构化类型和 Ref（若与当前周期相关），但不包含或总结
  partial 正文。

D01 继续保留原始 committed/pending Ref，保证 Decision 可以直接核查完整结论。
`observationRef` 只减少重复读取成本，不替代原始输入。

### 7.2 分支效果预览

Script 在调用 Decision 前生成本次允许分支的确定性预览，例如：

```json
{
  "branchEffects": [
    {
      "decision": "RUN_WORKER",
      "nextAction": "CREATE_ANCHOR",
      "targetRef": null,
      "sequence": ["WORKER", "REVIEWER", "DECISION"]
    },
    {
      "decision": "RUN_REVIEWER",
      "nextAction": "REVIEW_DIRECTION",
      "targetRef": "results/work-direction-....json",
      "sequence": ["REVIEWER", "WORKER", "REVIEWER", "DECISION"]
    },
    {
      "decision": "FINISH_WORKFLOW",
      "nextAction": "FINALIZE",
      "targetRef": null,
      "sequence": []
    }
  ]
}
```

它只描述 `workflow.ts` 已经能够确定的后果，不增加新的 Decision 字面量，也不
解析 guidance。预览函数和实际绑定函数必须共享同一选择逻辑，测试证明预览与
随后真实 TaskBinding 一致。

## 8. Decision 的外循环方法

### 8.1 每个 Decision Turn 的顺序

Decision Skill 后续应按以下顺序工作：

1. 检查 pending Worker/Reviewer 是否存在会破坏 workflow 的语义错误；
2. 必要时从允许集合选择 `RETRY_WORKER` 或 `RETRY_REVIEWER`；
3. 读取研究记忆、进展轨迹和分支效果；
4. 比较本轮与此前结论，判断新增了什么、排除了什么、哪些关键问题仍开放；
5. 结合 Goal、当前动态 Anchor 6L 空间和 accepted/rejected 结论判断下一轮的
   信息价值；
6. 从 Script 允许集合选择下一角色分支或完成；
7. guidance 只给下一 Turn 一个精简关注点；Script 仍不解释 guidance。

“深化、扩展、结束”等可以作为 Decision 的内部推理概念，但不是新增协议
字面量。正式输出仍只有：

```text
RUN_WORKER
RUN_REVIEWER
RETRY_WORKER
RETRY_REVIEWER
FINISH_WORKFLOW
```

### 8.2 进展轨迹的使用方式

轨迹只提供信号，不提供结论。例如：

- 多轮 accepted counts 未变化；
- remaining requirements 重复不变；
- 相同 action 连续出现；
- semantic retry 增加；
- 每个通过对象的 token/时间成本持续升高。

Script 不把这些机械现象命名为“停滞”或“饱和”。Decision 结合具体 W01/R01
判断它们代表：

- 合理的同一对象深化；
- 查询方法无效；
- Reviewer 要求反复不一致；
- Topic 子空间仍值得继续；
- 继续运行的边际信息价值已经很低。

### 8.3 强化语义完成标准

Decision 只有同时认为以下条件成立，才应选择 `FINISH_WORKFLOW`：

1. Script 已允许该字面量，即最小机械 requirement 为空；
2. accepted Anchor 集合对 Goal 所要求的 Topic 范围形成适当覆盖，而不只是刚好
   一个最小样例；
3. 每个进入最终结果的 Direction 都具有明确 baseline change、机制、条件化
   影响、guardrails、失败条件和可证伪测量计划；
4. 不存在会实质改变主要结论或最终建议的 blocking query gap；
5. REJECT 和主要负面结论已被考虑，后续工作不会明显重复已排除路线；
6. accepted 结论能够组织成一份连贯的人类可读答案；
7. 再增加一轮 Anchor、Direction 或独立审阅的预期信息增益已经较低；
8. 结束不是由 maxRounds、token 成本、长上下文或单次 Agent 疲劳替代决定。

这些是 Decision Skill 的语义约束。Script 不把它们实现成固定字段检查。

`FINISH_WORKFLOW` 的 guidance 宜简要说明为何语义闭合，但它仍是 Skill 要求，
不是 Script 的阻断性协议字段。

## 9. 领域专家 Skill 路由

### 9.1 角色边界

Orchestra Skill 只供内容型 Agent 增强专业方法：

```text
Worker   → 可选择 0–2 个与当前 Task 最接近的专家 Skill
Reviewer → 保持现有 0–2 个专家 Skill 策略，并独立选择
Decision → 不加载领域专家 Skill，保持简单的调度控制方法
Script   → 不选择、不解释、不校验专家 Skill
```

Topic 不能预先假设。Worker/Reviewer 必须先读取 Goal、T01 和当前对象，再按
实际技术对象选择 Skill；没有紧密匹配时使用 0 个。

### 9.2 Worker 的新增方法约束

Worker 使用专家 Skill 时：

- 专家 Skill 只增强技术对象、实现边界、机制、控制变量、失败条件和测量方法；
- 专家 Skill 不能改变 Goal、T01、对象绑定或 Work Result Ref；
- 专家 Skill 的一般知识和示例不能作为当前 Direction/Anchor 的来源证据；
- 事实证据仍必须来自 Worker 实际深读的知识库来源；
- 最多两个 Skill，第二个必须补充不同的技术或评价边界；
- 最终仍只返回一个 W01；
- 不增加 `selectedSkills` 输出字段。

### 9.3 Reviewer 的独立性

Reviewer 不需要沿用 Worker 选择的专家 Skill。它应根据实际审阅缺口独立选择，
避免同一方法偏见被直接复制到审阅环节。

Reviewer 主 Skill 中现有的 0–2 策略和安装白名单继续有效。实施时可以复用一
份共享 routing Ref 避免 Worker/Reviewer 名单漂移，但不能把 Skill 选择变成
Script 调度或 Agent 输出协议。

## 10. 最小可检验 Direction

### 10.1 定义

“最小可检验”不是“只能改一个参数”或“只能修改一行代码”，而是：

> 一个 Direction 只提出一个可以相对 bound Anchor baseline 独立陈述、被控制
> 比较证伪的主要因果变化；为了让该变化可执行所必需的 enablers 必须与主要
> 变化明确区分。

例如，一个跨设备阶段解耦方案可能同时需要资源池划分、token transfer 和
独立 batching。若三者构成不可分的可执行包，可以作为一个 Direction，但只能
声明“完整配置包相对 baseline”的联合效果，不能在没有消融证据时分别声称某个
组件贡献了多少收益。

### 10.2 Worker 规则

Worker 后续必须：

- 在现有 `baselineChange` 中明确一个 primary change；
- 明确列出只是使 primary change 可执行的必要 enablers；
- 不把两个可以独立部署、独立测量的优化拼成一个 Direction；
- 若采用不可分联合包，限制结论为 package-level effect；
- 在 `mechanism` 中给出从主要变化到目标指标和 guardrails 的因果链；
- 在 `failureConditions` 中说明联合包或主要变化在哪些条件下收益消失；
- 在 `measurementPlan` 中先复现冻结 baseline，再测试完整主要变化，并在需要
  归因时预先定义最小消融；
- 固定或分层报告所有会影响公平比较的配置、工作负载、环境、质量和服务变量；
- 继续遵守“不执行新实验”，只形成可执行、可证伪的测量计划。

优先通过现有字段表达这些语义，不为了形式整齐立即增加新的必填 JSON 元素。

同时增加“最小充分表达”约束。一个 W01 应足以让 Reviewer 判断 Direction 是否
可证伪，但不应在一次 Turn 中展开未来实验阶段才会生成的全部枚举产物。例如：

- 可以给出候选生成器、冻结顺序、随机种子、比较口径和失败规则；
- 不必在 W01 中列出 64 个候选的完整 manifest、每个请求 ID 或每次 bootstrap
  重采样明细；
- 大型 manifest、trace 和逐配置表应写成未来实验 handoff 的预期 artifact，或
  引用已经存在的证据 Ref，而不是塞入一个超长 JSON 字符串；
- measurement plan 应保留复现所需的关键参数和歧义消除规则，但避免重复描述
  同一冻结条件；
- 如果一个计划需要数页附加细节才能说明，Worker 应先判断 Direction 是否混入
  了多个独立主张，或把实验执行手册误当成当前 Direction 结论。

这是一条 Worker Skill 和 Reviewer Rubric 的语义约束，不是 Script 的字符数、
数组长度或 token 数内容 Gate。Script 仍只执行 Provider 配置的运输预算和核心
控制字段校验；输出虽长但完整时，不得仅因“看起来复杂”而拒绝。

### 10.3 Reviewer 规则

Reviewer 后续必须检查：

- primary change 是否唯一且可证伪；
- enabler 是否被错误写成另一项独立优化；
- 多项变化能否分别部署和测量；若可以，是否应拆成多个 Direction；
- 不可分联合包是否只声明联合效果；
- 机制和 expected effects 是否超出了当前证据与可归因范围；
- measurement plan 是否包含 baseline reproduction、受控比较、guardrails 和
  必要消融；
- “联合效应无法拆分”是否被诚实保留为限制。
- W01 是否达到最小充分表达，还是把可外置的候选 manifest、重复冻结条件和
  实验执行明细无限展开，从而掩盖唯一 primary change；
- 精简建议是否不会删除复现、反证、公平比较和 guardrail 所必需的信息。

可拆分却混合多个独立变化，并因此无法判断 Direction 是否成立时，应产生
`BLOCKING` finding 和通常为 `REVISE` 的 verdict。该判断不进入 Script Gate。

### 10.4 Decision 规则

Decision 只在以下情况考虑语义 retry：

- Worker 明确违反绑定 Task/Direction Ref，却把多项无关变化包装成完成结果，
  且该错误会使 workflow 走向错误分支；
- Reviewer 明显漏审这一问题或给出与其 findings 不一致的 verdict。

普通的可修复组合过宽应按 Reviewer `REVISE` 进入正常深化路径，而不是把所有
专业不足都提升为通信错误。

## 11. Turn 运输边界、增量留痕与可恢复超时

### 11.1 明确三个不同状态

Runtime 和 Controller 必须区分：

```text
STREAMING_PARTIAL
  已收到一个或多个 output delta；文本可能停在任意 token，不能解析或提交

MESSAGE_COMPLETE
  Provider 已明确完成唯一 final-answer item；取得了一条完整候选协议消息

TURN_TERMINAL
  Provider 已发出 turn/completed 或明确失败终态；Turn 的 usage、tool events
  和最终 status 已封口
```

三者不能互相代替。控制台出现 JSON 不表示 `MESSAGE_COMPLETE`，完整
final-answer item 也不应伪装成正常 `TURN_TERMINAL`。只有完整候选消息才可能
进入现有 JSON 提取和核心控制校验；partial 永远不得进入 W01/R01/Decision
commit。

若超时时已经存在唯一 `MESSAGE_COMPLETE`，它与普通 partial 不同：Controller
可在中断完成、事件快照封口且确认没有第二个竞争 final-answer item 后，把它标记
为 `outputCapture=COMPLETE`，再通过与正常 Turn 完全相同的提取和核心校验做一次
确定性本地 replay。原 Provider Turn 仍记录为 timeout；replay 成功也不能改写旧
Turn 的运输历史。若上述条件不成立，只能新启 Agent Turn。

### 11.2 Turn 进行中就写盘

不能等 `runtime.run()` resolve 后才统一持久化。建议每个 Turn 在 provider 事件
到达时立即追加或更新：

```text
turns/<turn-id>/turn.json
  providerThreadId / providerTurnId 在 turn_started 时立即写入
  outputCapture = NONE | PARTIAL | COMPLETE
  partialOutputRef（可空）
  rawOutputRef 仍只指向完整候选消息

turns/<turn-id>/runtime.jsonl
  按到达顺序增量追加 provider raw event、tool event、usage/progress 和本地
  timeout/interrupt 事件

turns/<turn-id>/partial_output.txt
  按 delta 原样增量保存；仅用于审计或 runtime retry 参考

turns/<turn-id>/output.txt
  只保存唯一、完整、已选中的候选协议消息

turns/<turn-id>/runtime_error.json
  保存机械 failure kind、timeout policy、last activity、interrupt 结果和上述 Ref
```

`partial_output.txt` 与 `output.txt` 不得共用“最后一次写入覆盖”的语义。
`partial_output.txt` 即使恰好能被 `JSON.parse`，也仍是 partial；消息完整性来自
Provider item 边界，不来自 Script 猜测括号是否闭合。

`turn.json` 只允许在 `RUNNING` 期间更新上述运输投影；一旦封口为 completed、
runtime-failed 或其他终态，旧 Turn 即不可变，后续 retry/recovery 只能引用它。

建议把主运行失败字面量限制为少量机械类别：

```text
IDLE_TIMEOUT | HARD_TIMEOUT | PROVIDER_ERROR
```

`interruptError` 只作为可选附加事实，不能覆盖原 timeout。错误原文另存，不继续
扩张控制字面量。专业内容错误不属于这些类别。

### 11.3 双层超时和确定性终止顺序

单一 300 秒 wall-clock timeout 会把“持续有工具/输出活动的长 Turn”和“真正卡死”
混为一类。Fresh run 应冻结并记录：

```text
idleTimeoutMs
hardTimeoutMs
interruptGraceMs
```

- output delta、item completion、tool start/complete、usage/progress 或 compaction
  等有意义 provider event 重置 idle timer；普通 stderr 噪声不重置；
- hard timer 从 Turn 启动起计算，任何活动都不能重置；
- Decision、Worker、Reviewer 可以有不同 profile，但每个 Turn 实际使用的值必须
  写入运行记录；
- 初始默认值应通过真实长 Turn 校准。此次 300 秒 Turn 在约 246 秒后才开始输出，
  因此不能继续把 300 秒同时当作合理 idle window 和绝对 hard cap；
- timeout 不是“立即删除 pending”。顺序必须是：冻结当前 snapshot → 写盘 →
  请求 interrupt → 等待短 grace window → 接收可能到达的终态/完整 item →
  再封口并 resolve 为结构化 runtime failure；
- 即使 interrupt 请求失败，也要保留 timeout 前的全部现场，并以原 timeout 为
  主失败、interrupt error 为附加事实。

具体毫秒数是运行预算，不是工作流语义。建议新运行先采用明显大于当前 300 秒
的 hard cap，并保留约 300 秒量级的 idle window；最终默认值在实现测试和真实
样本测量后冻结，而不是写进 Agent Skill。

### 11.4 Runtime retry 是同一绑定的全新 Turn

对只有 partial 的 timeout，Script 自动重试时继续复用冻结的 TaskBinding 或
DecisionContext，不让 Script 根据半截内容改变 action。新 Prompt 追加一个简短
机械块，例如：

```text
[RUNTIME_RETRY]
previousTurnRef: turns/<old-turn>/turn.json
failure: HARD_TIMEOUT
partialOutputRef: turns/<old-turn>/partial_output.txt
instruction: 重新执行同一冻结任务；返回一个完整、精简的协议 JSON。上次文件
             是未完成草稿，不得从断点续写，也不得把它当作已校验结论。
```

追加块不内联巨大 partial，不诊断其专业内容，也不修改原 T01/D01。Agent 可以
把 partial 当作不可信工作草稿，避免再次无界展开；所有最终事实仍须在新的完整
结果中自洽表达。若失败发生在完整消息的格式/核心字段校验之后，则继续使用既有
output-correction 路径，不混入 runtime retry。

Runtime retry budget 计算 attempt，不计算 output-correction 或 semantic retry。
每次 retry 都保留独立 Turn、provider IDs、prompt、partial/complete output 和
runtime log。

### 11.5 Runtime budget 用尽后的显式恢复

`resume` 继续只恢复 `PAUSED`，不应含糊地复活任意 `FAILED`。用户必须通过显式
恢复命令和一次性、幂等 recovery token 授权新的尝试：

```text
node scripts/simple_semantic_loop.ts recover-runtime \
  --work-dir <dir> \
  --recovery-token <token> \
  [--idle-timeout-ms N] [--hard-timeout-ms N]
```

Script 只在以下机械条件全部成立时允许恢复：

- lifecycle 为 `FAILED`，且结构化 `failureKind` 为
  `RUNTIME_RETRY_EXHAUSTED`；
- 最新失败 Turn 仍绑定当前 TaskBinding/DecisionContext；
- 失败后没有结果 commit、对象 revision 或冲突的 active Turn；
- store audit 和 Ref 完整性校验通过；
- 用户显式授权一个新的 recovery attempt。

恢复命令新增一条 immutable recovery record/event，必要时记录本次 timeout
override，然后创建全新 Turn。它不增加或修改旧 Turn，不把 partial 提升为结果，
也不通过手工编辑 `state.json` 清零计数。重复执行同一 recovery token 必须幂等；
其他 FAILED 原因仍拒绝恢复。

### 11.6 控制台标签不能暗示错误完成状态

实时转发继续保留，但标签改为明确的运输事件：

```text
agent:stream-start       第一个 delta 到达
agent:message-complete   唯一 final-answer item 完成
turn:complete            Provider Turn 已进入终态
turn:timeout             idle/hard timeout，并显示 capture=NONE|PARTIAL|COMPLETE
```

`agent:stream-start` 后的正文仍可原样实时打印。这样人类可以看到进展，又不会把
“控制台已经打印一个左花括号”误认为 Agent 已成功输出 JSON。timeout 行应给出
`runtimeRef` 和 `partialOutputRef`，便于直接定位现场。

### 11.7 与最小 Controller 校验原则的关系

本节只增强运输事实的可观测性和恢复性：

- Script 可以判断消息是否完整、是否超时、是否存在唯一候选消息以及能否重试；
- Script 不判断 Direction 是否过度设计、证据是否充分或 measurement plan 是否
  专业正确；
- 超长输出的根因由 Worker 的最小充分表达和 Reviewer/Decision 语义审阅处理；
- Provider token/output budget 仍是资源上限，但不被包装成专业内容 Gate；
- partial 可以供新 Turn 参考，却永远不是“最近 Agent 结论”。

因此该修改不会重新引入 05 已排除的强机械语义校验。

## 12. 阶段性人类报告

使用与 Agent 相同的观察层生成确定性 checkpoint，不增加报告 Agent：

```text
<work-dir>/observations/checkpoints/round-<n>.md
```

建议在以下时机生成：

- `maxRounds` 暂停前；
- workflow finish 前；
- 用户显式执行只读 checkpoint/status 命令时。

checkpoint 只模板化展示：

- Topic 和 objective；
- accepted/revise/reject 对象目录及原始 Ref；
- 动态 6L 覆盖投影；
- open query gaps；
- 最近 trajectory；
- token、时间和重试统计；
- runtime timeout、capture state 和 partial/complete output Ref；
- 当前 mechanical requirements；
- 最近 Decision guidance。

缺少非核心正文不得导致 checkpoint 失败；使用 Ref 和原始 JSON 附录退化。
checkpoint 不替代最终 `report.md`，也不宣布语义完成。

## 13. 消息和存储契约变化

### 13.1 Agent 输出

无变化：

```text
W01 / R01 / Decision line protocol
```

Runtime 增加的 `outputCapture`、timeout kind 和 partial Ref 都是 Script 内部运输
记录，不是 Agent 需要输出的字段。Agent 协议消息只有在完整消息边界成立后才被
提取；Script 不要求 Agent 回显 provider 状态。

### 13.2 Agent 输入

D01 仅增加：

```text
observationRef
```

Worker/Reviewer 的 T01 不增加全局 trajectory 或 memory 正文，继续保持局部、
单一任务输入。Decision guidance 仍可按现有方式原样转发。

只有同绑定 runtime retry 会在普通调用 Prompt 末尾追加 11.4 的
`[RUNTIME_RETRY]` 块。它属于 Script 已知运输事实，不修改 T01/D01，不内联
partial，也不要求 Agent 输出新的 retry 字段。

### 13.3 Script 内部和派生存储

新增：

```text
observations/progress_trajectory.jsonl
observations/research_memory.json
contexts/<decision-id>/decision_observation.json
observations/checkpoints/round-<n>.md

turns/<turn-id>/runtime.jsonl
turns/<turn-id>/partial_output.txt
turns/<turn-id>/output.txt
turns/<turn-id>/runtime_error.json
recoveries/<recovery-id>.json
```

观察层前三者提供给 Decision 或审计；checkpoint 面向人类。Turn runtime 和
recovery 记录属于权威运行历史，不能由观察层反向覆盖。所有文件必须记录可
追溯的来源 Ref；派生 observation 可重建，原始 runtime/partial/recovery 历史
不可通过重建伪造。

## 14. 对现有文件的修改设计

### 14.1 `types.ts`

- 为 D01 增加单一 `observationRef`；
- 定义 Script 内部的 trajectory、research-memory 和 decision-observation
  类型；
- 用 role-aware `idleTimeoutMs`、`hardTimeoutMs`、`interruptGraceMs` 取代单一
  `turnTimeoutMs`，并在 Turn 记录实际采用的 profile；
- 为 Turn 增加最小运输投影：`outputCapture`、`partialOutputRef`、结构化
  `runtimeFailureKind`，且保持 `rawOutputRef` 只代表完整消息；
- 让 timeout result 保留 provider IDs、elapsed、usage、tool events、raw event
  refs 和 capture state，不再构造全空失败结果；
- 为 State/Recovery 增加可机械判断的 `RUNTIME_RETRY_EXHAUSTED`，禁止通过解析
  自然语言 `reason` 决定能否恢复；
- 不把这些类型加入 Agent Result 联合类型；
- 不增加 Agent 输出核心字面量。

### 14.2 `workflow.ts`

- 抽取无副作用的 Worker action preview；
- 抽取无副作用的 pre-review target preview；
- 让 preview 和真实 TaskBinding 共用同一选择逻辑；
- 机械计算 accepted counts、requirements delta 和 branch effects；
- 不根据轨迹自动选择 Decision 或完成。

### 14.3 `controller.ts`

- 在调用 Decision 前生成冻结 observation snapshot；
- 在 Decision 周期提交或语义 retry 后幂等追加 trajectory；
- 在 commit/pre-review 后重建 research memory；
- pause/finish 前生成 checkpoint；
- 在 provider start 时立即写 provider IDs，并通过 Runtime event sink 增量持久化
  raw/tool/usage/delta；
- timeout 后根据 `NONE|PARTIAL|COMPLETE` 走固定分支：partial 只重试，满足
  11.1 条件的 complete candidate 才可本地 replay 核心校验；
- runtime retry 复用冻结绑定并追加机械 retry block；
- retry budget 用尽时写结构化 recoverable failure；显式 recovery 只创建新 Turn；
- 不要求 Agent 回显 observation；
- 不因可选摘要提取失败而拒绝 W01/R01。

### 14.4 `store.ts`

- 增加 observation 路径与幂等 append/rebuild 支持；
- 保存 observation 的来源 revision/Ref；
- 支持中断恢复时检查同一 Decision 是否已经写入 trajectory；
- 提供 per-Turn append-only runtime event 和 partial delta 写入；
- 在 timeout/进程崩溃后仍能从已落盘事件重建 capture snapshot；
- 保存 recovery record、幂等 token 和 timeout override，禁止覆盖旧 Turn；
- 保证派生文件删除后可以重建。

### 14.5 `prompts.ts`

- Decision Prompt 仍只给 D01 路径和 allowed literals；
- 不内联完整 trajectory/memory；
- 不增加新的 Decision 输出字段；
- 可简短强调先读 `observationRef`，再按需回读原始结果。
- 对 runtime retry 追加固定的短块，明确旧 partial 是未校验草稿、必须从头返回
  一个完整且精简的协议结果；
- 不把 partial 全文、Script 猜测的专业错误或新的 action 注入 retry Prompt。

### 14.6 `validation.ts`

- 校验 D01 的 observation Ref 存在且绑定本次冻结状态；
- 校验 trajectory 的来源 Decision/round 不重复；
- 校验 observation 中的原始结果 Ref 可解析；
- 不校验摘要的专业正确性；
- research-memory 的可选字段缺失只产生 advisory 或触发可重建，不把 Agent
  内容判为无效。
- 明确拒绝从 `partialOutputRef` 调用协议解析；只有 `rawOutputRef` 或满足 11.1
  的 captured-complete replay Ref 可进入现有核心校验；
- 校验 recovery eligibility、旧 Turn 不变性和同一 recovery token 幂等；
- 不增加 Direction 字符数、measurementPlan 数量或专业细节强度检查。

### 14.7 `runtime.ts` 与 `live_console.ts`

- Runtime 在事件到达时增量发出可持久化 snapshot/event，不再只在
  `turn/completed` 后返回聚合结果；
- timeout 前先封存 pending，再 interrupt 和等待 grace，禁止先删除 pending；
- 分别跟踪 delta、item-completed final answer 和 provider terminal；
- 支持 idle/hard 两个 timer，记录 last meaningful activity；
- 让控制台区分 stream start、message complete、turn complete 和 timeout
  capture state；
- 保持正文 delta 的实时转发。

### 14.8 `run_setup.ts` 与 `simple_semantic_loop.ts`

- 把单一 `turnTimeoutMs` 迁移为冻结的 role-aware idle/hard timeout profile 和
  interrupt grace；
- 新增显式 `recover-runtime` 命令及参数校验；
- recovery override 写入单独 record，不静默修改旧 `run.json`；
- `resume` 继续只处理 PAUSED。

### 14.9 `renderer.ts`、`status` 和只读命令

- 复用容错字段提取生成 research memory/checkpoint；
- status 可显示最近 trajectory 和 accepted counts；
- status/checkpoint 显示最近 timeout kind、capture state、provider IDs、partial Ref
  和 recovery eligibility，不把 partial 渲染成 Agent conclusion；
- 最终 report 继续从权威 final object refs 生成，不从 research memory 生成；
- 不让 checkpoint 成为隐藏完成 Gate。

### 14.10 Decision Skill

- 增加 observation 的读取顺序；
- 增加每轮全局外循环反思；
- 增加强化后的语义完成标准；
- 使用 branch effects 理解正式字面量的真实后果；
- 保持输出仍为 decision + optional guidance；
- 不加载领域专家 Skill。

### 14.11 Worker Skill 与 Direction Ref

- 增加 0–2 个领域专家 Skill 的动态选择规则；
- 强化最小可检验 primary change、enabler、不可分联合包和归因边界；
- 增加最小充分表达：保留复现与证伪关键规则，把未来实验生成的巨型 manifest
  和逐样本明细外置为 handoff artifact；
- 优先复用现有 Direction JSON 字段；
- 不新增 trajectory 或 selectedSkills 输出。

### 14.12 Reviewer Skill 与 Rubric

- 保持现有 0–2 专家 Skill 策略；
- 增加独立选择要求，避免直接复制 Worker 的方法偏见；
- 强化原子主要变化、联合包、归因边界和 baseline reproduction 检查；
- 检查过度展开是否掩盖唯一主张，并在不损失复现/证伪信息的前提下要求精简；
- 继续由 Reviewer/Decision 负责语义，不把规则迁入 Script。

## 15. 测试设计

后续实施至少覆盖：

1. Worker/Reviewer/Decision 输出协议未增加任何 trajectory/memory 字段；
2. trajectory 完全由 Script 从现有 Turn/Object/Runtime 记录生成；
3. 中断恢复不会为同一 Decision 重复追加 trajectory；
4. trajectory 的 workOutcome/reviewVerdict/decision 与 control projection 一致；
5. D01 只增加一个 observation Ref，不内联长期增长正文；
6. branch preview 与随后真实 Worker TaskBinding 完全一致；
7. branch preview 与随后真实 pre-review target 完全一致；
8. research memory 包含全部当前 accepted/revise/reject 条目及原始 Ref；
9. 非标准但 core-valid 的 W01/R01 可退化为仅 Ref 记忆，不导致流程失败；
10. 删除派生 observation 后可以从权威状态重建；
11. checkpoint 在可选正文缺失时仍可生成；
12. Script 不因多项 Direction 变化而阻断或格式重试；
13. Reviewer mock 可对可拆分的多变化 Direction 给出 `REVISE`；
14. 不可分联合包且只声明 package-level effect 时可以通过相应语义审阅；
15. Decision mock 能利用 trajectory/memory/branch effects 选择现有正式字面量；
16. `FINISH_WORKFLOW` 的 Script 机械条件保持不变；
17. maxRounds 仍为暂停，不被 trajectory 解释为语义完成；
18. Worker/Reviewer 使用专家 Skill 时仍只返回唯一 W01/R01；
19. output delta 在 Turn 完成前已增量写入 runtime log 和 partial output；
20. 只有 partial 的 timeout 不生成 `rawOutputRef`、不进入 JSON Gate，也不进入
    research memory/最近 Agent 结论；
21. timeout 保留 provider thread/turn IDs、tool events、已知 usage、elapsed、
    raw event refs 和结构化 failure kind；
22. timeout 前存在唯一 completed final-answer item 时，interrupt/grace 后可按
    11.1 条件本地 replay，且仍保留原 Turn timeout 历史；
23. 多个竞争 final answer、只有 delta 或 item 未完成时不得本地 replay；
24. 有意义 activity 会重置 idle timer，但不能延长 hard cap；stderr 不重置 idle；
25. timeout 实现先 snapshot/写盘、后 interrupt、最后删除 pending；interrupt
    失败不丢失原现场；
26. 控制台分别输出 stream-start、message-complete、turn-complete 和 timeout，
    同时继续实时转发正文；
27. runtime retry 保持同一 TaskBinding/DecisionContext，仅追加短 retry block，
    不续写 partial、不改变 action；
28. runtime retry budget 与 output/semantic retry 分开计数，用尽后产生结构化
    `RUNTIME_RETRY_EXHAUSTED`；
29. `recover-runtime` 只接受合格的 recoverable FAILED，一次授权只创建一个新
    Turn，重复 recovery token 幂等，旧 Turn 和旧 partial 不变；
30. Controller 进程在流式输出期间崩溃后，status/audit 可从落盘 runtime 事件
    重建 capture state；
31. Worker mock 可用更短的最小充分 measurement plan 表达同一 Direction，
    Reviewer 能审阅过度展开，而 Script 不增加字符数或数组长度 Gate。

## 16. 与 todo 19–35 的闭合关系

| todo | 本设计的增量闭合 |
|---|---|
| 19–21 | Worker/Reviewer 按实际 Task 动态路由专家 Skill；Reviewer query gaps 仍定义查询缺口，Script 不生成关键词 |
| 23 | Decision 通过外循环观察 accepted/rejected/gaps，判断 Topic 优化潜力是否还需扩展 |
| 24 | Script 生成阶段性 checkpoint；最终报告仍由最终 PASS 结果确定性渲染 |
| 25 | observation 是 Script→Agent 输入 Ref，不是 Agent 输出；trajectory 和 memory 属于派生存储；runtime retry 只追加 Script 已知的失败 Ref/字面量，不修改原任务消息 |
| 28–29 | research memory 展示 active Anchor 集合的动态 6L 覆盖，但不替代原始 Anchor |
| 30 | Direction 被收紧为一个最小可检验主要变化或诚实声明的不可分联合包 |
| 31 | 每个 Decision 既关闭局部 Worker/Reviewer 周期，也执行一次全局外循环反思 |
| 33–35 | trajectory、memory、observation snapshot 和 checkpoint 外部化；provider IDs、raw/tool/usage events 与 partial output 在 Turn 中增量持久化；超时通过同绑定新 Turn 或显式 recovery 恢复，旧记录不可变 |

## 17. 实施顺序

建议按以下顺序实施，先消除已经复现的数据丢失，再增加外循环能力：

1. 先修改 runtime types/store：增量保存 provider IDs、raw/tool/usage events 和
   partial output，并为旧 300 秒失败编写回归 fixture；
2. 实现 idle/hard timeout、snapshot-before-interrupt、grace finalization 和新的
   console 标签；
3. 修改 Controller/Prompt 的同绑定 runtime retry，实现结构化失败和显式
   `recover-runtime`，补齐崩溃/幂等/旧 Turn 不变性测试；
4. 强化 Worker 的最小充分表达和 Reviewer 对过度展开的语义审阅，避免重新产生
   巨型 W01，但不增加 Script 内容 Gate；
5. 实现纯 Script trajectory，并接入 status；
6. 实现 branch-effect preview 及 preview/actual 一致性测试；
7. 实现容错 research memory 和冻结 decision observation；
8. 修改 D01 与 Decision Skill，加入外循环和完成标准；
9. 修改 Worker 专家 Skill 路由；
10. 生成 checkpoint，并更新 01–05、README 和使用说明中的权威关系。

Agent wire protocol 本身不因本设计增加字段。但 timeout profile、Turn capture
字段、结构化 failure 和 recovery record 改变了 Controller 持久格式，因此完整
实现应提升 work-dir `formatVersion`。已有 v4 运行必须保持只读可审计：不得用
新二进制静默重解释或原地覆盖；若需要恢复旧失败运行，应提供显式、可审计的
copy migration，把旧目录完整保留并在新目录写入 migration/recovery record。
Skill/Ref hash、D01 契约也必须随新运行冻结。

## 18. 完成判据

只有同时满足以下条件，06 才算实施完成：

- Agent 输出协议没有因 trajectory/memory 增加字段；
- 进展轨迹完全由 Script 生成并可由 Decision/人类观察；
- Script 不从轨迹推导专业质量、饱和或完成；
- Decision 能看到冻结的 branch effects，并清楚每个允许字面量的机械后果；
- 长期 committed results 有压缩导航且所有原始 Ref 仍可回读；
- Decision 每轮执行全局外循环判断并采用强化完成标准；
- Worker/Reviewer 可动态使用 0–2 个领域专家 Skill，但不把 Skill 当证据；
- Direction 的主要变化最小可检验，联合包的归因边界明确，并以最小充分而非
  无界实验手册形式表达；
- Reviewer/Decision 承担上述语义检查，Script 仍只校验安全调度核心；
- checkpoint 可供人类观察但不成为隐藏 Gate；
- 流式 delta、provider IDs、工具事件和已知 usage 在 Turn 未完成时已经落盘；
- partial、complete message 和 provider terminal 三种状态不会混淆，partial
  永远不能进入结果 commit；
- idle/hard timeout、snapshot-before-interrupt 和 grace finalization 有回归测试；
- 同绑定 runtime retry 会收到简短明确的失败上下文，但不会继承或续写半截 JSON；
- runtime budget 用尽后可通过用户显式、幂等、留痕的 recovery 新建 Turn，旧
  Turn 不被改写；
- 中断恢复、幂等写入、容错渲染和最小机械 completion guard 保持有效。

最终关系为：

> Script 可靠记录“工作流正在发生或已经发生了什么”，只把完整核心消息用于
> 状态机；Worker/Reviewer 形成和审阅“研究内容是什么”；Decision 根据完整
> 需求、压缩记忆、进展轨迹和实际分支后果判断“下一轮由谁执行，以及是否真正
> 可以完成”。

## 19. format version 5 实施记录

本设计已落到以下活动实现：

- `types.ts`、`store.ts`、`runtime.ts`、`live_console.ts`：三态 capture、增量
  runtime/delta/tool/usage 持久化、role-aware idle/hard timeout、
  snapshot-before-interrupt 和 grace；
- `controller.ts`、`prompts.ts`、`recovery.ts`：同绑定 fresh runtime retry、
  DecisionContext 复用、完整消息 replay、结构化
  `RUNTIME_RETRY_EXHAUSTED` 与一次性 `recover-runtime`；
- `workflow.ts`、`observations.ts`：branch-effect preview、幂等 trajectory、容错
  research memory、冻结 decision observation 和 checkpoint；
- D01 只增加 `observationRef`；W01、R01 和 Decision line protocol 未新增控制
  字段；
- Decision Skill 已加入全局外循环和强化完成标准；Worker/Reviewer Skill 已加入
  独立 0–2 专家 Skill 路由、原子主要变化、联合包归因边界和最小充分表达；
- `validation.ts` 只校验上述运输、Ref、状态和恢复一致性，没有新增专业正文长度
  或数组数量 Gate；
- CLI/README 已更新为 format v5，并提供 timeout 初始化参数、`checkpoint`、
  `status` 观察和 `recover-runtime`。

本地自动测试覆盖 37 项协议、端到端、观察、timeout 与 recovery 行为；三个活动
Skill 均通过 `skill-creator` 的结构校验。旧 format 运行不会被 v5 原地修改；
当前未提供旧目录到 v5 的自动迁移，继续运行应初始化新的 v5 work directory。
