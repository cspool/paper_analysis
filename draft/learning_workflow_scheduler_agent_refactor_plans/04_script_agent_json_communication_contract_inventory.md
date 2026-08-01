# Script–Agent 精简 JSON 通信契约

> 状态：format version 5 的权威 Agent 通信契约；Agent 可见消息采用最小 wire
> shape，W01/R01 完整 JSON Schema 只作为 Result Ref 推荐模板，在线 gate
> 只提取状态机必需的核心控制字面量。06 只为 D01 增加一个
> `observationRef`，不增加 W01/R01/Decision 输出字段；timeout、trajectory、
> memory 和 recovery 都是 Script 内部记录。已同步到活动 Skill、Ref、
> Controller、测试和 README。  
> 前置设计：`01_agent_types_and_scheduling_responsibilities.md`、
> `02_loop_requirement_closure_design.md`、
> `03_script_agent_message_and_storage_contract_design.md`。  
> 依据：当前 `learning-loop-decision`、`learning-loop-worker`、
> `learning-loop-reviewer` Skill/Ref 及 refactor 实现。  
> 范围：Agent 可见的消息类型、JSON 元素和字面量，以及实现必须保持的固定
> 映射。既有旧格式运行数据不原地迁移。

## 1. 设计原则

### 1.1 Script 没有智能

Script 只允许执行以下确定性操作：

- 根据状态机选择下一个固定节点；
- 根据枚举字面量选择固定转换；
- 创建冻结的消息/结果文件和引用，并以版本号更新内部状态；
- 执行传输选择、JSON 解析和核心控制字面量校验；
- 维护对象、Turn、Round、Attempt 和 requirement 索引；
- 根据明确的 Reviewer verdict 更新机械 requirement；
- 根据 Decision 字面量执行固定分支；
- 原样转发 guidance，不理解、归纳或匹配其中的自然语言；
- 按已提交结果引用渲染最终报告。

Script 不允许：

- 理解研究内容；
- 从 summary、finding、query gap、guidance 或 evidence 文本推断分支；
- 判断某个 Anchor/Direction 是否专业正确；
- 根据自然语言猜测“新 Anchor”“新 Direction”或“继续深化”；
- 为含义模糊的 JSON 元素补充语义解释。

任何会改变状态机分支的含义，都必须由固定字面量或固定机械条件表达。

### 1.2 Agent 可见消息最少化

Agent 不再遍历 Controller 的内部 Run、Turn、Object Index、Round、Event 和
Ref Catalog。

每次正常 Turn 最多直接读取：

- 一个不可变 Goal；
- 一个当前 Turn Context；
- Context 明确引用的既有 Work/Review Result。

协议重试时，额外读取一个 Error Report 和上次原始输出。

### 1.3 元素不重复

- Script 已知的 ID、Round、Attempt、revision、Skill、哈希和预算不进入
  Agent 输出；
- Goal 内容不在每个 Task 或 Decision Context 中重复；
- Task 已确定的对象类型不由 Work Result 回显；
- Reviewer Result 不回显被审对象；
- Decision Context 只提供已提交结果、当前待决结果和待决结果提交后的
  requirement；
- Validation 不复制状态、Task、guidance 或结果正文。

### 1.4 字面量表达自身作用域

避免以下无法独立判断作用域的字面量：

```text
ACCEPTED
COMPLETE
COMPLETED
CANDIDATE
eligible = true
```

目标字面量应直接表达：

- 是内容任务、审阅、Turn 还是整个 workflow；
- 是当前已提交状态，还是待决结果提交后的状态；
- 是格式校验结果，还是专业审阅结论；
- 是正常下一轮，还是重试同一任务。

### 1.5 JSON 数据、Ref 模板和在线 gate 必须分层

JSON 消息中的自定义字段与描述这些字段的 JSON Schema 不是同一层：

```text
W01 数据字段：
content.scope6L.L1

描述该字段的 Schema 关键字：
type / anyOf / minLength
```

Codex 可以输出任意 JSON object。W01/R01 的完整模板仍以 Markdown Ref 和
机器可读 JSON Schema 发布，但不作为 Provider `outputSchema`，也不作为
Controller 的阻断性在线契约。

约束归属固定为：

| 层 | 负责的约束 | 例子 |
|---|---|---|
| Script online gate | 只有安全调度直接需要的传输、JSON object 和控制字面量 | Worker `workOutcome`、Reviewer `reviewVerdict`、Decision `decision` |
| Ref + Skill | Agent 应遵循的完整内容合同和方法 | Anchor/Direction 字段、finding、query gap、outcome/verdict 跨字段规则 |
| Reviewer/Decision semantic validation | 需要理解 Goal、Task、Ref、证据和专业内容的正确性 | baseline 是否成立、字段是否缺失、机制是否可证伪、verdict 是否诚实 |
| Template advisory | 诊断完整结果与推荐 Ref 形状的偏差 | 缺少 `measurementPlan`、新增正文元素 |

Script 不检查 content、数组数量、6L 非空、finding 或 outcome/verdict 与正文
的跨字段关系。即使这些关系可被代码表达，也不能穷尽合法语义；它们由
Reviewer/Decision 接管。

完整合同因此是：

```text
Script core gate
+ Skill/Ref semantic contract
+ Reviewer/Decision semantic checks
```

完整模板 lint 和发布文件一致性仅为 advisory，不影响 Turn、output-correction
预算、allowed decisions 或对象状态。Provider 若返回
`invalid_request_error`，仍按不可重试 runtime 配置错误保留，但当前
Worker/Reviewer dispatch 不发送正文 `outputSchema`。

### 1.6 Codex Turn 消息流与协议结果不是同一个边界

Codex App Server 的一个 Turn 可以在工具调用前后产生多个完整
`agentMessage`。这些中间消息即使看起来像完整 JSON，也仍然是 Turn 内进度
消息，不是多个 W01/R01。

Runtime 必须按以下规则处理：

- 所有 `agentMessage` 增量和完整消息都实时转发到控制台；
- 所有原始 Provider 事件都写入 RuntimeLog；
- `phase=commentary` 的消息只属于进度与审计，不进入协议解析；
- 恰好一个 `phase=final_answer` 的非空消息才是交给 Script 协议解析器的
  payload，位置先后不参与判断；
- 某些旧 Provider 可能不提供 phase；仅当整个 Turn 恰好只有一个非空、
  phase-unknown `agentMessage` 时，允许把它作为兼容 payload；
- 多个 `final_answer`，或没有 `final_answer` 且存在多个 phase-unknown 消息，
  都属于传输歧义，Script 不猜测哪个内容更完整；
- 禁止把多个 `agentMessage` 用换行拼接后当作一个 JSON payload。

这是依据 Codex `agentMessage.phase` 的 Provider transport 规则，不是
“最后一条看起来最像答案”，也不是宽松的“从任意文本中寻找一个可解析
JSON”。被显式选中的消息仍必须独立通过 JSON 和角色核心控制字面量校验。

## 2. 目标消息清单

Script–Agent 通信与调用者结果由现状的 J01–J12 精简为七种；其中 G01–E01
是 Agent 通信，O01 只面向调用者：

| 编号 | 消息 | 发送方 → 接收方 | 是否每轮出现 |
|---|---|---|---|
| G01 | `WORKFLOW_GOAL` | Script → 三类 Agent | Run 内复用 |
| T01 | `TURN_TASK` | Script → Worker/Reviewer | 是 |
| D01 | `DECISION_CONTEXT` | Script → Decision | 是 |
| W01 | `WORK_RESULT` | Worker → Script；后续 Agent 可读 | Worker Turn |
| R01 | `REVIEW_RESULT` | Reviewer → Script；后续 Agent 可读 | Reviewer Turn |
| E01 | `OUTPUT_ERROR_REPORT` | Script → 被重试 Agent | 仅输出校验失败 |
| O01 | `RUN_OUTCOME` | Script → 调用者 | Run 结束 |

Decision → Script 继续使用一个非 JSON 的极简行协议：

```text
decision = <一个允许的决策字面量>
guidance = <可选的一行补充说明>
```

不再把以下 Controller 存储文件定义为 Agent 消息：

- `run.json`；
- `state.json`；
- `state_snapshot.json`；
- `current_conclusions.json`；
- `turn.json`；
- `objects/index.json`；
- `rounds/<round>.json`；
- `ref_catalog.json`；
- `validation.json` 的完整审计版本；
- `events.jsonl`；
- `runtime.jsonl`。

其中有用的最小信息分别投影到 G01、T01、D01 和 E01。Controller 原始记录
仍可保留，但 Agent 不再沿引用遍历它们。

## 3. 精简后的通信图

### 3.1 Worker

```text
Prompt
  └─ T01 TURN_TASK
       ├─ G01 WORKFLOW_GOAL
       └─ 任务所需的 W01/R01 引用

Worker Skill + Work Result Ref
  └─ W01 WORK_RESULT
```

### 3.2 Reviewer

```text
Prompt
  └─ T01 TURN_TASK
       ├─ G01 WORKFLOW_GOAL
       ├─ 被审阅 W01
       └─ Direction 审阅所需的 bound Anchor W01

Reviewer Skill + Review Result Ref
  └─ R01 REVIEW_RESULT
```

### 3.3 Decision

```text
Prompt
  ├─ D01 DECISION_CONTEXT
  │    ├─ G01 WORKFLOW_GOAL
  │    ├─ 本轮 W01 对应的 T01
  │    ├─ 已提交 W01/R01
  │    └─ 本轮待决 W01/R01
  └─ 本次允许 Decision 字面量

Decision Skill
  └─ Decision 行协议
```

### 3.4 Retry

```text
原 Prompt
  + 上次原始输出
  + E01 OUTPUT_ERROR_REPORT
  + 正确 Ref 名称
```

语义重试不产生新 JSON 类型：

- `RETRY_WORKER` 仍重输 W01；
- `RETRY_REVIEWER` 仍重输 R01；
- Decision guidance 原样附加到重试 Prompt；
- Script 不解析 guidance。

## 4. G01：WORKFLOW_GOAL

固定文件名：

```text
workflow_goal.json
```

它只保存所有 Agent 都必须理解的不可变需求：

```json
{
  "topic": "多模态推理加速，优先优化延迟，保证较高吞吐",
  "objective": "从本地知识库识别并形成可验证的加速方向",
  "acceptanceCriteria": [
    "Topic 的 6L 空间由 active Anchor 集合动态定义",
    "每个最终 Anchor 必须有独立 Reviewer PASS",
    "每个 active Anchor 至少有一个 Direction",
    "每个最终 Direction 必须有独立 Reviewer PASS",
    "保留延迟、吞吐、质量、硬件、精度、batch、输入输出规模和 SLO 公平性",
    "需要新实验的内容只形成测量计划，不执行实验"
  ]
}
```

仅保留三个元素：

- `topic`：用户原始研究主题；
- `objective`：本工作流最终要形成什么；
- `acceptanceCriteria`：最终结果必须满足的不可变标准。

以下内容只属于 Controller 存储，不进入 G01：

- run ID；
- createdAt；
- projectRoot；
- model；
- Skill 路径与哈希；
- 预算；
- Ref Catalog；
- format version。

## 5. T01：TURN_TASK

固定文件名：

```text
turn_task.json
```

T01 同时覆盖 Worker 和 Reviewer Task，不再定义两类 Task 消息。

```json
{
  "goalRef": "workflow_goal.json",
  "action": "DEEPEN_DIRECTION",
  "objective": "根据最近审阅修订当前 Direction",
  "inputs": {
    "boundAnchor": "results/work-anchor-r2.json",
    "currentWork": "results/work-direction-r1.json",
    "latestReview": "results/review-direction-r1.json"
  },
  "requirements": [
    "明确相对 baseline 的唯一改变",
    "补足延迟、吞吐和质量门控",
    "保留最强反例和失败条件"
  ],
  "constraints": [
    "不得改变绑定 Anchor",
    "不得执行新实验"
  ]
}
```

### 5.1 `action` 字面量

```text
CREATE_ANCHOR
DEEPEN_ANCHOR
CREATE_DIRECTION
DEEPEN_DIRECTION
REVIEW_ANCHOR
REVIEW_DIRECTION
```

字面量使用“动作 + 对象”顺序，直接说明本 Turn 的唯一工作。

### 5.2 `inputs` 固定键

`inputs` 只允许以下按需出现的键：

```text
boundAnchor
currentWork
latestReview
reviewTarget
```

语义：

- `boundAnchor`：Direction 必须保持归属的 Anchor Work Result；
- `currentWork`：`DEEPEN_*` 时是需要深化的当前 Work Result；
  `REJECT` 后的替代 `CREATE_*` 时是不得照搬的被拒绝前序 Work Result；
- `latestReview`：与 `currentWork` 配对、需要处理的当前 Reviewer Result；
- `reviewTarget`：Reviewer 本次唯一被审阅的 Work Result。

使用具名引用而不是无语义的 `inputRefs` 数组，避免 Agent 自行猜测每个路径的
用途。

固定组合：

| action | required inputs | optional inputs |
|---|---|---|
| `CREATE_ANCHOR` | 无 | 替代创建时 `currentWork + latestReview` 成对出现 |
| `DEEPEN_ANCHOR` | `currentWork + latestReview` | 无 |
| `CREATE_DIRECTION` | `boundAnchor` | 替代创建时 `currentWork + latestReview` 成对出现 |
| `DEEPEN_DIRECTION` | `boundAnchor + currentWork + latestReview` | 无 |
| `REVIEW_ANCHOR` | `reviewTarget` | 无 |
| `REVIEW_DIRECTION` | `boundAnchor + reviewTarget` | 无 |

`currentWork` 与 `latestReview` 不得只出现一个。

### 5.3 删除的元素

Agent 可见 T01 不包含：

- `taskId`；
- `target.objectId`；
- `baseRevision`；
- `parentAnchorId`；
- `outputTemplate`。

这些绑定信息保存在 Controller 内部 TaskRecord。

`action → Agent role → Result Schema/Ref` 使用固定映射：

| action | role | result |
|---|---|---|
| `CREATE_ANCHOR` | Worker | W01 + Anchor content schema |
| `DEEPEN_ANCHOR` | Worker | W01 + Anchor content schema |
| `CREATE_DIRECTION` | Worker | W01 + Direction content schema |
| `DEEPEN_DIRECTION` | Worker | W01 + Direction content schema |
| `REVIEW_ANCHOR` | Reviewer | R01 |
| `REVIEW_DIRECTION` | Reviewer | R01 |

Script 只查固定映射表，不理解 Task 文本。

## 6. D01：DECISION_CONTEXT

固定文件名：

```text
decision_context.json
```

D01 取代：

- `state_snapshot.json`；
- `current_conclusions.json`；
- Decision 对 `turn.json`、`objects/index.json`、`task.json` 和
  `ref_catalog.json` 的遍历。

固定形状：

```json
{
  "goalRef": "workflow_goal.json",
  "committedResults": [
    {
      "objectKind": "ANCHOR",
      "work": "results/work-anchor-r2.json",
      "review": "results/review-anchor-r2.json"
    },
    {
      "objectKind": "DIRECTION",
      "anchorWork": "results/work-anchor-r2.json",
      "work": "results/work-direction-r1.json",
      "review": "results/review-direction-r1.json"
    }
  ],
  "pendingResults": {
    "objectKind": "DIRECTION",
    "anchorWork": "results/work-anchor-r2.json",
    "workTask": "tasks/turn-task-direction-r2.json",
    "work": "results/work-direction-r2.json",
    "review": "results/review-direction-r2.json"
  },
  "remainingRequirementsAfterPendingCommit": []
}
```

### 6.1 `committedResults`

包含每个已提交对象当前版本的 Work/Review 对。这里的“当前”包括后来通过
`RUN_REVIEWER` 提交的新审阅；`PASS`、`REVISE` 和 `REJECT` 都保留。

它不表示 Reviewer 一定 `PASS`。Decision 必须读取对应 R01 的
`reviewVerdict` 和正文；保留 `REJECT` 结论可避免后续内容重复已经否定的
路线。结构无效、被语义重试替代或从未提交的结果不进入这里。

`committed` 只表达：

> 这些结果已经进入 Controller 权威工作流历史。

### 6.2 `pendingResults`

只包含本次 Decision 正在判断的一组核心控制字段合法结果：

- 产生当前 W01 的一个精简 T01；
- 一个 W01；
- 依赖该 W01 的一个 R01；
- Direction 时增加所属 Anchor Work Result 引用。

`pending` 只表达：

> 这些结果已经通过 JSON object 和核心控制字面量校验，但尚未提交或被重试
> 替代；其正文可能不符合 Ref 或存在语义错误。

Decision 使用 `workTask` 核对 W01 是否改变对象范围、遗漏本次 requirement
或违反 constraint。`workTask` 是 Agent 已读的精简 T01，不是 Controller
内部 TaskBinding。

不再使用 `retryCandidates`。待决结果可以被正常提交，也可以被重试，
不能预先称为“重试候选”。

### 6.3 `remainingRequirementsAfterPendingCommit`

该数组明确表达时间语义：

> 如果当前 pending W01/R01 按正常分支提交，仍有哪些机械 requirement
> 没有闭合。

允许的 requirement 关键词：

```text
ANCHOR_REQUIRED
ANCHOR_REVIEW_PASS_REQUIRED:<anchor-result-ref>
DIRECTION_REQUIRED:<anchor-result-ref>
DIRECTION_REVIEW_PASS_REQUIRED:<direction-result-ref>
```

空数组表示：

```text
提交当前 pending results 后，机械 requirement 全部闭合。
```

它不表示：

- 当前 pending results 已经提交；
- Decision 必须选择完成；
- 内容不存在语义错误；
- Decision 已完成专业判断。

Script 只能根据以下机械事实生成 requirement：

- 是否存在 active Anchor；
- Anchor 最新 W01 是否为 `READY_FOR_REVIEW` 且 R01 是否为 `PASS`；
- 每个 active Anchor 是否至少有一个未被 `REJECT` 的 Direction；
- Direction 最新 W01 是否为 `READY_FOR_REVIEW` 且 R01 是否为 `PASS`；
- Direction 是否仍绑定当前 Anchor revision。

Script 不读取 R01 的 summary、finding、query gap 或自然语言内容来生成
requirement。

### 6.4 删除的元素

D01 不包含：

- state revision；
- lifecycle；
- round；
- node；
- focus；
- 当前 `coverage`；
- retry budgets；
- object index path；
- completion booleans；
- `eligible`；
- candidate Turn ID。

这些内容要么属于 Controller 审计，要么已经由 Prompt 中的允许 Decision
集合表达。

## 7. W01：WORK_RESULT

Anchor 和 Direction 不再是两个顶层消息类型。它们统一使用一个 Work
Result envelope，由 T01 `action` 确定 `content` 的 Schema。

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {
    "name": "...",
    "mechanism": "...",
    "baselineChange": "...",
    "expectedEffects": [
      {
        "metric": "...",
        "effect": "...",
        "conditions": "..."
      }
    ],
    "tradeoffs": ["..."],
    "failureConditions": ["..."],
    "measurementPlan": ["..."]
  },
  "evidence": [
    {
      "sourceRef": "vault/path.md#heading",
      "supports": "..."
    }
  ],
  "unresolved": ["..."]
}
```

### 7.1 `workOutcome` 字面量

```text
READY_FOR_REVIEW
PARTIAL_RESULT
BLOCKED_NO_RESULT
```

精确定义：

- `READY_FOR_REVIEW`：本次 T01 要求的完整 Work content 已产生，可以进入
  独立 Reviewer；
- `PARTIAL_RESULT`：产生了可保留内容，但明确没有覆盖全部 T01 requirement；
- `BLOCKED_NO_RESULT`：缺少必要输入或发生不可在本 Turn 内解决的阻塞，
  没有可审阅 Work content。

约束：

- `READY_FOR_REVIEW` 时 `content` 满足当前 action 的完整 schema，且
  `unresolved=[]`；
- `PARTIAL_RESULT` 时 `content` 满足完整 schema，且 `unresolved` 至少有
  一个未完成项；
- `BLOCKED_NO_RESULT` 时 `content=null`，且 `unresolved` 至少有一个
  blocker；
- `READY_FOR_REVIEW` 只表示有界 Worker 任务产出可审阅，不表示 Reviewer
  PASS，更不表示 workflow 完成。

### 7.2 `content` 的两种固定 Schema

T01 为 Anchor action 时：

```json
{
  "name": "...",
  "scenario": "...",
  "baseline": "...",
  "performanceTension": "...",
  "scope6L": {
    "L1": "该 Anchor 涉及的算法或流水线对象",
    "L2": "该 Anchor 涉及的服务或运行时对象",
    "L3": null,
    "L4": null,
    "L5": null,
    "L6": null
  },
  "constraints": ["..."]
}
```

T01 为 Direction action 时：

```json
{
  "name": "...",
  "mechanism": "...",
  "baselineChange": "...",
  "expectedEffects": [
    {
      "metric": "...",
      "effect": "...",
      "conditions": "..."
    }
  ],
  "tradeoffs": ["..."],
  "failureConditions": ["..."],
  "measurementPlan": ["..."]
}
```

不增加 `objectKind`、ID、revision、Task 或调度字段。Agent 根据 T01 action
选择对应 Result Ref；Script 只使用 action 判断应提取哪个核心字段。

### 7.3 W01 约束分配

Script 在线只表达：

- 顶层是一个 JSON object；
- `workOutcome` 是三个允许字面量之一。

Result Ref 推荐完整 content、evidence 和 unresolved 结构，并保留
`scope6L.L1`–`L6` 的清晰表达。完整模板 lint 可记录缺失、类型偏差和未知
字段，但只能形成 advisory。

Reviewer 判断 content、outcome、evidence、6L 和 unresolved 是否符合
Goal/Task/Ref；Decision 复核 Worker/Reviewer 是否会错误闭合需求，并在必要
时选择 `RETRY_WORKER`。

## 8. R01：REVIEW_RESULT

```json
{
  "reviewVerdict": "REVISE",
  "summary": "...",
  "findings": [
    {
      "severity": "BLOCKING",
      "issue": "...",
      "basis": "...",
      "expected": "..."
    }
  ],
  "queryGaps": [
    {
      "question": "...",
      "dimension": "experiment",
      "reason": "..."
    }
  ]
}
```

### 8.1 `reviewVerdict` 字面量

```text
PASS
REVISE
REJECT
```

精确定义：

- `PASS`：没有 `BLOCKING` finding，当前 Work Result 可进入最终结果；
- `REVISE`：至少有一个 `BLOCKING` finding，且所有 blocking 问题都可在
  不改变对象绑定的情况下通过深化同一对象修复；
- `REJECT`：至少有一个 `BLOCKING` 问题无法通过深化同一对象修复，当前
  对象不得进入最终结果；只要存在这类问题，`REJECT` 优先于 `REVISE`。

`PASS` 仅属于 `reviewVerdict`，不用于 JSON、Schema、Turn 或 workflow 状态。
“能否在同一对象内修复”是 Reviewer 的专业判断；Script 不从 finding 文本
推断它。

### 8.2 其他字面量

`severity`：

```text
BLOCKING
NON_BLOCKING
```

`dimension`：

```text
experiment
idea
knowledge
human
```

角色语义约束：

- `PASS` 不得含 `BLOCKING` finding；
- `REVISE` 和 `REJECT` 都至少含一个 `BLOCKING` finding；
- query gap 只描述对象局部缺口，不包含调度命令。

每个 query gap 只有一个 `dimension`。同一宽泛不确定性需要多个 resolution
channel 时，Reviewer 将其拆成多个有界 query gap。

这些规则均由 Reviewer 按 Skill 判断，并由后续 Decision 读取完整 R01 进行
语义复核；Script 不根据 finding 内容或数量重写 verdict。

### 8.3 R01 约束分配

Script 在线只表达：

- 顶层是一个 JSON object；
- `reviewVerdict` 是 `PASS | REVISE | REJECT`。

`dimension` 使用单个枚举而不是数组。它直接绑定一个解决通道；多通道问题
拆分为多个 gap，让每个 gap 更容易由后续内容 Agent 使用。

完整 Review Ref 模板只做 advisory lint。Reviewer 负责判断 finding 是否
真实、是否 blocking、同对象是否可修复、非标准 Worker 正文是否仍可理解；
Decision 负责识别核心字段合法但会破坏 workflow 的漏审或 verdict 语义错误，
并在必要时选择 `RETRY_REVIEWER`。

## 9. E01：OUTPUT_ERROR_REPORT

E01 只在传输、JSON 或核心控制协议错误重试时提供给原 Agent。

```json
{
  "errors": [
    {
      "check": "CORE_CONTROL",
      "path": "/reviewVerdict",
      "message": "expected one of PASS, REVISE, REJECT"
    }
  ]
}
```

只保留 `errors`。不保存成功检查，不保存 workflow semantic 状态。

`check` 字面量：

```text
DECISION_PROTOCOL
JSON_PARSE
CORE_CONTROL
```

完整 Validation 审计可以保存在 Controller 内部，但 Agent 通信只使用 E01。

完整 Ref-template 偏差不生成 E01。Provider
`invalid_request_error` 也不属于 Agent 输出错误，因此不生成 E01；防御性
运行时分类直接写 `RUNTIME_FAILED` 和 `FAILED` O01，并在 reason 中保留
Provider 错误。

## 10. Decision 行协议

Decision 输出保持两个元素，不改为 JSON：

```text
decision = RUN_WORKER
guidance = 可选的一行补充说明
```

### 10.1 决策字面量

```text
RUN_WORKER
RUN_REVIEWER
FINISH_WORKFLOW
RETRY_WORKER
RETRY_REVIEWER
```

精确定义：

#### `RUN_WORKER`

- 提交当前 pending W01/R01；
- 按固定正常序列启动 Worker → Reviewer → Decision；
- guidance 原样转发给新 Worker/Reviewer；
- Script 不解析 guidance。

#### `RUN_REVIEWER`

- 提交当前 pending W01/R01；
- 按固定序列启动 Reviewer → Worker → Reviewer → Decision；
- 第一个 Reviewer 的合法 R01 会替换目标对象当前版本的旧 R01，并立即更新
  该对象是否 rejected；随后 Worker 通过 `latestReview` 接收该结论；
- 新 R01 为 `REJECT` 时，Script 创建同类替代对象；否则深化同一对象；
- guidance 原样转发；
- Script 不解析 guidance。

#### `FINISH_WORKFLOW`

- 只在 `remainingRequirementsAfterPendingCommit` 为空时由 Script 提供；
- 提交当前 pending W01/R01；
- 生成最终报告；
- 写入 O01；
- 停止工作流。

Decision 仍须检查最近 W01/R01 是否存在会导致错误闭合的语义错误。即使
Script 提供 `FINISH_WORKFLOW`，Decision 也可以选择正常继续或语义重试。

#### `RETRY_WORKER`

- 不提交当前 pending W01；
- 不提交依赖它的 pending R01；
- 两者标记为被 Worker retry 替代；
- 使用同一个 Controller TaskBinding 重启 Worker；
- Decision 角色应让 guidance 说明语义错误、影响和正确 Work Result Ref；
- 若 Worker 语义重试预算已耗尽，pending 保持未提交，Script 写 `FAILED`
  O01，不启动新 Attempt。

#### `RETRY_REVIEWER`

- 保留当前 pending W01；
- 不提交当前 pending R01；
- 当前 R01 标记为被 Reviewer retry 替代；
- 使用同一个 Controller TaskBinding 重启 Reviewer；
- Decision 角色应让 guidance 说明语义错误、影响和正确 Review Result Ref；
- 若 Reviewer 语义重试预算已耗尽，pending 保持未提交，Script 写 `FAILED`
  O01，不启动新 Attempt。

`RETRY_WORKER` 或 `RETRY_REVIEWER` 在存在对应 pending 时始终保留在允许集合。
预算限制控制是否还能启动新 Attempt，不得把 Decision 逼到会提交已判错误结果
的正常分支。

上述 retry guidance 是 Decision Skill 的语义要求；Script 不要求 guidance
存在、不搜索 Ref 名称，也不据其内容改变分支。

### 10.2 guidance 的边界

guidance：

- 是给下一 Agent 的非权威内容提示；
- 不改变对象绑定；
- 不改变 Task action；
- 不改变 Result Ref 或核心控制协议；
- 不被 Script 解析；
- 不受 Script 的关键词、内容或字符数校验；
- 与 T01 冲突时以 T01 为准。

因此 Script 不再通过关键词匹配 guidance 来选择：

- create/deepen；
- Anchor/Direction；
- Task target；
- Reviewer 角度；
- requirement 状态。

## 11. O01：RUN_OUTCOME

向调用者返回的结果精简为：

```json
{
  "workflowOutcome": "FINISHED",
  "reportRef": "final/report.md",
  "reason": null
}
```

`workflowOutcome` 字面量：

```text
FINISHED
FAILED
PAUSED
```

精确定义：

- `FINISHED`：最终 pending 已提交、机械 requirement 已闭合、最终报告已写入；
- `FAILED`：Controller 无法继续且未形成可提交的最终结果；
- `PAUSED`：保留可恢复状态。因 `maxRounds` 暂停时，下一 Round 和固定序列
  已准备好，但只有显式 `resume` 才继续调用 Agent。

Controller 内部可以继续保存：

- final state revision；
- source result refs；
- 结束事件；
- 失败栈和恢复信息。

这些审计信息不需要出现在调用者的最小 O01 中。

## 12. Controller 内部存储

以下数据继续存在，但不再作为 Decision/Worker/Reviewer 的输入消息：

| 内部记录 | 作用 |
|---|---|
| RunRecord | ID、时间、模型、Skill/Ref 哈希和预算 |
| TaskBinding | Task ID、对象 ID、revision、父 Anchor、action 和 Result Ref |
| TurnRecord | role、Attempt、输入、原始输出、完整结果、core projection 和 Turn 状态引用 |
| CoreControlProjection | Worker `workOutcome`、Reviewer `reviewVerdict`、Decision decision/guidance |
| CanonicalState | 当前节点、序列、预算和最新引用 |
| ObjectIndex | Anchor/Direction revision、归属、结果引用、`workOutcome` 和 `reviewVerdict` |
| RoundIndex | Round 内 Turn 顺序 |
| ValidationAudit | 阻断性 JSON/core 检查和非阻断 Ref-template advisory |
| EventLog | append-only 状态变化 |
| RuntimeLog | Provider、工具和流式输出审计 |
| FinalManifest | 最终报告实际使用的结果引用 |

Agent 需要的信息由 Script 使用固定字段拷贝和引用规则投影到 G01、T01、
D01 或 E01。投影过程不得总结自然语言内容。

## 13. Controller 内部状态字面量

虽然 TurnRecord 不再是 Agent 消息，其日志字面量也应消除歧义。

固定 `turnState`：

```text
RUNNING
INVALID_OUTPUT
PENDING_DECISION
COMMITTED
SUPERSEDED_BY_RETRY
RUNTIME_FAILED
```

定义：

- `RUNNING`：Provider Turn 尚未结束；
- `INVALID_OUTPUT`：输出未通过传输协议、JSON 或核心控制字面量校验；
- `PENDING_DECISION`：核心控制字段合法，等待 Decision 决定提交或语义重试；
- `COMMITTED`：结果已进入 Controller 权威工作流历史；正常 W01/R01 对和
  `RUN_REVIEWER` 产生的当前 R01 都使用该状态；
- `SUPERSEDED_BY_RETRY`：结果未提交或已退出当前有效链，由新 Attempt 替代；
- `RUNTIME_FAILED`：Provider Turn 未产生可校验结果。

不再使用：

```text
ACCEPTED
CANDIDATE
SEMANTIC_INVALID
STRUCTURE_INVALID
PROTOCOL_INVALID
FAILED
```

具体错误类型由 E01/ValidationAudit 的 `check` 和错误内容表达，不在
`turnState` 重复。

ValidationAudit 不再保存重复的：

```text
workflowSemantic.status = ACCEPTED | RETRY
```

是否提交或重试由 TurnRecord `turnState`、Decision Turn 引用和 EventLog
唯一表达。

## 14. 固定机械映射

Script 只执行以下表驱动逻辑。

### 14.1 Task → Result

```text
CREATE_ANCHOR     → Worker → W01/Anchor Ref
DEEPEN_ANCHOR     → Worker → W01/Anchor Ref
CREATE_DIRECTION  → Worker → W01/Direction Ref
DEEPEN_DIRECTION  → Worker → W01/Direction Ref
REVIEW_ANCHOR     → Reviewer → R01
REVIEW_DIRECTION  → Reviewer → R01
```

### 14.2 Review → Requirement

```text
READY_FOR_REVIEW + PASS → 对应 *_REVIEW_PASS_REQUIRED requirement 可关闭
REVISE  → requirement 保持打开；正常 Worker 应深化同一对象
REJECT  → 当前对象退出可闭合对象集合；后续 Worker 创建同类替代
```

Script 只匹配 `reviewVerdict` 字面量，不读取审阅正文。

### 14.3 Decision → Sequence

```text
RUN_WORKER       → commit pending → Worker → Reviewer → Decision
RUN_REVIEWER     → commit pending → Reviewer → commit current R01
                 → Worker → Reviewer → Decision
FINISH_WORKFLOW  → commit pending → render → O01 → stop
RETRY_WORKER     → supersede pending pair → retry same Worker task
RETRY_REVIEWER   → retain Worker → supersede Reviewer → retry same Reviewer task
```

对两个 `RETRY_*`，预算耗尽时映射为“保留 pending → FAILED O01”，而不是改选
正常提交。

### 14.4 Error → Retry

```text
DECISION_PROTOCOL → retry same Decision context
JSON_PARSE        → retry same T01
CORE_CONTROL      → retry same T01
```

Script 不调用 Decision 解释无法安全提取控制字面量的错误；Ref 字段、跨字段和
专业语义错误则不进入这张表，正常流向 Reviewer/Decision。

### 14.5 Ref template → advisory

```text
完整 W01/R01 template
  → published template consistency lint
  → ValidationAudit / doctor advisory
  → 不改变 Agent Turn 或状态机
```

Worker/Reviewer dispatch 不发送完整正文 `outputSchema`。若 Provider 仍因
其他请求参数返回 `invalid_request_error`：

```text
记录一次 RUNTIME_FAILED
→ 不做相同请求的 runtime retry
→ FAILED O01
```

超时、连接中断和无输出等可能瞬态的 runtime failure 才适用 runtime retry
预算。

## 15. 当前问题如何被消除

### 15.1 当前 coverage 与候选完成事实混用

删除 Agent 可见的：

```text
coverage
completionFacts
eligible
```

改为一个具有明确时间语义的：

```text
remainingRequirementsAfterPendingCommit
```

它只回答：

> 如果提交当前 pending results，仍有什么机械 requirement 没闭合？

### 15.2 `retryCandidates` 含义错误

删除 `retryCandidates`。

当前待 Decision 判断的结果统一放入：

```text
pendingResults
```

是否重试由 Decision 字面量决定，而不是由字段名预判。

### 15.3 `ACCEPTED` 与 Reviewer `PASS` 混淆

Turn 使用：

```text
COMMITTED
```

Reviewer 使用：

```text
reviewVerdict = PASS
```

两者不再共享“接受/通过”的模糊含义。

### 15.4 Worker `COMPLETE` 与 workflow `COMPLETE` 混淆

Worker 使用：

```text
workOutcome = READY_FOR_REVIEW
```

Decision 使用：

```text
FINISH_WORKFLOW
```

最终调用者使用：

```text
workflowOutcome = FINISHED
```

三个层级由字面量自身表达，不依赖 Agent 根据字段位置猜测。

### 15.5 Validation 重复保存语义状态

Agent 可见 E01 只保存错误。

Controller 内部是否提交、重试或替代只由 TurnRecord 和 EventLog 表达，
不再在 Validation 中复制一份 `workflowSemantic`。

### 15.6 Decision 需要遍历过多内部记录

D01 直接提供：

- 已提交最新结果；
- 当前待决结果；
- 当前待决 W01 对应的精简 T01；
- 待决结果提交后的剩余机械 requirement。

Decision 不再读取 Controller 内部 Turn、TaskBinding、ObjectIndex、
RefCatalog 或 CanonicalState。

### 15.7 JSON 字段可自定义与 Script 是否应阻断被混淆

保留 W01/R01 必要的自定义业务字段和易于理解的推荐 shape：

```text
scope6L.layers + region
  → scope6L.L1 ... scope6L.L6（string | null）

queryGap.dimensions[]
  → queryGap.dimension；多通道拆成多个 gap
```

这些完整字段用于 Agent 专业表达和 Reviewer/Decision 理解，不用于 Script
强制正文。即使字段/数组形状偏离模板，只要核心字段合法也进入语义链路。

因此不采用以下两种错误修复：

- 为了固定正文继续把完整 Schema 传给 Provider；
- 把语义内容编码成 bitmask、组合枚举或更多控制 JSON 元素。

## 16. format version 3 到 format version 4

format version 4 相对 format version 3 的 wire-gate 变化：

| format version 3 | format version 4 |
|---|---|
| 完整 W01/R01 Provider `outputSchema` | Worker/Reviewer `outputSchema=null` |
| Script 阻断完整 Schema 和跨字段机械错误 | Script 只阻断 JSON/core-control 错误 |
| `SCHEMA`、`MECHANICAL` E01 | `CORE_CONTROL` E01 |
| template mismatch 消耗 output retry | template mismatch 只写 advisory |
| Turn 仅引用完整结果 | Turn 另引用独立 `control.json` |
| ObjectIndex 只索引结果/Review | ObjectIndex 另索引 `workOutcome`/`reviewVerdict` |
| Renderer 假定完整强类型正文 | Renderer 容错渲染并保真附录非标准结果 |
| `run/state.formatVersion = 3` | `run/state.formatVersion = 4` |

v3 及更旧运行目录不原地迁移或恢复。消息类型仍是 G01、T01、D01、W01、
R01、E01 和 O01；变化的是 W01/R01 的有效性边界与 Controller 内部投影。

消息类型变化：

```text
旧 Agent 可见/可遍历 JSON：J01–J12
format version 4 Agent 通信 JSON：G01、T01、D01、W01、R01、E01
调用者 JSON：O01
```

### 16.1 format version 5 的 06 增量

format version 5 保持上述 Agent 输出 wire 不变，只做以下增量：

- D01 增加一个 `observationRef`，指向冻结的 Script 派生观察；
- W01、R01 和 Decision line protocol 不增加 trajectory、memory、runtime 或
  recovery 字段；
- Turn 内部区分 `outputCapture=NONE|PARTIAL|COMPLETE`，只有 COMPLETE 的唯一
  协议消息可成为 `rawOutputRef`；
- role-aware idle/hard timeout、partial、runtime error、trajectory、research
  memory、checkpoint 和 recovery record 都是 Agent 不需要回显的 Script 记录；
- format version 4 及更旧目录由 v5 Controller 明确拒绝，不原地重解释。

## 17. Prompt 最小形状

### 17.1 Worker/Reviewer

```text
使用 $<skill_name>

本次任务：<absolute_path_to_turn_task.json>
Decision guidance：<原样文字；无则写“无”>

按照 Skill 指定的 Result Ref 输出一个 JSON 对象。
```

### 17.2 Decision

```text
使用 $<decision_skill_name>

本次决策上下文：<absolute_path_to_decision_context.json>

[ALLOWED_DECISIONS]
- <allowed literal>
- ...

[OUTPUT_PROTOCOL]
decision = <一个允许的字面量>
guidance = <可选的不透明自然语言；Script 只保存并转发，不解释>
```

### 17.3 输出校验失败重试

```text
[OUTPUT_CORRECTION]
上次输出：<absolute_path>
错误报告：<absolute_path_to_output_error_report.json>
正确 Ref：<registered_ref_name>

重做同一任务并返回完整结果。
```

Prompt 不复述 JSON 中已有内容。

## 18. 实现同步范围

format version 4 的以下部分必须始终使用同一组元素和字面量：

1. 03 消息与存储设计；
2. Decision、Worker、Reviewer Skill；
3. Work/Review Result Ref；
4. 作为 Ref 模板的 JSON Schema；
5. Script core-control extractor 和非阻断 template advisory；
6. TypeScript 类型；
7. Prompt Builder；
8. Controller 投影、校验、提交和重试逻辑；
9. 最终 Renderer；
10. 单元测试、端到端测试和 README；
11. 日志状态名和迁移说明。

当前活动实现已按此清单同步，并通过联合校验。后续修改任一项时必须同时检查
其余项；旧格式运行目录只保留审计，不与当前状态机混合恢复。
