# Script–Agent 消息与持久存储设计

状态：与 04、05、06 同步。  
适用实现：`scripts/simple_semantic_loop/refactor/` format version 5。  
权威原则：04 定义 Agent wire 契约，05 定义最小在线 Gate，06 增补
`observationRef`、派生观察、增量 runtime 现场、idle/hard timeout 和显式
recovery；本文其余部分定义这些契约如何落到运行目录和状态机。

## 1. 职责边界

Controller Script 是唯一持久化进程和执行权威：

- 保存不可变需求、对象绑定、Turn、校验、事件和最终产物；
- 根据固定字面量和固定映射推进状态；
- 构造 Agent 可见消息；
- 启动一个临时 Agent Turn；
- 校验输出，并按固定 Decision 字面量、预算和状态条件执行提交、重试、暂停
  或失败；
- 不解释 Agent 正文，不从 guidance 推断任务。

三个 Agent 都是一次性 Turn：

- Worker：完成一个 Anchor 或 Direction 内容任务；
- Reviewer：独立审阅一个 Work Result，并负责对象局部 query gaps；
- Decision：结合完整需求、已提交结论、待决结论和 Script requirement，选择
  一个 Script 允许的下一分支。

辅助 Agent 只能在 Worker 或 Reviewer Turn 内部使用，不是 Controller 的顶层
节点，也不产生顶层通信消息。

## 2. Agent 可见消息

| 编号 | 名称 | 生产者 | 消费者 |
|---|---|---|---|
| G01 | `WORKFLOW_GOAL` | Script | Worker、Reviewer、Decision |
| T01 | `TURN_TASK` | Script | Worker、Reviewer |
| D01 | `DECISION_CONTEXT` | Script | Decision |
| W01 | `WORK_RESULT` | Worker | Reviewer、Decision、Script |
| R01 | `REVIEW_RESULT` | Reviewer | Worker、Decision、Script |
| E01 | `OUTPUT_ERROR_REPORT` | Script | 原输出 Agent |
| O01 | `RUN_OUTCOME` | Script | 调用者 |

除此之外的 JSON 都是 Controller 内部记录。Agent 不遍历内部目录来恢复状态。

## 3. 不可变需求 G01

运行目录根部固定保存：

```text
workflow_goal.json
```

其字段只有：

```json
{
  "topic": "...",
  "objective": "...",
  "acceptanceCriteria": ["..."]
}
```

G01 不包含 ID、模型、预算、路径哈希或运行状态。

## 4. 单 Turn 任务 T01

每个 Worker 或 Reviewer Task 保存为：

```text
tasks/<binding-id>/turn_task.json
```

字段只有：

```json
{
  "goalRef": "workflow_goal.json",
  "action": "CREATE_DIRECTION",
  "objective": "...",
  "inputs": {
    "boundAnchor": "results/<turn-id>.json"
  },
  "requirements": ["..."],
  "constraints": ["..."]
}
```

`action` 固定为：

```text
CREATE_ANCHOR
DEEPEN_ANCHOR
CREATE_DIRECTION
DEEPEN_DIRECTION
REVIEW_ANCHOR
REVIEW_DIRECTION
```

`inputs` 只允许按需出现：

```text
boundAnchor
currentWork
latestReview
reviewTarget
```

固定映射：

| action | role | Result Ref |
|---|---|---|
| `CREATE_ANCHOR` | Worker | `work-result-anchor-v2` |
| `DEEPEN_ANCHOR` | Worker | `work-result-anchor-v2` |
| `CREATE_DIRECTION` | Worker | `work-result-direction-v2` |
| `DEEPEN_DIRECTION` | Worker | `work-result-direction-v2` |
| `REVIEW_ANCHOR` | Reviewer | `review-result-v2` |
| `REVIEW_DIRECTION` | Reviewer | `review-result-v2` |

对象 ID、revision、父 Anchor 和 Schema 名称只保存在 TaskBinding。

## 5. Decision 投影 D01

每次正常 Decision 检查点只写入一个冻结的：

```text
contexts/<context-id>/decision_context.json
```

字段只有：

```json
{
  "goalRef": "workflow_goal.json",
  "committedResults": [],
  "pendingResults": null,
  "remainingRequirementsAfterPendingCommit": []
}
```

`committedResults` 投影每个已提交对象当前版本的 Work/Review 引用；
`PASS`、`REVISE` 和 `REJECT` 都保留，供 Decision 理解有效结论与失败历史。
Direction 额外投影所属 Anchor Work 引用。

`pendingResults` 只投影当前一组已经通过 JSON 解析和角色核心控制字面量校验的：

- 精简 T01；
- W01；
- R01；
- Direction 所属 Anchor Work（若适用）。

`remainingRequirementsAfterPendingCommit` 只允许：

```text
ANCHOR_REQUIRED
ANCHOR_REVIEW_PASS_REQUIRED:<anchor-result-ref>
DIRECTION_REQUIRED:<anchor-result-ref>
DIRECTION_REVIEW_PASS_REQUIRED:<direction-result-ref>
```

D01 不包含 Controller node、round、预算、重试次数、对象 ID 或完成布尔量。

## 6. 内容结果

Worker 的 W01 始终使用同一 envelope：

```json
{
  "workOutcome": "READY_FOR_REVIEW",
  "content": {},
  "evidence": [],
  "unresolved": []
}
```

`workOutcome`：

```text
READY_FOR_REVIEW
PARTIAL_RESULT
BLOCKED_NO_RESULT
```

Anchor 与 Direction 的 `content` Schema 由 T01 action 唯一决定。
`READY_FOR_REVIEW` 返回完整对象且 `unresolved=[]`；`PARTIAL_RESULT` 返回
完整对象且至少有一个 unresolved item；`BLOCKED_NO_RESULT` 的 `content`
必须为 `null` 且至少有一个 blocker。所有内容结果都不返回 patch。

Reviewer 的 R01：

```json
{
  "reviewVerdict": "PASS",
  "summary": "...",
  "findings": [],
  "queryGaps": []
}
```

`reviewVerdict`：

```text
PASS
REVISE
REJECT
```

`PASS` 没有 blocking finding；`REVISE` 至少有一个 blocking finding，且所有
blocking 问题都可在不改变绑定的情况下通过深化同一对象修复；`REJECT` 至少
有一个无法通过深化同一对象修复的 blocking 问题，并在这种问题存在时优先于
`REVISE`。

W01/R01 的 Ref 继续推荐上述完整形状，但在线 wire gate 只要求：

```text
Worker   → 顶层 JSON object + 合法 workOutcome
Reviewer → 顶层 JSON object + 合法 reviewVerdict
```

Worker/Reviewer Turn 不发送完整正文 Provider `outputSchema`。`content`、
`evidence`、`unresolved`、`summary`、`findings`、`queryGaps` 以及未知正文
字段均原样保存。完整 Ref 模板检查只形成 advisory；跨字段关系、专业正确性和
Agent 对 Ref/Task/Goal 的 follow 能力由 Reviewer/Decision 判断。

## 7. Prompt

Worker/Reviewer：

```text
使用 $<skill-name>

本次任务：<absolute-turn-task-path>
Decision guidance：<原样文字或“无”>

按照 Skill 指定的 Result Ref 输出一个 JSON 对象。
```

Decision：

```text
使用 $learning-loop-decision

本次决策上下文：<absolute-decision-context-path>

[ALLOWED_DECISIONS]
- <literal>

[OUTPUT_PROTOCOL]
decision = <一个允许的字面量>
guidance = <可选的不透明自然语言；Script 只保存并转发，不解释>
```

guidance 原样传给后续内容 Agent。Script 不搜索其中的 Anchor、Direction、
create、deepen 或审阅角度关键词。

## 8. Controller 状态机

初始序列：

```text
Worker → Reviewer → Decision
```

Decision 正常分支：

```text
RUN_WORKER
  → 提交 pending
  → Worker → Reviewer → Decision

RUN_REVIEWER
  → 提交 pending
  → Reviewer
  → 将新 R01 提交为目标版本的当前审阅
  → Worker → Reviewer → Decision

FINISH_WORKFLOW
  → 提交 pending
  → 再检查 requirement 为空
  → 写最终报告、manifest 和 O01
  → 停止
```

Decision 语义重试：

```text
RETRY_WORKER
  → 不提交 pending
  → 原 Worker 与依赖 Reviewer 标记为 SUPERSEDED_BY_RETRY
  → 使用同一 Worker TaskBinding
  → Worker → Reviewer → Decision

RETRY_REVIEWER
  → 保留 pending Worker
  → 原 Reviewer 标记为 SUPERSEDED_BY_RETRY
  → 使用同一 Reviewer TaskBinding
  → Reviewer → Decision
```

`RETRY_*` 始终可用于表达当前 pending 不能提交。若对应语义重试预算已经
耗尽，Script 不隐藏该字面量，也不强迫 Decision 改选正常提交；它保留 pending
审计并确定性写 `FAILED` O01。

Decision 使用 `max` reasoning effort；Worker 和 Reviewer 使用 `high`。

## 9. Script 的机械任务选择

Script 不理解 guidance 正文。需要 Worker 时按以下固定优先级绑定 T01：

1. 有 pre-review：先将其提交为目标版本的当前 R01；然后深化该对象；若其
   verdict 为 `REJECT`，退出旧对象并创建同类替代对象；
2. 有未通过的 active Anchor：深化该 Anchor；
3. 有未通过的 active Direction：深化该 Direction；
4. 没有 active Anchor：创建 Anchor；
5. 某个通过的 Anchor 没有未拒绝 Direction：为其创建 Direction；
6. 最低 requirement 已闭合但 Decision 仍选择 `RUN_WORKER`：创建新 Anchor，
   机械扩展由 Anchor 集合定义的 Topic 6L 空间。

Reviewer 的唯一目标来自 `inputs.reviewTarget`。`boundAnchor` 只用于限定
Direction 范围。

深化任务和因 `REJECT` 产生的替代创建任务都通过
`inputs.currentWork + inputs.latestReview` 接收对象与当前审阅。对深化动作，
它们是同一对象的当前版本；对替代创建动作，它们是被拒绝的前序对象及其
R01。Worker 必须处理其中的 blocking findings 和相关 query gaps。

## 10. 最小在线校验、advisory 和 E01

发布的完整 JSON Schema 是 Result Ref 的机器可读模板，不是 Provider 或
Controller 的在线权威 gate。模板缺失、漂移或正文不匹配只进入 doctor /
ValidationAudit advisory，不阻止 Agent Turn 或状态机推进。

每个实际 Agent 输出依次执行：

```text
Codex Turn 中唯一 phase=final_answer 的 completed agentMessage
→
协议或 JSON 解析
→ Worker workOutcome / Reviewer reviewVerdict / Decision decision
→ 状态、引用和预算前提
```

`phase=commentary` 的消息是实时进度与审计数据，不与协议结果拼接。没有显式
final phase 时只允许“整个 Turn 恰好一个非空 phase-unknown 消息”的旧
Provider 兼容情况；其余歧义不由 Script 猜测。完整 Provider 事件仍全部保存
到 RuntimeLog。

失败时原 Turn 进入 `INVALID_OUTPUT`，Script 写：

```json
{
  "errors": [
    {
      "check": "CORE_CONTROL",
      "path": "/workOutcome",
      "message": "expected one of READY_FOR_REVIEW, PARTIAL_RESULT, BLOCKED_NO_RESULT"
    }
  ]
}
```

`check` 只允许：

```text
DECISION_PROTOCOL
JSON_PARSE
CORE_CONTROL
```

重试 Prompt 追加：

```text
[OUTPUT_CORRECTION]
上次输出：<absolute-path>
错误报告：<absolute-E01-path>
正确 Ref：<registered-ref-name>

重做同一任务并返回完整结果。
```

格式/控制错误重试复用同一 TaskBinding 或 D01。E01 只包含无法安全调度的
错误。成功的 JSON/core 检查和非阻断 Ref-template advisories 分栏保存在
ValidationAudit。

Worker/Reviewer 的字段缺失、未知字段、跨字段矛盾、错误 readiness/verdict
等可解析内容问题不生成 E01，也不消耗 output-correction retry；它们进入正常
Reviewer → Decision 语义路径，由 Decision 按允许集合选择 `RETRY_WORKER`
或 `RETRY_REVIEWER`。

Provider 的 `invalid_request_error` 不是 Agent 输出错误；记录一次
`RUNTIME_FAILED` 后立即 `FAILED`。超时、连接中断或无输出等可能瞬态错误才
适用 runtime retry。

## 11. 持久存储

```text
<work-dir>/
  workflow_goal.json
  run.json
  state.json
  ref_catalog.json
  events.jsonl
  bindings/
  tasks/
  contexts/
  turns/
  results/
  audits/
  objects/index.json
  rounds/
  final/
    report.md
    manifest.json
    outcome.json
```

内部记录：

| 记录 | 用途 |
|---|---|
| RunRecord | 模型、Skill/Ref 哈希和预算 |
| TaskBinding | 对象 ID、revision、父 Anchor、action 和 Result Ref |
| TurnRecord | role、Attempt、输入输出引用和 turnState |
| CanonicalState | 当前序列、pending、pre-review、重试和最新决策 |
| CoreControlProjection | 每个成功 Turn 提取出的 `workOutcome`、`reviewVerdict` 或 decision/guidance |
| ObjectIndex | Anchor/Direction revision、归属、结果引用、`workOutcome` 和 `reviewVerdict` |
| RoundIndex | 每轮 Turn 顺序与选择分支 |
| ValidationAudit | 阻断性 JSON/core 检查与非阻断 Ref-template advisories |
| EventLog | append-only 状态变化 |
| RuntimeLog | Provider、工具、usage 和流式事件 |
| FinalManifest | 最终报告实际使用的结果引用 |

Turn 状态只使用：

```text
RUNNING
INVALID_OUTPUT
PENDING_DECISION
COMMITTED
SUPERSEDED_BY_RETRY
RUNTIME_FAILED
```

## 12. 中断恢复

- Turn 已捕获完整输出：即使输出已校验但尚未写入 pending，也不再次调用
  Provider；重放同一输出、执行同一最小校验并完成状态消费；
- Turn 未捕获输出：原 Turn 记为 `RUNTIME_FAILED`，使用同一绑定创建新 Attempt；
- 输出错误：使用同一冻结输入和 E01 创建新 Attempt；
- 对象提交采用对象 ID + revision 比较，重复提交相同引用为幂等，冲突引用失败
  关闭；
- EventLog 和 RuntimeLog 不作为 Agent 指令。

## 13. 最终输出

只有以下条件同时成立才允许 `FINISH_WORKFLOW`：

- pending 提交后的 requirement 列表为空；
- 每个最终 Work Result 为 `READY_FOR_REVIEW`；
- 每个最终 Anchor 和 Direction 的最新 Reviewer verdict 为 `PASS`；
- 每个最终 Anchor 至少有一个未拒绝 Direction。

O01 只有：

```json
{
  "workflowOutcome": "FINISHED",
  "reportRef": "final/report.md",
  "reason": null
}
```

其他结果为 `FAILED` 或 `PAUSED`，且 `reportRef` 为 `null`。

Renderer 不再次审查专业内容，也不把 Ref 模板当隐藏 gate。可识别字段按标准
章节渲染；缺失可选字段不导致完成阶段失败；非标准结果以保真的 JSON 附录
写入报告，manifest 保留实际使用的 Work/Review 引用。

## 14. 版本边界

format version 4 不恢复或用新 gate 重新解释 format version 3 及更旧运行
记录。旧目录保留作审计；重新运行时初始化新的 work directory。v4 对每个
成功 Agent Turn 增加独立 core control projection，并把完整模板不匹配降为
advisory。
