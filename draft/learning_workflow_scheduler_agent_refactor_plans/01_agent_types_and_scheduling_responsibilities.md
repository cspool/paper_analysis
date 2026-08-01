# Agent 类型与调度职能设计

状态：format version 5 已采用；最小在线校验边界以 05 为准，外循环观察、
运行时恢复和内容型 Agent 专家 Skill 路由以 06 为准。  
本文件只定义角色和调度职权；具体 Agent wire 字段以 04 为准。

## 1. 顶层组成

顶层只有：

1. Controller Script；
2. Decision Agent；
3. Worker Agent；
4. Reviewer Agent。

Controller 不是 Agent。它是确定性状态机、持久状态库和唯一执行者。

辅助 Agent 不属于顶层节点。Worker 或 Reviewer 可在本 Turn 内按需使用辅助
Agent，但必须由当前 Turn 汇总为唯一 W01 或 R01。

## 2. 职责总表

| 角色 | 核心职能 | 调度权 | 状态写入权 |
|---|---|---:|---:|
| Controller | 保存事实、绑定任务、提取核心控制字段、执行固定转换 | 机械执行 | 有 |
| Decision | 理解完整需求和结论，选择一个允许分支 | 有限语义选择 | 无 |
| Worker | 创建或深化一个 Anchor/Direction | 无 | 无 |
| Reviewer | 独立审阅一个 Work Result并提出对象局部 query gaps | 无 | 无 |

所有 Agent 都是 fresh one-turn。历史对话不是权威状态。

## 3. Controller

Controller 负责：

- 保存 G01、TaskBinding、Turn、Object、Round、Validation、Event 和 Runtime；
- 根据固定状态机启动 Agent；
- 为 Worker/Reviewer 生成一个精简 T01；
- 为 Decision 生成一个精简 D01 和允许 Decision 集合；
- 保存原始输出；
- 执行传输选择、JSON 和核心控制字面量校验；
- 无法安全提取控制字段时以同一冻结输入和 E01 重试原 Agent；
- 将完整 Ref-template 偏差保存为非阻断 advisory；
- 按 Decision 正式字面量提交、替代、重试、完成、暂停或失败；
- 确定性生成最终人类报告。

Controller 不负责：

- 生成具体 Anchor 或 Direction；
- 生成查询词；
- 判断证据的专业含义；
- 解释 guidance；
- 根据自然语言选择 create/deepen、对象类型、目标或审阅角度；
- 修改用户目标。

## 4. Decision

Decision 是调度控制型 Agent。它读取：

- G01 中的最终 Topic、objective 和 acceptance criteria；
- D01 中全部最新已提交 Worker/Reviewer 结论；
- 当前待决 T01、W01 和 R01；
- pending 提交后的机械 requirement；
- Script 本次允许的正式 Decision。

Decision 先检查当前 core-valid 结果是否遵循 Ref/Task/Goal，以及是否存在会
错误关闭需求或破坏 workflow 的语义问题，再选择一个允许值：

```text
RUN_WORKER
RUN_REVIEWER
FINISH_WORKFLOW
RETRY_WORKER
RETRY_REVIEWER
```

含义：

- `RUN_WORKER`：提交 pending，进入 Worker → Reviewer → Decision；
- `RUN_REVIEWER`：提交 pending，重新审阅一个当前对象，将该 R01 更新为该
  对象当前版本的权威审阅，再进入 Worker → Reviewer → Decision；
- `FINISH_WORKFLOW`：提交 pending 并进入 Script 完成路径；
- `RETRY_WORKER`：不提交 pending，使用同一 Worker TaskBinding 重做；
- `RETRY_REVIEWER`：保留 Worker，使用同一 Reviewer TaskBinding 重做。

Decision 不负责：

- 创建具体研究内容；
- 创建新的状态机节点；
- 编写 T01；
- 修改对象绑定或 Result Ref；
- 给出 Reviewer verdict；
- 启动 Agent；
- 写入或提交状态。

正常 guidance 可选，只能给下一 Agent 提供精简的非权威关注点。重试 guidance
是 Decision 的角色要求，应包含错误、闭合影响、正确 Result Ref 和修正预期；
Script 将它当作可选不透明文本，不检查这些自然语言内容。guidance 不是正式
决策，也不改变 T01。

## 5. Worker

Worker 接收一个 T01，并只执行以下一种 action：

```text
CREATE_ANCHOR
DEEPEN_ANCHOR
CREATE_DIRECTION
DEEPEN_DIRECTION
```

Worker 负责：

- 读取 G01 和 T01 的具名输入；
- 根据当前任务选择知识库维度和查询关键词；
- 使用 Obsidian omnisearch 查询需要的本地维度；
- 深读来源并保留实际 sourceRef；
- 形成完整 Anchor 或 Direction；
- 返回一个符合对应 Work Result Ref 的 W01。

Worker 不决定下一步，不输出调度字段，不写 Controller 状态。深化任务返回完整
修订对象，不返回 patch。

## 6. Reviewer

Reviewer 接收一个 T01，并只执行：

```text
REVIEW_ANCHOR
REVIEW_DIRECTION
```

`inputs.reviewTarget` 是唯一审阅目标；Direction 的 `boundAnchor` 只是范围
上下文。

Reviewer 负责：

- 独立检查目标范围、baseline、证据、机制或性能矛盾；
- 使用 G01 定义的主指标和 guardrails，而不预设 Topic；
- 按对象类型应用 Review Rubric；
- 产生互斥的 `PASS`、`REVISE` 或 `REJECT`；
- 记录对象局部 query gaps 及其解决渠道；
- 返回一个 R01。

Reviewer 每 Turn 可在读取当前 Topic 后选择 0–2 个主 Skill 列出的已安装专家
Skill。专家 Skill 只增强技术检查，不改变角色、目标、证据规则或输出格式。

Reviewer 不调度，不修改 Work Result，不判断全局完成。

## 7. 查询缺口

Reviewer 拥有 query gap 的定义权；Worker 拥有执行查询的内容职责：

```text
Reviewer 提出对象局部问题和 resolution channels
→ Script 将 R01 提交为当前审阅并作为 latestReview 输入
→ Worker 在同一绑定范围内构造查询并深化内容；若 R01 为 REJECT，则创建
  同类替代对象
```

`experiment`、`idea`、`knowledge`、`human` 是缺口解决渠道，不是 Script
调度命令。

## 8. 顶层 Loop

```text
START
  ↓
Worker → Reviewer → Decision
                       ├─ RUN_WORKER ───────→ Worker → Reviewer ─────→ Decision
                       ├─ RUN_REVIEWER ─────→ Reviewer → Worker
                       │                       → Reviewer ────────────→ Decision
                       ├─ RETRY_WORKER ─────→ retry Worker
                       │                       → Reviewer ────────────→ Decision
                       ├─ RETRY_REVIEWER ───→ retry Reviewer ───────→ Decision
                       └─ FINISH_WORKFLOW ──→ Script finalize ──────→ END
```

右侧 Decision 每次都是新的临时 Turn，但逻辑上回到同一个状态机节点。Agent
之间不存在直接调用。

## 9. 控制协议错误与语义错误

传输或核心控制错误：

```text
协议消息歧义 / JSON parse 失败 / 核心字面量缺失或未知
→ Script 写 E01
→ 同一冻结输入和同一绑定创建新 Attempt
```

工作流语义错误：

```text
W01/R01 JSON 和核心字段合法，但正文或跨字段关系可能错误
→ Decision 判断错误会影响闭合
→ RETRY_WORKER 或 RETRY_REVIEWER
→ Script 执行固定恢复分支
```

缺少 Ref 推荐字段、未知正文元素、错误 outcome/verdict、普通专业薄弱、
证据不足或深度不足首先属于 Reviewer/Decision 语义判断，不应伪装为通信错误。

## 10. 权威顺序

发生冲突时：

```text
G01 最终需求
→ T01 action、绑定、requirements、constraints
→ 对应 Skill 和 Result Ref
→ Decision guidance
→ 专家 Skill 或其他参考
```

运行事实只存在于 Controller 存储。Skill 保存稳定方法，不保存运行状态。
