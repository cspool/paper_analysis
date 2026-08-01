# 冻结 Decision 快照、语义收敛与 Runtime Delta 去重设计

> 状态：已于 2026-08-01 实施并通过本地测试。  
> 前置设计：`05_minimal_controller_validation_and_semantic_handoff_design.md`、
> `06_outer_loop_memory_trajectory_and_atomic_direction_design.md`。  
> 实施目标：修复 format version 5 真实运行暴露的 Decision 历史输入漂移、
> 全局完成依据不足、Direction 规格膨胀、Decision 重复回读和 runtime delta
> 重复存储问题。  
> 本文只覆盖已经确认采纳的 P0、全部 P1，以及 P2 的 runtime delta 去重；
> provider 元数据、最终 manifest 哈希、运行目录提示和专家 Skill 路由不在本轮
> 范围内。

## 1. 已确认采纳的修改

本轮采纳五项修改：

1. P0：Decision 使用的 research memory 和 progress trajectory 必须是真正不可变的
   本轮快照，不能继续只引用会被后续刷新或追加的全局观察文件；
2. P1：Decision 的 `FINISH_WORKFLOW` 必须区分“对象局部没有 query gap”和
   “Topic 的下一轮扩展预期信息增益较低”，宽泛 Goal 需要可回读的安静扩展证据；
3. P1：Worker 和 Reviewer 必须阻止 Direction 从最小可证伪说明逐轮膨胀为未来
   实验执行手册；
4. P1：Decision 应以冻结压缩记忆和 pending 结论为默认输入，减少回读旧 revision，
   并停止承担论文或知识库的独立内容研究；
5. P2：`item/agentMessage/delta` 在 `runtime.jsonl` 中只保存一份规范化事件，取消
   同一 delta 同时写入 `raw_event` 和 `output_delta`。

这些修改共同遵守一条边界：

> Script 继续只负责机械快照、持久化、核心字面量和状态转换；是否有新 Anchor、
> 是否语义饱和、某个 Reviewer 要求是否过度、某个 Direction 是否最小充分，仍由
> Worker、Reviewer 和 Decision 理解，不转化为 Script 的专业内容 Gate。

## 2. format version 5 运行暴露的问题

审计对象：

```text
learning_outputs_codex/multimodal_inference_latency_first_v5_20260731
```

该运行的运输层和核心状态机是健康的：

- 6 个完整 Decision 周期；
- 18 个 fresh Turn；
- 2 个 Anchor、2 个最终 PASS Direction；
- 无格式纠错、语义 retry、runtime retry 或 timeout；
- 最长 Worker Turn 约 394 秒，新的 idle/hard timeout 正常工作；
- 全部 Worker/Reviewer 原始 JSON 与提交 Result 一致。

问题集中在观察输入、语义收敛和成本，而不是 JSON 支持或强校验不足。

### 2.1 所谓冻结观察仍引用可变全局文件

第一轮 `decision_observation.json` 记录：

```text
stateRevision = 9
accepted = 0 Anchor / 0 Direction
researchMemoryRef = observations/research_memory.json
trajectoryRef = observations/progress_trajectory.jsonl
```

运行结束后，同一 `researchMemoryRef` 已被更新为 state revision 67，并包含 2 个
Anchor、2 个 Direction 和全部 Decision trail；全局 trajectory 也已经追加至第
6 轮。

因此：

- Agent 在当时运行时读到的内容是正确的；
- 但历史 DecisionContext 现在无法恢复当时的完整输入；
- `checkpoint` 再次刷新全局 research memory 时，还会继续改变所有旧 Context
  间接引用到的内容；
- 当前 validator 只检查 Ref 存在，不检查 observation revision 与 memory revision
  一致，因而无法发现漂移。

这是持久化和重放正确性缺陷，属于 P0。

### 2.2 最终完成缺少全局安静扩展证据

第 5 轮 `CREATE_ANCHOR` 仍成功发现一个新的 L3–L4 Anchor，第 6 轮为其创建并
通过一个 Direction，随后 Decision 立即选择 `FINISH_WORKFLOW`。

该决定满足 Script 的机械闭环，但下面的推理证据不足：

```text
openQueryGaps = []
→ Topic 没有重要缺口
→ 再扩展一轮的信息增益较低
```

原因是 Reviewer 的 `queryGaps` 按既有设计只表示当前被审对象的局部问题；它不
负责询问“知识库中是否还有第三个重要 Anchor”。最近一次扩展仍然成功，也不能
直接支持“继续扩展已经低收益”。

### 2.3 第一个 Direction 出现规格递增

同一个 Direction 的三版 Result 大小约为：

```text
revision 1:  8.5 KB
revision 2: 11.4 KB
revision 3: 14.4 KB
```

第一轮 Reviewer 提出 3 个 BLOCKING finding；修订后 Reviewer 又提出 5 个更细的
BLOCKING finding，最终结果加入了具体哈希字符串、浮点舍入规则、bootstrap
细节和队列窗口算法。

最终 Direction 可证伪，但已经接近实验执行手册。问题不是输出太长本身，而是
Reviewer 没有稳定地区分：

- 会改变科学主张、baseline、pass/fail 或归因的必要定义；
- 可在未来 experiment handoff 中选择、生成或记录的操作细节。

### 2.4 Decision 压缩输入仍未形成稳定读取边界

6 个 Decision Turn 共使用约 229k input tokens。正常 revision 周期中，Decision
会重新读取多个旧 Work/Review Result；第 5 轮还直接深读了 VideoNSA 原始论文
内容。

Decision 必须理解完整 Goal、当前结论和流程错误，但不应重新执行 Worker 的来源
研究或 Reviewer 的证据核验。否则压缩记忆只变成额外输入，而不是旧正文的替代
导航层。

### 2.5 Agent delta 在 runtime 中重复落盘

审计运行的 `runtime.jsonl` 共约 41,491 行，其中：

```text
raw_event:    21,000
output_delta: 19,953
```

每个 `item/agentMessage/delta` 先作为完整 provider `raw_event` 写入，再作为
`output_delta` 写入；同一文本还会增量写入 `partial_output.txt`。其中
`output_delta` 和 partial 文件对崩溃恢复有直接用途，而 raw delta 是重复的
JSONL 存储。

## 3. 不改变的协议和职责

本轮不改变正常 Loop：

```text
Worker → Reviewer → Decision
  ↑                    │
  └──── Script branch ─┘
```

不改变 Agent 输出核心协议：

```text
Worker:   一个 JSON object；Script 只阻断性提取 workOutcome
Reviewer: 一个 JSON object；Script 只阻断性提取 reviewVerdict
Decision: decision = <本次允许字面量>
          guidance = <可选不透明文本>
```

不增加以下 Agent 输出字段：

```text
saturation
globalGap
quietExpansion
selectedSkills
memoryUpdate
trajectoryUpdate
experimentHandoff
```

不允许 Script：

- 根据搜索次数或字符串判断 Topic 已饱和；
- 根据 Result 字节数或数组长度拒绝 Direction；
- 根据 Reviewer finding 数量决定是否继续；
- 解释 Decision guidance；
- 把 `openQueryGaps=[]` 当作全局完成条件；
- 从 runtime 文本推断研究价值。

## 4. P0：真正冻结 Decision 观察输入

### 4.1 目标目录结构

新 Context 创建时，写入以下不可变兄弟文件：

```text
contexts/<decision-context-id>/
  decision_context.json
  decision_observation.json
  research_memory_snapshot.json
  progress_trajectory_snapshot.jsonl
```

`decision_observation.json` 继续只提供 Ref 和少量机械观察：

```json
{
  "generatedAt": "...",
  "stateRevision": 64,
  "round": 6,
  "researchMemoryRef": "contexts/<id>/research_memory_snapshot.json",
  "trajectoryRef": "contexts/<id>/progress_trajectory_snapshot.jsonl",
  "trajectoryTail": [],
  "branchEffects": [],
  "accepted": {
    "anchors": 2,
    "directions": 1
  },
  "remainingRequirements": [],
  "retries": {},
  "recentRuntimeFailures": []
}
```

不为了形式完整继续增加 hash、cursor 或 Agent 需要回显的字段。不可变路径、
`stateRevision` 对齐和 write-once 行为提供本轮所需的一致性。

### 4.2 全局观察文件的定位

保留：

```text
observations/research_memory.json
observations/progress_trajectory.jsonl
```

但重新定义为：

- `research_memory.json`：供 `status`、`checkpoint` 和人类查看的最新投影；
- `progress_trajectory.jsonl`：供状态观察的全局 append-only 轨迹；
- 两者都不能再作为某个历史 Decision Turn 的唯一输入证据。

Decision 只读取本 Context 的 snapshot Ref。`checkpoint` 刷新全局观察时，不得
修改任何 `contexts/<id>/` 下的文件。

### 4.3 快照生成顺序

创建 DecisionContext 时按以下顺序执行：

```text
读取同一个 state revision
→ 从权威对象索引机械构造 research memory value
→ 读取截至当前时刻的完整 trajectory prefix
→ write-once 写入两个 Context snapshot
→ 从同一 snapshot 构造 decision_observation
→ 写入 decision_context
→ 启动 Decision Turn
```

要求：

- 四个 Context 文件只能创建一次；
- runtime retry 和 output correction 复用原 Context，不重新生成 snapshot；
- 若目标文件已存在且内容不同，Script 报持久状态冲突，不能覆盖；
- `generatedAt` 不参与重复构造；已有 Context 必须整体复用；
- Context 创建过程中任一步失败时，不允许启动 Decision Turn。

### 4.4 Validator 的机械检查

Validator 新增的检查仍然完全机械：

1. `researchMemoryRef` 和 `trajectoryRef` 必须解析到当前 Context 目录下；
2. 两个 snapshot 均存在；
3. memory 的 `sourceStateRevision` 必须等于 observation 的 `stateRevision`；
4. `trajectoryTail` 必须等于冻结 trajectory snapshot 的最后至多 5 条；
5. trajectory snapshot 中不能出现当前 Decision 尚未作出的本轮 decision；
6. retry Turn 引用的 DecisionContext 必须与原 Turn 相同；
7. `checkpoint` 前后所有既有 Context snapshot 的字节内容保持不变。

这些检查验证“输入是否真冻结”，不验证其中专业总结是否正确。

## 5. P1：用语义上的安静扩展支撑完成

### 5.1 区分两类缺口

Reviewer 的 `queryGaps` 继续只表示：

> 某个当前 Anchor 或 Direction 的局部未知问题，且回答后可能改变 finding 或
> verdict。

它不表示：

> 整个 Topic 是否还有新的 Anchor、场景、机制或 6L 区域值得探索。

Decision Skill 必须加入明确规则：

```text
openQueryGaps = []
≠ global knowledge gap = none
≠ Topic saturated
≠ FINISH_WORKFLOW justified
```

### 5.2 安静扩展的语义定义

对于范围宽泛、没有显式数量或子空间边界的 Goal，Decision 在结束前通常应看到
一次近期、受限、可审阅的 Anchor 扩展尝试，并由 Worker/Reviewer 结论表明没有
形成新的非重复 Anchor。

这可以完全使用现有协议表达：

```text
机械闭环已满足
→ Decision: RUN_WORKER
→ Script: CREATE_ANCHOR
→ Worker: BLOCKED_NO_RESULT
   unresolved 中说明已完成的受限检索和没有可诚实建立的新 Anchor
→ Reviewer: REJECT
   判定该候选不能成为对象；同时确认这是可信的 no-novel-result，
   或指出搜索不足、仍应继续
→ Decision: 结合 Goal、轨迹、既有覆盖和该负面结论决定 FINISH 或继续
```

这里的 `REJECT` 只表示“不提交这个 Anchor 对象”，不是宣称 workflow 失败。
若 Reviewer 认为 Worker 搜索明显不足或错误缩窄范围，应按语义给出可修复的
`REVISE`，该轮不能作为安静扩展证据。

### 5.3 Decision 的完成规则

Decision Skill 增加以下判断顺序：

1. 先检查 Script 是否允许 `FINISH_WORKFLOW`；
2. 区分 Goal 是显式有界还是开放探索：
   - 显式有界：用户已指定对象数量、技术范围或停止边界，满足该边界即可判断
     完成；
   - 开放探索：不能只凭最小对象数量完成；
3. 若最近一次 `CREATE_ANCHOR` 成功产生新 PASS Anchor，则这通常是“仍有新增
   信息”的证据；完成该 Anchor 的 Direction 后，不应自动推断 Topic 已饱和；
4. 查找最近一次可信的 no-novel-result、重复候选 REJECT 或等价负面探索结论；
5. 结合被拒路线、动态 6L 覆盖、Goal 和最终建议连贯性，判断再扩展一轮的预期
   信息增益；
6. 只有语义闭合时选择 `FINISH_WORKFLOW`。

“通常应看到安静扩展”是 Decision 的方法规则，不是 Script 新增的固定 Gate。
若某个 Goal 明确只要求一个 Direction，Decision 可以依据用户边界完成，而不必
制造无意义的搜索轮次。

`maxRounds` 耗尽仍只触发 PAUSED；不能把没有预算完成安静扩展解释为语义完成。

### 5.4 Worker 和 Reviewer 的配合

Worker 在 `CREATE_ANCHOR` 时：

- 不得为了避免 `BLOCKED_NO_RESULT` 而包装重复 Anchor；
- 在真实、受限检索后没有新对象时，诚实返回 `BLOCKED_NO_RESULT`；
- 使用现有 `unresolved` 简洁说明检索覆盖、主要重复路线和为什么不能形成新对象；
- 不新增 `quietExpansion` 等控制字段。

Reviewer 在审阅该结果时：

- 检查检索是否明显缩窄 Topic、遗漏已有可见候选或把工具失败伪装成无结果；
- 搜索充分但没有对象时，可以用 `REJECT` 保存一个可信的负面结论；
- 搜索方法可修复时使用 `REVISE`；
- 不作出全局完成决定。

## 6. P1：阻止 Direction 规格逐轮膨胀

### 6.1 最小充分表达边界

一个 Direction 必须足以让 Reviewer 判断：

- 唯一主要变化或不可分联合包是什么；
- baseline 和冻结变量是什么；
- 因果机制如何连接目标指标；
- latency、throughput 和质量 guardrail 如何判定；
- 最强失败条件和反例是什么；
- 哪组受控测量能够证伪主张。

以下内容只有在会改变上述判断时才进入当前 Direction：

- 特定统计估计量或置信判据；
- workload 分层或配对原则；
- 会改变公平性的确定性生成规则；
- baseline 可复现所需的关键环境和运行条件。

以下内容默认留给未来 experiment handoff，不要求当前 Direction 展开：

- 完整请求 manifest；
- 每个候选配置或样本 ID；
- 具体 hash 文本格式；
- 与主要结论无关的 binary64 舍入细节；
- bootstrap 每一次抽样的生成公式；
- 逐窗口数据表、trace 或执行脚本；
- 不会改变 pass/fail 的异常处理实现细节。

不增加字符数、数组数量或 measurement-plan 条目数 Gate。最小充分仍是语义判断。

### 6.2 Reviewer 的 BLOCKING 阈值

Reviewer 只在以下情况下把测量定义缺口标为 BLOCKING：

- 两个合理实现会得到不同的主要比较对象；
- 缺口会改变 Direction 的 pass/fail；
- baseline 无法复现；
- 主要变化和 enabler 无法区分；
- 质量或吞吐 guardrail 可以被绕过；
- 因果归因会因缺少控制而失效。

如果某项细节可以在实验执行前机械选择并冻结，且不同选择不改变 Direction 的
科学主张和判定语义，应使用 NON_BLOCKING 建议或不形成 finding，不应要求下一版
把它写成实现手册。

Reviewer 还应：

- 将同一因果缺口下的相关细节合并为一个有界 finding；
- 纠正上一轮 blocker 后，不因表达更具体而无限发现更低层操作细节；
- 只有当前修订引入新矛盾，或旧 blocker 的真正闭合暴露了新的结论级缺陷时，
  才增加新的 BLOCKING finding；
- 对“精简后仍足以复现和证伪”作显式判断。

### 6.3 为修订审阅提供前一轮边界

当 Reviewer 审阅 `DEEPEN_ANCHOR` 或 `DEEPEN_DIRECTION` 产生的新 revision 时，
Reviewer Task 可增加一个可选 Ref：

```json
{
  "inputs": {
    "reviewTarget": "results/current-work.json",
    "boundAnchor": "results/anchor.json",
    "previousReview": "results/previous-review.json"
  }
}
```

约束：

- `previousReview` 只用于确认上一轮 correction boundary；
- 当前 `reviewTarget` 仍是唯一审阅对象；
- Reviewer 不得复制旧 verdict，也不得因旧 finding 存在而自动继续 REVISE；
- Script 只绑定这个 Ref，不判断当前内容是否已解决旧 finding。

这是本轮唯一建议增加的 Task 内容 Ref，不改变 R01 输出格式。

### 6.4 Decision 对 Reviewer 规格膨胀的处理

Decision 读取当前 Work、当前 Review 和可用的 previous Review 后判断：

- Reviewer 新增 blocker 是否会实质影响最终结论；
- verdict 是否与最小充分 rubric 一致；
- 是否把未来 handoff 细节错误提升为当前对象准入条件。

若 Reviewer 明显违反上述边界并会使 workflow 进入错误的无限深化，且
`RETRY_REVIEWER` 在允许集合中，Decision 应使用该分支并在 guidance 中指出：

```text
仅审查会改变主要主张、归因或 pass/fail 的缺口；
未来 manifest 和执行实现细节不得作为新增 BLOCKING 要求。
```

普通、合理的 `REVISE` 仍进入正常 Worker 深化，不被 Script 当成错误。

## 7. P1：收紧 Decision 的读取职责

### 7.1 默认读取阶梯

Decision Skill 将输入读取顺序收紧为：

```text
必读
  Goal
  DecisionContext
  冻结 decision_observation
  冻结 research memory snapshot
  冻结 trajectory snapshot / tail
  pending WorkTask、Work Result、Review Result
  与 pending 对象匹配的 Result Ref

按需读
  memory 中 summaryAvailable=false 的当前对象
  当前 Work/Review 明确冲突所涉及的最新 canonical revision
  FINISH 前仍无法仅靠摘要确认的最终 active Work/Review

默认不读
  已被后续 revision 替代的旧正文
  与本次分支无关的全部 committed Result
  Controller state、events、runtime、turn 或对象索引
  论文、idea note、knowledge note、experiment note 等原始专业来源
```

`committedResults` 仍可保留在 D01 作为权威 Ref 投影，不要求本轮修改消息结构；
“可访问”不等于“每轮全部读取”。

### 7.2 不再独立研究来源

当 Decision 发现：

- Worker 的 evidence 与正文冲突；
- Reviewer 没有核验一个可能改变 verdict 的来源问题；
- Worker 和 Reviewer 对同一事实结论相互矛盾；

Decision 应选择可用的 `RETRY_REVIEWER` 或 `RETRY_WORKER`，把核验目标写入精简
guidance，而不是自行搜索知识库并重新形成专业结论。

Decision 可以检查 Ref 是否存在于 pending/accepted 结论中，但不承担来源深读。
这保证：

```text
Worker   = 内容研究
Reviewer = 独立证据和语义审阅
Decision = 全局需求、流程正确性和下一分支判断
```

### 7.3 FINISH 前的读取范围

FINISH 前不要求回读所有历史 revision。Decision 应：

1. 读取冻结 memory 中当前 active accepted 条目；
2. 读取本轮 pending Work/Review；
3. 只对摘要缺失、摘要冲突或最终主结论不明确的 active latest Result 回读正文；
4. 使用 rejected lessons 和 trajectory 理解已排除路线，不回读每份旧全文；
5. 完成语义判断后只输出 decision 和精简 guidance。

## 8. P2：Runtime Agent Delta 单份落盘

### 8.1 当前重复路径

当前 `runtime.ts` 对任一 `item/agentMessage/delta` 执行：

```text
persist raw_event(message)
→ persist output_delta(itemId, delta)
→ emit live delta
→ Controller append partial_output.txt
```

前两项包含同一 delta 文本。

### 8.2 新的持久化规则

format version 6 writer 对 Agent delta 改为：

```text
收到 item/agentMessage/delta
→ 内存中继续更新 pending message
→ 只写 output_delta {at, itemId, delta}
→ 继续发送 live console event
→ 继续增量 append partial_output.txt
```

不再为该方法额外写 `raw_event`。

其他 provider 事件本轮保持原行为：

- `item/completed`；
- tool start/complete；
- usage；
- turn completed/failed；
- compaction；
- 未有规范化投影的 provider 通知。

本轮不顺带重构所有 runtime event，也不删除 `partial_output.txt`、`output.txt` 或
`message_completed`。

### 8.3 恢复和控制台不变式

必须保持：

- `partial_output.txt` 是所有流式 delta 的按序拼接；
- console 每个 delta 只实时显示一次；
- `rebuildTurnCaptureFromRuntime()` 继续只依赖 `output_delta` 重建 partial；
- `message_completed` 仍决定完整候选消息边界；
- Provider Turn terminal 状态仍独立于 output capture；
- timeout、interrupt 和 runtime retry 行为不变；
- v5 reader 仍能读取同时包含 raw delta 和 output delta 的旧日志。

按审计样本估算，该修改可直接减少约 19,953 条重复 JSONL 记录，但节省量不是
状态机或 Agent 的完成判据。

## 9. 消息与存储契约变化汇总

### 9.1 Agent 输出

无变化：

- W01 核心字段不变；
- R01 核心字段不变；
- Decision line protocol 不变；
- 不增加 Agent 需要精确生成的新 JSON 元素。

### 9.2 Agent 输入

变化：

- D01 仍只有一个 `observationRef`，但其中两个观察 Ref 改为 Context 内不可变
  snapshot；
- 修订对象的 Reviewer Task 可选增加 `inputs.previousReview`；
- Skill 明确读取阶梯、局部 query gap 边界和安静扩展语义。

不变化：

- Goal、Task 和 Ref 仍通过路径提供；
- Script 不内联完整历史或运行日志；
- Decision 允许字面量集合不变。

### 9.3 Script 内部存储

新增：

```text
contexts/<id>/research_memory_snapshot.json
contexts/<id>/progress_trajectory_snapshot.jsonl
```

修改：

```text
contexts/<id>/decision_observation.json
turns/<id>/runtime.jsonl
```

保留：

```text
observations/research_memory.json
observations/progress_trajectory.jsonl
turns/<id>/partial_output.txt
turns/<id>/output.txt
```

## 10. 建议修改的实现位置

### 10.1 `observations.ts`

- 将 research memory 构造与“写最新全局投影”拆成两个函数；
- 创建 DecisionContext 时把同一个 memory value 写入 Context snapshot；
- 把当前 trajectory prefix 写入 Context snapshot；
- `trajectoryTail` 从冻结 snapshot 生成；
- `checkpoint` 只重建全局观察，不接触 Context。

### 10.2 `store.ts`

- 增加只创建或同内容复用的 immutable write helper；
- 已存在且内容不同必须失败，不允许覆盖历史 Context snapshot；
- 保持原子临时文件加 rename 的落盘方式。

### 10.3 `types.ts`

- 将 `DecisionObservation.researchMemoryRef` 从固定全局字面量改为受控相对路径；
- trajectory Ref 同步允许 Context snapshot；
- `TurnTask.inputs` 允许可选 `previousReview`；
- 不改变 Agent output 类型的核心控制字段。

### 10.4 `workflow.ts` / `controller.ts`

- Context 创建流程绑定同一个 state revision 的 snapshot；
- Reviewer Task 在 revision > 1 时绑定 previous Review Ref；
- Decision retry 复用原 Context；
- `allowedDecisions()` 和机械 FINISH 条件保持不变；
- 不新增 quiet-expansion 计数或 saturation Gate。

### 10.5 `validation.ts`

- 增加第 4.4 节的 Context snapshot 一致性检查；
- 对 v6 禁止 Decision observation 指向全局 mutable memory/trajectory；
- 对 v5 保持只读兼容，不原地重解释为真正冻结；
- 不增加 Direction 长度或全局完成语义检查。

### 10.6 `runtime.ts`

- `item/agentMessage/delta` 分支在写 raw event 前分流；
- 该方法只持久化一个规范 `output_delta`；
- 保留内存 pending message、live console、partial append 和完整消息选择逻辑。

### 10.7 三类 Skill 与 Ref

Decision Skill：

- 加入局部 query gap 不等于全局饱和；
- 加入安静扩展和显式有界 Goal 的完成判断；
- 加入默认读取阶梯；
- 禁止独立知识库研究，冲突交给 Worker/Reviewer retry。

Worker Skill：

- 允许受限探索后诚实返回 `BLOCKED_NO_RESULT`；
- 强化最小充分表达和 future handoff 边界；
- 不展开不会改变 Direction 判定的运行细节。

Reviewer Skill / Rubric：

- 明确 BLOCKING 的结论级阈值；
- 读取可选 `previousReview` 并防止 correction scope 递归膨胀；
- 正确审阅 no-novel-result，但不判断全局完成。

现有 work/review JSON 模板不增加新核心字段。

### 10.8 format version 与 README

建议将新运行目录标为 format version 6，原因是：

- Decision observation 的 Ref 语义改变；
- Context 新增不可变 snapshot；
- runtime JSONL 对 Agent delta 的落盘形态改变；
- Reviewer Task 可能新增 previous Review Ref。

旧 v5 工作目录继续只读保留。不得通过 `resume` 把 v5 运行原地切换成 v6；需要
继续研究时初始化新的 v6 work directory。

README 应说明：

- v6 Decision snapshot 真正不可变；
- `checkpoint` 只刷新全局人类观察；
- Agent output wire protocol没有变化；
- runtime delta 去重不影响实时输出和恢复。

## 11. 测试设计

### 11.1 冻结快照测试

1. 第一轮 Context 创建后保存 snapshot 字节；运行多个后续周期和 checkpoint，
   原 snapshot 字节不变；
2. 每个 memory snapshot 的 `sourceStateRevision` 等于 observation revision；
3. `trajectoryTail` 精确等于 snapshot 最后至多五条；
4. 第一轮 snapshot 不包含第二轮及之后结果；
5. Decision runtime retry 和 output correction 使用同一 Context 路径；
6. 尝试用不同内容覆盖 snapshot 时确定性失败；
7. v5 运行仍可 status/validate，但不会被写成 v6。

### 11.2 语义完成场景测试

这些测试验证 Skill/集成行为，不成为 Script 硬 Gate：

1. 宽泛 Goal、最近一次 CREATE_ANCHOR 成功、尚无安静扩展：Decision 不应仅因
   requirements 为空而 FINISH；
2. 宽泛 Goal、受限 CREATE_ANCHOR 返回可信 BLOCKED_NO_RESULT、Reviewer REJECT、
   既有对象闭合：Decision 可以结合全局结论 FINISH；
3. BLOCKED_NO_RESULT 实为工具失败或明显搜索不足：Reviewer 应要求 REVISE，
   Decision 不应把它视作饱和；
4. 用户显式限定只需要一个 Anchor/Direction：满足边界后可以直接 FINISH；
5. `openQueryGaps=[]` 但仍有明显未探索范围：不得单独作为完成依据；
6. maxRounds 到达但语义证据不足：PAUSED，不是 FINISHED。

### 11.3 最小充分表达测试

1. Reviewer 不要求完整 manifest、每个样本 ID 或每次 bootstrap 抽样；
2. 缺少会改变 pass/fail 的 guardrail 时仍产生 BLOCKING finding；
3. 修订结果解决 prior Review 后，Reviewer 不把无关低层实现细节升级成新 blocker；
4. 当前修订引入新的归因矛盾时，Reviewer仍可新增 BLOCKING finding；
5. Decision 能识别 Reviewer 明显违反最小充分 rubric，并在允许时重试 Reviewer；
6. Script 对长但核心合法的 JSON 仍不执行内容长度 Gate。

### 11.4 Decision 读取边界测试

1. 普通 Decision fixture 只需 snapshot 和 pending refs 即可作出分支；
2. memory summary 缺失时可以回读对应 active Result；
3. 证据冲突通过 Reviewer retry 解决，不要求 Decision 直接搜索知识库；
4. FINISH fixture 只回读必要的 active latest Result，不依赖被替代 revision；
5. Decision output 仍只有允许字面量和可选 guidance。

### 11.5 Runtime 去重测试

1. 每个 provider Agent delta 在 v6 `runtime.jsonl` 中恰有一个 `output_delta`；
2. 同一 delta 不再存在对应 `raw_event`；
3. `partial_output.txt` 等于全部 output delta 的顺序拼接；
4. console 每个 delta 只显示一次；
5. message complete、turn complete、timeout 和 usage 不受影响；
6. 中断后可从 output delta 重建 partial capture；
7. v5 含双份 delta 的旧日志仍可读取和验证。

## 12. 实施顺序

建议顺序：

1. 先增加 format v6 类型和旧格式只读分支；
2. 实现 immutable Context snapshot 和 validator；
3. 增加 snapshot/checkpoint/retry 测试；
4. 修改 Decision Skill 的读取阶梯和完成语义；
5. 修改 Worker/Reviewer Skill 与 Reviewer Task 的 `previousReview`；
6. 增加语义场景 fixture；
7. 修改 runtime delta 持久化并验证恢复、console 和旧日志兼容；
8. 更新 README；
9. 用新 work directory 做一次短 smoke run；
10. 再以宽泛 Topic 做一次包含成功扩展和安静扩展的完整运行审计。

不应先修改 Skill 再继续使用 v5 可变 observation；否则新的完成逻辑仍建立在无法
重放的输入上。

## 13. 完成判据

本设计实施完成必须同时满足：

1. 历史 DecisionContext 不再引用可变全局 research memory 或全局 trajectory；
2. checkpoint、status 和后续轮次不能改变旧 Context snapshot；
3. Validator 能发现 observation revision 与 memory snapshot revision 不一致；
4. Agent 输出核心协议没有新增字段；
5. Script 没有新增全局饱和、Direction 长度或 finding 数量 Gate；
6. Decision 明确不把对象局部 `queryGaps=[]` 当成 Topic 饱和；
7. 宽泛 Goal 的完成可以由现有 BLOCKED_NO_RESULT/REJECT 形成安静扩展证据；
8. Reviewer 能区分结论级 blocker 与 future handoff 操作细节；
9. 修订 Review 可以看到前一轮 correction boundary；
10. Decision 默认不读取论文或知识库来源，不反向承担 Worker/Reviewer 职能；
11. v6 Agent delta 在 runtime JSONL 中只落一份；
12. partial capture、实时控制台、完整消息边界、timeout 和 runtime recovery 均不
    回归；
13. v5 目录继续只读可审计，新运行使用 v6；
14. 单元、协议、恢复和端到端测试全部通过。

## 14. 本轮明确不做

以下优化留待后续单独讨论，不在 07 中实施：

- 持久化 model、effort、approvalPolicy 和 sandbox；
- 为最终 manifest 增加 SHA-256；
- 向 Prompt 增加运行目录字段；
- 强制 Worker/Reviewer 使用 Orchestra 专家 Skill；
- 重构 final report 的正文和 appendix；
- 去重 tool、usage、message-completed 等其他 runtime 原始事件；
- 让 Script 根据覆盖层数、搜索次数或无结果次数自动停止；
- 执行任何新实验。

最终关系保持为：

> Script 冻结并保存当时事实，提供可重放的机械输入；Worker 研究内容，Reviewer
> 审阅内容且控制修订边界，Decision 依据真实历史判断下一分支和语义完成；runtime
> 只消除可证明冗余的传输记录，不改变任何研究结论。
