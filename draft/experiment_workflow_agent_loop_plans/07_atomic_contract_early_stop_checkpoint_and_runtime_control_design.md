# 原子实验合同、语义早停、检查点与运行控制设计

状态：已实现于 Direction Experiment Loop format v7（2026-08-06）。承接
`06_experiment_decision_fidelity_and_controlled_weakening_design.md`，目标版本为 Direction
Experiment Loop format v7。

本设计来自 format v6 编号 03“分任务视觉 Token 预算开关”的完整运行审计。它不改变
Experiment Decision、Direction Lab Goal、Evidence Judge 三角色主干，也不把科研语义交给
确定性 Script；它修复的是合同粒度、终止条件、阶段结果交接、超时恢复和运行成本。

## 1. 运行事实与问题定位

编号 03 的 Cycle 1 在真实 Qwen2.5-VL-7B、RTX 4090 和真实 benchmark 子集上完成了
`B0 / R100 / M0 / M30 / G0 / M1` 比较。独立 Judge 将其判为
`INCONCLUSIVE + LOCAL_SINGLE_GPU_PERFORMANCE`：

- M1/B0 吞吐比通过，但中位数和 P95 延迟区间均跨越 1；
- OCRBench、VideoMME 的逐任务质量区间没有通过 0.98 下界；
- GPU 时钟未锁定；
- 所谓 E2E 不含媒体预处理；
- requests/s 来自 concurrency=1 closed-loop，只是串行服务率，不是生产并发吞吐。

Cycle 2 使用与 Cycle 1 和确认集互斥的真实样本，完整执行四个 calibration arm，每个
arm 2,200 个请求。机制审计、样本隔离和冻结哈希均有效。校准得到：

```text
pi2(task) = 100% x 11
G0        = 100%
20%/30%   = 没有任何任务满足质量保持率 95% CI 下界 >= 0.98
```

Contract 2 已明确把“`pi2` 退化为任一恒定预算或全部 100%”定义为有效负结果。然而同一
合同又把完整 confirmation 和 performance 产物写成无条件必需项。Lab 因而在终止条件已
命中后继续启动 3,300 请求的 confirmation B0，最终在 B0 完成约 2,361 条时触发 6 小时
hard timeout。Cycle 2 没有生成 `result.md`，Script 没有把已完成校准交给 Judge。

因此根因不是 JSON 无法解析，也不是实验环境失败，而是四个协作缺口：

1. 一个 Cycle 同时承载校准、确认和性能三个决策阶段；
2. 合同的终止条件与完整产物要求互相冲突；
3. Lab 把“Judge 独立裁决”误解成“Lab 不能在预定义边界停止执行”；
4. Controller 只接受 provider `complete`，无法接收 timeout 前已形成的可审阅结果。

## 2. 保留的主干原则

### 2.1 不增加 Agent 类型

继续使用：

```text
Experiment Decision
  -> Direction Lab Goal
  -> Evidence Judge
  -> Experiment Decision
```

- Experiment Decision：理解 Direction、完整轨迹和最终验证需求，冻结下一次原子合同；
- Direction Lab Goal：实现并执行一个原子合同，在预定义边界处停止并保存最窄观察；
- Evidence Judge：独立判断该轮证据的有效性、范围和未决项；
- Script：保存状态、合同、引用和事件，只执行有限状态转换。

不增加 Stage Planner、Progress Agent、Statistic Agent 或实验辅助 Agent。实现、诊断、统计
和局部修复仍由 Lab 在当前合同内完成。

### 2.2 Script 不理解科研语义

Script 不得判断：

- `pi2=100%` 是否构成科学负结果；
- 哪个 baseline 足够强；
- 某个质量区间是否支持或拒绝 Direction；
- 哪些实验臂可以省略；
- 应选择什么模型、数据、软件或硬件；
- 应如何修改 Direction。

Script 只机械校验：

- contract revision/hash 和 cycle 绑定；
- 控制字面量；
- result/checkpoint 引用是否存在；
- 文件是否非空且位于当前 cycle 目录；
- 状态转换是否合法；
- 声明的时间估计是否超过当前运行包络。

## 3. 顶层 Loop 改为原子证据循环

### 3.1 一轮只回答一个会改变下一决策的问题

推荐主流程：

```text
Decision：是否能形成非退化且安全的校准策略？
  -> Lab：只执行 calibration
  -> Judge
  -> Decision

Decision：冻结策略在独立样本上是否通过质量/正确性 guard？
  -> Lab：只执行 confirmation
  -> Judge
  -> Decision

Decision：通过质量门的策略是否有本地性能增量？
  -> Lab：只执行 performance
  -> Judge
  -> Decision
```

并非每个 Direction 都必须经过三个固定阶段。Decision 每次只选择当前最有可能改变最终
判断的一个不确定性；已有证据已经回答时，直接 `RUN_JUDGE`、完成或返回 Learning Flow。

### 3.2 禁止单 Cycle 的超大内部流水线

一个 `RUN_LAB` 合同不得同时要求：

- 大规模重新校准；
- 全量独立质量确认；
- 高功效性能测试；
- 新环境部署或大规模数据获取；
- 多种载体之间的外部有效性验证。

如果一个问题必须依赖前一步输出才能定义下一步输入，它们必须由顶层 Loop 分成多个
Cycle。这样每个 Lab 结果都能独立交 Judge，并让 Decision 在真实证据出现后重新选择。

## 4. Experiment Decision 的修改

### 4.1 新增强制的合同可执行性检查

Decision 在选择 `RUN_LAB` 前必须：

1. 指出本轮唯一决策性不确定性；
2. 列出最早可能决定该问题的证据；
3. 读取 Script 注入的 Lab 运行包络；
4. 使用既往 Cycle 实测速率、已有 pilot 或保守估计计算预计时间；
5. 为环境故障、统计和结果打包保留时间余量；
6. 若无法在一个原子 Cycle 内完成，缩小为更早的证据门，而不是生成超大合同。

没有可用速率时，先生成最小 pilot 或校准合同，不直接生成全量确认合同。

### 4.2 合同最小增量

保持 Decision 外层 JSON 和六个决策字面量不变。仅在 `experimentContract` 中增加两个直接
影响状态机推进的字段：

```json
{
  "objective": "本轮唯一不确定性",
  "comparison": "baseline、variant 和必要消融",
  "conditions": "载体、数据、指标、阶段入口和 guard",
  "stopConditions": [
    "按优先级排列的、可由本轮产物判断的终止条件"
  ],
  "estimatedMinutes": 240,
  "allowedWeakening": [],
  "forbiddenWeakening": [],
  "completionEvidence": "各退出路径分别需要的最小证据"
}
```

不再增加 phase DAG、任意 predicate AST、动态任务图或大型嵌套 Schema。

Script 对新字段只做弱机械校验：

- `stopConditions` 是非空字符串数组；
- `estimatedMinutes` 是正数；
- 估计值加预留不超过当前单次 Lab 运行窗口；
- 不判断 stop condition 的科研含义或是否合理。

### 4.3 终止条件优先级必须无歧义

Decision 形成合同时必须遵循：

```text
terminal stop
  > downstream phase entry condition
  > entered phase required artifacts
  > generic completeness requirement
```

`completionEvidence` 必须按退出路径写条件化产物。例如：

```text
若 calibration 策略退化：保存 calibration raw、机制审计、统计、冻结策略和最窄结果，
立即结束；不要求 confirmation/performance。

若 calibration 非退化：当前合同仍只保存校准结果并结束；下一轮是否 confirmation 由
Judge 后的 Decision 决定。
```

禁止同时写“条件 X 是终止负结果”和“无论 X 是否发生都必须完成下游所有臂”。若输入
Direction 自身存在这种冲突，Decision 必须在合同中消解；不能把冲突留给 Lab。

### 4.4 RUN_JUDGE 的优先级

Decision 的分支顺序调整为：

1. 存在尚未独立审阅的新结果：`RUN_JUDGE`；
2. 最新 Judgment 已经回答决策性问题：完成、拒绝或返回 Learning Flow；
3. 只有缺少新的决定性测量时才 `RUN_LAB`；
4. 不得在公平负结果后通过局部 regime hunting 重复寻找有利区间。

## 5. Direction Lab Goal 的修改

### 5.1 Lab 可以停止执行，但不能决定整个 Direction

Lab 的权限应明确区分：

- 允许：判断“已完成产物是否命中当前合同明文定义的 stop condition”；
- 允许：报告“观察到合同定义的局部正/负/无效/不足边界”；
- 必须：命中后停止下游实验并输出最窄结果；
- 禁止：把局部观察升级为整个 Direction 的 `SUPPORT/REJECT`；
- 禁止：选择下一合同、调整证据范围或修改核心 lever。

Judge 负责判断这个终止观察是否有效，Decision 负责全局完成。

### 5.2 每个昂贵动作前执行 Stop Gate

Lab 的固定方法改为：

```text
绑定合同并生成条件化 checklist
  -> 建立并验证当前阶段 baseline/variant
  -> 执行当前原子阶段
  -> 校验当前阶段产物
  -> 逐条求值 stopConditions
       |- 命中：写 result.md，结束 Goal
       `- 未命中：仅完成当前原子合同要求，写 result.md，结束 Goal
```

因为一轮合同已经原子化，Lab 通常不需要在同一 Goal 内进入新的科学阶段。若合同内部仍有
诊断、安装和分片步骤，每启动新 arm、全量 sweep 或高成本命令前都必须重复 Stop Gate。

### 5.3 合同冲突时停止，不选择更昂贵解释

若 Lab 发现：

- stop condition 与必需产物冲突；
- 一个字段要求停止，另一个字段要求继续；
- 完成合同必须改变 forbidden weakening；
- 剩余时间无法完成最小结果包；

则保存冲突和已完成证据，写出可审阅结果并结束。不得自行选择更昂贵、更完整或更有利的
路线。

### 5.4 检查点与最终结果分离

每个 cycle 使用：

```text
workspace/cycles/<N>/checkpoint.json
workspace/cycles/<N>/result.md
```

`checkpoint.json` 是运行中和恢复用状态，至少包含：

```json
{
  "cycle": 2,
  "contractRevision": 2,
  "contractHash": "...",
  "phase": "CALIBRATION",
  "completedUnits": ["B0", "R100"],
  "validatedArtifacts": ["..."],
  "lastProgressAt": "...",
  "activeCommand": null,
  "resumeAction": "...",
  "partialExcludedRefs": []
}
```

它不承载调度决策，也不要求 Script 理解 `phase` 或产物语义。Lab 原子更新它，恢复时将其
视为当前执行事实；旧对话中冲突状态作废。

`result.md` 只在结果已经可以交 Judge 时写到最终路径。写作期间使用临时文件，完成后原子
提交，避免 Script 把草稿当成完成结果。

最小 `result.md` 包含：

- contract revision/path/hash 和 carrier；
- 当前原子问题及实际执行范围；
- baseline/variant/机制审计状态；
- 完成的 arm、样本和分片数；
- 命中的具体 stop condition，或未命中事实；
- guard、失败和排除的 partial artifacts；
- 最窄 observation 与不能推出的结论；
- code、commands、raw、analysis 和 freeze 引用。

### 5.5 分片执行与恢复

长实验按 task、arm 或固定样本 shard 运行：

- 每个 shard 输出独立 raw、完成标记和 hash；
- 全臂聚合只读取完成 shard；
- timeout 后恢复只补缺失 shard；
- partial shard 保留并标记排除，不伪装为完整结果；
- 不依赖只有进程正常退出时才生成的单一最终评分文件。

### 5.6 降低持久 Goal 的上下文成本

Lab 不应每分钟以 LLM Turn 轮询样本计数。调整为：

- runner 机械记录 heartbeat、完成数和错误；
- 控制台直接转发 runner 输出；
- Lab 只在阶段完成、实质错误、stop gate、检查点和长时间无进展时介入；
- 状态更新以 `checkpoint.json` 为准，不依赖对话记忆；
- 不把完整 raw stdout、累计 diff 或大日志重新读入 Prompt。

## 6. Evidence Judge 的修改

Judge 增加以下明确规则：

1. 首先审查 Lab 声明的 stop condition 是否由绑定正确的产物真实命中；
2. 合同要求命中后停止时，不得因下游阶段未运行而把结果判为“不完整”；
3. 区分“冻结策略在当前规则下退化”和“整个方法族无效”；
4. 校准退化可以是精确策略定义下的有效负结果，但只能覆盖该预算域、数据、模型、阈值和
   统计规则；
5. 安装失败、无效实现和未触发代理仍不能构成科学负结果；
6. 强制报告真实性能测量的实际范围：是否含预处理、并发模型、batch、到达过程、硬件和
   时钟控制；
7. 小样本中零观测差异不能自动解释为总体不确定性为零。

Judge 仍然只输出 `VALID_POSITIVE / VALID_NEGATIVE / INCONCLUSIVE / INVALID`、实际证据
范围、理由和主要未决项，不调度下一轮。

## 7. Controller 的修改

### 7.1 Lab 结果采用机械文件握手

当前逻辑仅在 provider `goalStatus=complete` 时索引 Lab 结果。v7 改为：

```text
Lab invocation 返回
  |- result.md 存在、非空、路径和 contract 绑定通过
  |    -> 无论 provider 是 complete、paused、blocked 或 timeout
  |       都记录原 provider status，索引 result，进入 Judge
  |
  |- 只有 checkpoint.json
  |    -> 保存 pause 原因和 checkpoint ref，保持同 Goal 可恢复
  |
  `- 两者都没有
       -> 异常暂停，保留 runtime/error refs
```

Script 不解析 result 内容，也不依据 checkpoint 的 `phase` 选择科学分支。

### 7.2 细分运行状态和事件

至少记录：

- provider goal status；
- timeout kind；
- invocation ordinal；
- provider thread/turn；
- started/completed/elapsed；
- result/checkpoint/runtime refs；
- 本次是新 Goal 还是同线程 resume。

事件增加：

```text
LAB_GOAL_INVOCATION_STARTED
LAB_GOAL_CHECKPOINT_INDEXED
LAB_GOAL_TIMEOUT
LAB_GOAL_RESUMED
LAB_RESULT_ADOPTED_AFTER_INTERRUPTION
```

`DirectionGoalRecord.attempt` 不能永远为 1；每次 resume invocation 都要有独立审计记录。

### 7.3 向 Decision 和 Lab 注入运行包络

Script 机械提供：

- Lab idle/hard timeout；
- 当前 invocation 的 `deadlineAt`；
- 当前 cycle 和剩余授权 cycles；
- active Goal 已累计时间；
- prior Lab `timeUsedSeconds`；
- 最新 checkpoint/result refs。

Decision 和 Lab 无需回显这些字段。Script 不估算实验成本，只校验 Decision 声明的
`estimatedMinutes` 是否超过包络。

### 7.4 明确 hard timeout 语义

v7 将 Lab hard timeout 定义为单次 provider invocation 的安全 watchdog，不是整个科研
Cycle 的科学预算：

- timeout 必须产生 checkpoint 或 result；
- resume 可以重新获得新的 invocation 窗口；
- 不允许启动预计无法在当前 invocation 内形成 checkpoint 的不可分割命令；
- Cycle 的科研终止由实验合同、Judge 和 Decision 决定，而不是 wall clock 自动决定。

### 7.5 修复实时 pause

当前 Controller 在整个 Lab 执行期间持有 run lock，`pause` CLI 无法取得锁，因而不能实时
暂停。v7 使用锁外的 pause-request 文件或独立控制通道：

1. `pause` 写入不可歧义的请求；
2. 运行时轮询并中断当前 provider turn/Goal；
3. Controller 等待 interrupt grace；
4. 索引 checkpoint/result；
5. 写入 `OPERATOR_REQUESTED` 状态。

同时处理 SIGINT/SIGTERM，优先请求 checkpoint/interrupt，而不是只依赖进程退出。

### 7.6 精简运行日志

- 主 `events.jsonl` 保存规范化的小事件和 refs；
- provider raw stream、stdout 和完整 diff 独立存储，可压缩；
- 不重复保存累计 `turn/diff/updated`；
- 代码变化由 workspace 文件、patch 和 hash 证明，不需要在每个 runtime event 重复全文；
- status/trajectory 直接索引最新 checkpoint 和 invocation 摘要，不要求 Agent 搜索数百 MB 日志。

## 8. 实验方法方面的修正

### 8.1 精确命名性能指标

- 不含媒体预处理时，使用“模型路径延迟”，不宣称完整 request E2E；
- concurrency=1 closed-loop 使用“串行服务率”，不宣称生产服务吞吐；
- 若 Direction 依赖服务吞吐，合同必须定义并执行 open-loop 或足够并发/continuous
  batching 载体，不能用串行 rate 替代；
- evidence scope 和最终 reason 必须保留这些边界。

### 8.2 非劣效与小样本不确定性

Cycle 1 的小校准集中，B0 与候选偶然逐项一致会使普通配对 bootstrap 产生 `[1,1]`。
这只说明观测样本一致，不能说明未观测总体不存在退化。

Decision/Judge 应要求：

- 非劣效样本量或功效依据；
- 对 disagreement/error rate 使用适合有限样本的上界；
- 多任务、多候选选择的保守性；
- 校准与确认严格隔离；
- “没有观测到失败”与“证明满足非劣界”明确区分。

## 9. Cycle 隔离与共享缓存

### 9.1 每个 Cycle 的可变源码隔离

后续 Cycle 不得在共享 `workspace/source/` 中直接修改上一 Cycle 使用过的源码。采用：

- immutable base repository pin；
- `workspace/cycles/<N>/source/` 或 cycle-specific git worktree；
- cycle-specific patch、build 和 config；
- 每轮结果引用自己的 source hash。

前一 Cycle 的 raw、analysis、freeze 和可执行源码保持可复核。

### 9.2 大对象共享只读缓存

模型、数据和基础环境可以放在项目级只读缓存，以 revision 和内容 hash 定位。每个 run
只保存：

- cache ref；
- revision/hash manifest；
- 本轮选择的文件清单；
- 可变 adapter、patch 和配置。

避免 13 个 Direction 分别复制约 16 GB 模型、完整环境和重复数据，同时保持 provenance。

## 10. format v6 到 v7 的迁移

现有 run 在初始化时固定了 Script/Skill 哈希。修改活动 Skill 后，不应直接用新 Skill
恢复旧 v6 persistent Goal，否则会破坏冻结执行绑定。

迁移策略：

1. v6 保留只读审计；
2. 对编号 03，先把 Cycle 2 已完成 calibration 封装为阶段结果，明确 partial confirmation
   不进入证据；
3. 将该结果交独立 Judge，获得正式窄范围判断；
4. 新建 v7 run 时把 v6 的合同、结果和 Judgment 作为历史证据导入；
5. 不继续当前无必要的五臂 confirmation；
6. v7 使用版本化的 Skill/参考文件和新 Controller format。

如果需要旧 run 在 Skill 升级后继续，应在初始化时保存完整 Skill/reference snapshot，并让
该 run 始终使用自己的冻结版本；不能只保存活动全局文件的 hash。

## 11. 实现顺序

### P0

1. 修改 Decision Skill/reference：原子合同、stop conditions、可执行性估计、条件化产物；
2. 修改 Lab Skill/reference：阶段前 Stop Gate、结果/检查点分离、命中即退出；
3. 修改 Judge Skill/reference：正确审阅早停结果；
4. 修改 contract type/protocol/prompt，只增加 `stopConditions` 和 `estimatedMinutes`；
5. 修改 Controller：任意退出状态下采用绑定正确的最终 `result.md` 并进入 Judge；
6. 增加编号 03 的 `all-100% calibration -> result -> Judge` 回归测试。

### P1

1. checkpoint、分片恢复和 invocation 审计；
2. timeout/deadline/runtime envelope 注入；
3. 实时 pause 和信号处理；
4. 规范化进度与 runtime 日志去重；
5. status/trajectory 增加 checkpoint 和 invocation 摘要。

### P2

1. 指标命名与 serving scope 规则；
2. 非劣效、小样本和多重选择审计标准；
3. cycle-specific source worktree；
4. 项目级模型、数据和环境缓存。

## 12. 必须覆盖的测试

### 12.1 原子合同与早停

- Decision 不能生成超过运行包络的合同；
- `stopConditions` 为空时协议拒绝；
- calibration 命中退化策略后，Lab 不启动 confirmation；
- Lab 写入 result 后立即结束当前合同；
- Judge 不因正确跳过下游阶段而判不完整。

### 12.2 Controller 路由

- provider complete + result -> Judge；
- provider timeout + final result -> Judge；
- provider timeout + checkpoint only -> PAUSED/LAB_GOAL；
- provider blocked + final result -> Judge；
- result contract hash 不匹配 -> 不采用并暂停；
- resume 使用同 provider thread，但生成新 invocation record；
- Cycle budget、operator pause 与 Goal pause 可以区分。

### 12.3 恢复和产物

- partial shard 不进入统计；
- resume 不重跑已完成且 hash 通过的 shard；
- 后一 Cycle 修改源码不改变前一 Cycle 的冻结文件；
- status 不读取 provider 大日志也能显示当前 phase、checkpoint 和下一机械动作。

### 12.4 03 回归场景

输入固定结果：

```text
pi2 = 100% x 11
G0  = 100%
mechanism audit = valid
```

预期：

```text
Lab writes result
-> Script indexes result
-> Judge reviews VALID_NEGATIVE/INCONCLUSIVE boundary
-> Decision chooses COMPLETE_REJECT / RETURN_TO_LEARNING / another atomic contract
```

Script 测试只验证路由和绑定，不硬编码 `pi2` 的科学解释。

## 13. 完成标准

本设计完成后应满足：

- 三角色闭环不增加新角色；
- 一次 Lab 只回答一个会改变下一决策的问题；
- 终止条件不会与无条件完整产物要求冲突；
- Lab 命中预定义边界后不会继续昂贵下游实验；
- timeout 前形成的最终结果不会因 provider 状态而丢失；
- checkpoint 可恢复，partial artifacts 不混入正式证据；
- Decision 能看到真实运行包络并避免明显不可完成的合同；
- Script 仍不判断科研语义；
- Judge 保持独立；
- 运行状态不依赖持久对话记忆；
- 日志、token、磁盘和重复实验成本显著下降；
- 编号 03 的失败路径可由自动测试稳定复现并正确闭合。
