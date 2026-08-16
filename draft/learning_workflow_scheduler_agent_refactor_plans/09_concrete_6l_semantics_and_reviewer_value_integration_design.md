# 具体 6L 语义与 Reviewer 价值融入设计

> 状态：formatVersion 8 已实施。第 3–6 节落实具体 6L；第 7–13 节落实
> “性能 baseline → 方法 baseline → Direction 有效性 → 实验参考”的 Reviewer
> 价值链；第 14–18 节落实 Decision 控制的按需 EXP Goal；第 19–23 节补充
> “纸面 gap ≠ 实测 gap”、EXP 信息增益优先、无 token budget、15 分钟无进展
> 超时、FINISHED continuation 和验证结果。Script 仍只校验控制字面量和必要引用，所有价值与
> 实验语义由 Agent 判断。  
> 前置设计：`02_loop_requirement_closure_design.md`、
> `05_minimal_controller_validation_and_semantic_handoff_design.md`、
> `06_outer_loop_memory_trajectory_and_atomic_direction_design.md`、
> `08_official_snapshot_round_lease_convergence_probe_and_delta_batching_design.md`。

## 1. 目的

当前 `learning_6l_v1.md` 能区分 L1–L6，但主要停留在层级名称和对象关键词。
这不足以帮助 Worker 判断“一个真正位于该层的可修改对象应该具体到什么程度”，
也不足以帮助 Reviewer 区分真实覆盖和关键词占位。

旧 `learning-experiment-from-notes-answer` Skill 中存在有用的具体示例，但其旧问答
流程、固定输出篇幅、逐层作答、图示和伪代码要求不属于当前 Learning Loop。
本设计只提取其 6L 认知示例，不迁移旧协议。

旧 `idea_question` Skill 中还包含另一类有用知识：哪些事实可能具有研究价值。
这些问题涉及方法或 baseline 参考、优化机会以及实现或实验环境复用。它们不能
直接变成一张机械打分表，需要先明确这些价值在 Learning Loop 中分别承担什么
作用。

## 2. 保持不变的边界

- Script 仍是不具备语义判断能力的状态机、持久状态库和 Agent 编排器；
- Script 不理解 6L 对象是否具体，也不判断研究价值；
- Worker 形成 Anchor 或 Direction 内容；
- Reviewer 独立判断单个对象是否可进入最终结果；
- Decision 理解最终需求、当前结论和全局轨迹，判断下一条允许分支；
- 不增加新的顶层 Agent 类型；
- 不把旧 Question/Answer Skill 的协议、轮次状态或输出模板迁入当前 Loop；
- 不因为本设计增加 Script 的 JSON 强校验。

## 3. 已确认：6L 是具体性能对象坐标

L1–L6 是“可能被修改并影响性能的对象”所在坐标，不是固定流水线，也不是每个
Anchor 或 Direction 都必须填满的六个盒子。

| 层级 | 具体对象示例 |
|---|---|
| L1 Algorithm/Pipeline | 计算流程、算子序列、输入输出张量、循环或算子维度、数据依赖、可并行子计算、近似或剪枝位置 |
| L2 Serving/Runtime | 请求队列、batch 形成过程、执行时间线、阶段资源映射、placement、cache、prefill/decode 编排、离线与在线决策边界 |
| L3 Compiler | IR 表达、依赖表示、pass 链、融合、代码生成、多版本生成及运行时版本选择 |
| L4 Kernel | grid/block 划分、tile、共享内存、寄存器、warp 职责、指令流水、同步、occupancy、bank conflict、kernel composition 和内存移动 |
| L5 Architecture | HBM/cache/SMEM/RF/计算单元的数据路径、调度与控制模块、DMA/TMA、隔离机制、NoC、带宽和延迟边界 |
| L6 Chip/System | 多 GPU 或多节点拓扑、chiplet、die-to-die、PIM、封装互连、NUMA、容量、功耗、面积、热和系统扩展约束 |

这些内容是非穷举示例，不是必答字段。具体程度以 Reviewer 能独立理解以下内容
为准：

1. 被观察或修改的对象是什么；
2. 它位于哪一层；
3. 它通过什么数据、控制、资源或同步接口影响其他层；
4. 为什么它可能影响 Goal 定义的指标；
5. 如何观察或证伪该影响。

## 4. 已确认：关键词不构成 6L 覆盖

以下表达过于抽象，不能独立证明该层已被探索：

```text
L2: 调度优化
L3: 编译优化
L4: kernel 优化
L5: 硬件加速
```

以下表达才接近可以审阅的具体对象：

```text
L2: 按请求模态和推理阶段维护独立队列，将视觉编码与 LLM prefill
    映射到可独立扩展的资源池。

L3: 在 IR 中保存跨模态算子依赖，离线生成若干融合版本，并由 runtime
    根据实际 token 数选择版本。

L4: 融合 token dispatch、grouped GEMM 和 weighted sum，减少中间张量
    的 HBM 往返与 kernel launch。

L5: 使用独立数据搬运单元与计算单元形成流水，并明确共享 cache 或 NoC
    竞争所限定的有效区间。
```

一个对象不需要包含上述所有细节。只保留与当前性能矛盾、因果机制、层间接口
和可证伪性有关的最小充分内容。

## 5. 已确认：6L 在各角色中的使用方式

### 5.1 Worker

Worker 使用具体示例来发现和表达对象，而不是逐层回答问题：

- 创建 Anchor 时，在 `scope6L` 中写实际性能对象，未涉及层保持 `null`；
- 深化 Anchor 时，补足 Reviewer 指出的对象、接口或边界，而不是扩写所有层；
- 创建 Direction 时，明确主要修改发生在哪个对象，作用如何传播到目标指标；
- 跨层 Direction 必须说明必要的数据、控制、资源或同步接口；
- 不得拼接无关层级方法来制造“全栈”覆盖。

### 5.2 Reviewer

Reviewer 使用同一份 6L 参考判断：

- Anchor 声称的非空层是否包含可识别的性能对象；
- 层级归属是否与对象和证据一致；
- 跨层因果链是否说明必要接口；
- 是否只是层级名、技术名或宽泛关键词占位；
- 是否为了宣称覆盖而静默扩大或缩窄 Topic。

Reviewer 不要求六层齐全，也不因缺少与当前对象无关的底层实现细节而阻塞。

### 5.3 Decision

Decision 不重新构造专业 6L 分析。它只应用全局覆盖原则：

```text
可计入 Topic 动态 6L 空间的区域
= Reviewer PASS 的 active Anchor 中语义具体的 scope6L 区域
```

六个键非空不等于六层已覆盖。若当前对象只给出关键词，而 Reviewer 没有识别，
Decision 应选择允许的 Reviewer 或 Worker 重试/深化分支；它不自行补写对象。

### 5.4 Script

Script 继续机械收集非空 `scope6L` 文本用于观察和索引，不对文本语义作 Gate。
Agent 是否滥用关键词由 Reviewer 和 Decision 处理。

## 6. 已确认：6L 实施时不扩张协议

当前实现更新现有共享 Ref：

```text
.codex/skills/learning-loop-worker/references/learning_6l_v1.md
```

并对 Worker、Reviewer、Decision Skill 做最小方法说明调整。Anchor 仍使用现有
`scope6L.L1` 至 `scope6L.L6`；不新增层级对象数组、接口数组、完整执行图或评分
字段。Script、W01、R01、D01 核心控制协议均无需因此改变。

## 7. 已实施：价值不是三类并列资产，而是一条验证链

Learning Loop 的主要任务始终是发现 Topic 内尚可成立的优化机会。方法 baseline
和实验参考围绕这个主要任务出现，逻辑顺序为：

```text
Worker 发现候选优化机会
  ↓
性能 baseline：当前执行是否确实存在可提升空间？
  ↓
方法 baseline：相同或最近的方法是否已经提出并解决该问题？
  ↓
Reviewer 判断 Direction 是否仍然有效
  ↓
若有效，再要求给出可复用的参考实验与实现环境
  ↓
形成可直接交给后续 EXP Goal 的 Direction
```

因此三类信息的地位不同：

- **优化机会**是被发现和审阅的主要对象；
- **性能 baseline**和**方法 baseline**是判断机会是否成立的必要依据；
- **参考实验/实现环境**是在 Direction 已经成立后补齐的验证 handoff，不用于把
  一个无效方向包装成有效方向。

Learning Flow 不需要为方法参考或环境资产增加新的顶层对象类型。

## 8. 已实施：必须区分两种 baseline

### 8.1 性能或执行 baseline

性能 baseline 回答“当前是否还有可提升空间”：

- 当前系统、模型或方法的实际执行路径是什么；
- 哪个阶段、资源或运行条件限制目标指标；
- 性能症状是平均延迟、尾延迟、吞吐、利用率、通信、容量还是其他 Goal 指标；
- 该症状在什么 workload、模型、硬件、精度、并发和 SLO 条件下出现；
- 当前结果是否已接近物理、算法或服务约束上限。

Anchor 的主要作用是保存“具体场景 + 性能/执行 baseline + 可观察性能矛盾”。如果
没有 baseline 性能事实或至少有界、可验证的 headroom 依据，Reviewer 无法判断
这里是否是真实优化区域。

### 8.2 方法 baseline

方法 baseline 回答“这个解决思路是否已经被提出”：

- 针对同一性能矛盾，最接近的已有方法是什么；
- 已有方法修改了什么对象，适用边界和退化条件是什么；
- 当前 Direction 与它相比真正新增、替换或重新组合了什么；
- 已有方法是否已经在相同条件下解决了该问题；
- 如果已有相似方法，当前 Direction 是复现、迁移、新 regime 下的扩展，还是仍有
  未解决的机制差异。

“已经有人提出”不自动等于无价值。只有在目标、条件、修改对象和预期效果都没有
实质差异时，才是重复 Direction。若当前工作解决不同适用区间、失败条件、层间
接口或硬件边界，仍可能形成有效 Direction，但必须明确差异。

## 9. 已实施：Reviewer 按对象阶段审阅价值

Reviewer 不进行统一的 high/middle/low 打分，而是按 Anchor 和 Direction 的不同
阶段回答不同问题。

### 9.1 REVIEW_ANCHOR

Reviewer 判断：

1. 场景和性能 baseline 是否真实、具体并位于 Topic 内；
2. baseline 性能是否显示与 Goal 相关的剩余优化空间；
3. 性能矛盾是否可观察，且不是只凭宽泛概念推断；
4. 6L 区域是否指向造成或承载该矛盾的具体性能对象；
5. 约束是否足以避免把某个局部现象外推为整个 Topic。

Anchor 阶段不要求完整实验 handoff，也不要求 Worker 已经穷尽所有已有优化方法。
它只需要证明这是值得继续形成 Direction 的真实性能区域。

### 9.2 REVIEW_DIRECTION：先判断方向有效性

Reviewer 先判断：

1. Direction 是否针对 bound Anchor 中已证实的性能矛盾；
2. 修改对象和因果路径是否明确、最小且可证伪；
3. 与最近方法 baseline 相比是否存在实质差异；
4. 如果方法已存在，当前 Direction 是否明确新的适用条件、机制、边界或未解决
   问题，而不是改名重复；
5. 预期效果、guardrail、tradeoff 和失败条件是否与该差异一致。

只有这一步成立，才进入参考实验审查。无效 Direction 不应因为拥有成熟代码、
模拟器或 benchmark 而获得 `PASS`。

### 9.3 REVIEW_DIRECTION：再判断实验 handoff 是否充分

对于已确定有效的 Direction，Reviewer 再检查 Worker 是否给出了围绕当前
baseline 和相近实验的参考：

- baseline 复现所需的来源、配置或公开实现；
- 与主要修改最接近的参考实验；
- 可复用的代码、框架、模拟器、profiler、benchmark、workload 或 trace；
- 参考环境能覆盖的模型、算子、硬件机制和指标；
- 需要移植、扩展、替换或重新校准的部分；
- 与当前 Direction 不一致的假设、硬件或测量边界。

目的不是在 Learning Flow 的 Worker Turn 中实现或执行实验，而是减少后续 EXP Goal
重新搜索 baseline、部署环境和评估方法的开销。

### 9.4 Reviewer 必须使用从 Question Skill 提炼的专家 Ref

只把若干 baseline 和 headroom 检查写进普通 rubric，不足以保留旧
`idea_question` Skill 中“什么内容可能有研究价值”的专家知识。当前实现从该
Skill 及其五类 references 中提炼一个新的、Topic-neutral 的 Reviewer Ref：

```text
.codex/skills/learning-loop-reviewer/references/
  optimization_value_questions_v1.md
```

它不是旧 Question Agent，不保留旧轮次、marker、追问协议、high/middle/low
评分或并发专用准入公式。它只保存 Reviewer 用来发现价值和缺口的专家问题，按
当前 Learning Loop 重组为三组：

| 当前审阅目标 | 从旧 Question Skill 提取的问题知识 |
|---|---|
| 优化机会与性能 baseline | 执行可分解性、动态变化、compute/memory/communication/queue 瓶颈、资源互补或竞争、额外运行时开销、workload/regime 边界、硬限制与可缓解限制 |
| 方法 baseline 与 Direction 差异 | 最接近方法、设计选择、修改粒度、同步和切换开销、静态/动态职责、资源隔离、已有框架实现、相同条件下是否已经解决、可迁移机制与退化条件 |
| 参考实验与环境复用 | 测量粒度、baseline 和指标、模拟器或真实硬件、误差与覆盖范围、代码/框架/benchmark/trace 可复用部分、扩展新架构所需模块 |

Reviewer 的使用顺序为：

1. 先读取 Goal、Task、reviewTarget、bound Anchor 和现有 rubric；
2. 再读取 `optimization_value_questions_v1.md`；
3. 只选择与当前 Anchor/Direction 有关的少量专家问题进行内部检查；
4. 从 W01 和实际证据中寻找答案，而不是把 Ref 的一般知识当作来源证据；
5. 未回答且可能改变 verdict 的问题形成 bounded finding 和 query gap；
6. 不相关问题直接忽略，不要求逐题回答，也不新增价值分数字段。

在三个对象阶段中，该 Ref 的作用不同：

- `REVIEW_ANCHOR`：重点使用优化机会、baseline 性能、瓶颈和适用边界问题；
- Direction 有效性审阅：重点使用最近方法 baseline、设计差异、机制和退化问题；
- Direction 有效后的 handoff 审阅：重点使用参考实验、工具覆盖和环境复用问题。

Reviewer 不向 Worker 发起旧式多轮问答。它把专家问题转化为 R01 finding；若需要
深化，由 Decision 和 Script 通过正常 Loop 再调用 Worker。

### 9.5 已实施：价值问题是标准，Worker 的纸面回答不是实测结论

`optimization_value_questions_v1.md` 是 Reviewer 判断研究价值的专家标准；
Worker 对这些问题的回答主要来自论文、笔记、参考代码和已有运行产物。因此必须
区分两种证据状态：

```text
source-reported gap
  = 论文、笔记或参考实现报告的性能现象
  = 支持“值得验证的候选优化机会”

experiment-observed gap
  = 当前 Loop 已整合的 EXP Goal 在实际环境中得到的观测
  = 支持“该环境和 regime 下确实存在的性能事实”
```

Reviewer 不得把第一种静默提升为第二种。来源报告的延迟、吞吐、利用率、通信或
资源 gap 可以证明一个方向值得投入最小验证，但不能自动证明相同 gap 存在于当前
可用实现、workload、硬件和运行条件中。

当对象的主要价值依赖尚未观察的性能 gap 时，Reviewer 必须判断：是否存在一个
最小判别实验，可以确认、否定或显著收窄 Worker 猜想。优先级依次是 trace、profile、
counter check、microbenchmark、最小复现、单一 ablation；不应先实现完整 Direction。

如果该实验有界且可行，并且结果会改变对象是否准入、主要 claim 或下一步工程投入，
Reviewer 使用现有 R01 表达：

```json
{
  "reviewVerdict": "REVISE",
  "findings": [
    {
      "severity": "BLOCKING",
      "issue": "当前性能 gap 只有来源报告，尚无当前环境观测",
      "basis": "Worker evidence 与当前可用实验状态",
      "expected": "执行一个最小判别实验并由 Worker 整合结果"
    }
  ],
  "queryGaps": [
    {
      "question": "一个能够改变当前 Direction 是否成立的单一经验问题",
      "dimension": "experiment",
      "reason": "观测会改变准入、主要 claim 或下一步投入"
    }
  ]
}
```

该 `REVISE` 不要求 Worker 再搜索论文或伪造测量；它向 Decision 提供清晰的经验
证据缺口。Reviewer 不直接选择 `RUN_EXP_GOAL`。如果实验目前无法有界、环境不可得，
或只能改变次要排序而不影响对象准入和主要 claim，则允许用 `NON_BLOCKING` finding
或 summary 明确保留“纸面候选、尚未实证”的 caveat，并按已有证据判断是否 `PASS`。

## 10. 已实施：参考实验资料补全（不是执行实验）

无需增加新的 Agent 或 Script Stage。现有 `REVISE → DEEPEN_DIRECTION →
REVIEW_DIRECTION` 闭环可以表达“补充参考实验资料”的两阶段过程：

```text
Worker 提出 Direction
  ↓
Reviewer 先审阅机会、性能 baseline、方法 baseline 和因果机制
  ├─ 方向无效且同一对象不可修复 → REJECT
  ├─ 方向本身仍不清楚 → REVISE，Worker 深化方向
  └─ 方向有效，但参考实验 handoff 不足
       → REVISE，finding 明确记录“方向有效，需补实验参考”
       → Worker 只补 baseline/相近实验/环境复用信息
       → Reviewer 再审
            ├─ handoff 充分 → PASS
            └─ 仍有结论级缺口 → REVISE 或 REJECT
```

这样可以避免：

- Worker 在方向尚未成立前生成过长实验手册；
- Reviewer 用实现环境成熟度替代方向有效性；
- 新增一个 Experiment-Reference Agent 或新的调度分支；
- Script 理解“方向已经有效但 handoff 未完成”这类语义状态。

该中间语义由 Reviewer 的 `summary` 和 findings 保存，Script 仍只读取
`PASS | REVISE | REJECT`。

本节描述的是知识检索和 handoff 完善，不是执行新的性能实验。实际执行实验时，
使用第 14–17 节定义的独立 EXP Goal 分支。

## 11. 已实施：Worker 的搜索与输出职责

### 11.1 Anchor Worker

Worker 优先搜索并形成：

- 当前执行 baseline；
- baseline 性能和剩余 headroom；
- 可观察性能矛盾及具体 6L 对象；
- 适用条件和反例。

### 11.2 首次 Direction Worker

Worker 围绕已通过 Anchor：

- 提出一个最小可证伪修改；
- 搜索最接近的方法 baseline；
- 说明已有方法是否覆盖该修改和 regime；
- 明确当前 Direction 相对已有方法的真实差异；
- 给出足以判断有效性的机制、效果、tradeoff 和失败条件。

Worker 可以保留检索过程中自然发现的参考实验，但第一次 Direction 不应为了
预防所有未来环境问题而无限扩写。

### 11.3 Direction 有效后的深化 Worker

当 Reviewer 已确认方向有效而要求实验 handoff 时，Worker 才集中检索：

- `experiment`：baseline 和相近方法的实验设计、配置、指标、环境和 trace；
- `knowledge`：环境依赖的实现事实、硬件机制和兼容边界；
- `idea`：必要的替代 baseline 或反例实验；
- `human`：本地证据不能决定的资源、授权或工程偏好。

它返回完整修订后的 Direction，而不是单独的环境报告；不执行新实验。

## 12. 已实施：优先复用现有 JSON

当前实现不增加 `valueType`、`valueScore` 或新的内容 Agent 消息类型。现有字段可
承载这条证据链：

| 信息 | 现有承载位置 |
|---|---|
| 性能/执行 baseline | Anchor `baseline`、`performanceTension`、`evidence` |
| 方法 baseline 及真实差异 | Direction `baselineChange`、`mechanism`、`tradeoffs`、`evidence` |
| 参考实验和环境复用 | Direction `measurementPlan`、`tradeoffs`、`evidence`，以及 bound Anchor `constraints` |
| 未满足且会影响准入的缺口 | Reviewer `findings`、`queryGaps` |
| “方向有效但 handoff 未完成” | Reviewer `summary` + `REVISE` finding |

Reviewer summary 应优先写清楚：

```text
性能机会是否成立；
与最近方法 baseline 相比是否仍有实质 Direction；
若 Direction 有效，参考实验 handoff 是否足以交给 EXP Goal。
```

Research Memory 已索引 Review summary，因此 Decision 可以先通过摘要理解对象处于
“机会待证实”“方法重复”“方向有效待补实验参考”或“完整 PASS”中的哪一种语义
状态，再按需读取完整 W01/R01。

如果实际运行证明 `baselineChange` 无法同时清晰表达 bound Anchor baseline 和
最近方法 baseline，再单独讨论增加一个简单的 `methodBaseline` 字段。当前不先
扩张格式。

## 13. 已实施：Decision 和最终闭环

Decision 仍负责全局边际价值，而不是重复 Reviewer 的局部技术审查：

- Anchor 被 Reviewer 证实后，决定是形成 Direction 还是继续其他允许分支；
- Direction 被判定为方法重复时，决定深化真实差异或换一个方向；
- Direction 有效但实验 handoff 不足时，选择正常 Worker 深化路径；
- 当已有结论留下一个会改变 Workflow 判断的经验问题时，可以选择启动一次
  EXP Goal，并在 guidance 中解释实验目标；
- Direction PASS 后，结合全部 Portfolio 判断是否还需新 Anchor、另一 Direction、
  新角度 Reviewer 或完成；
- 最终 Direction 应带有足够的 baseline 和参考实验信息，使报告能够降低后续
  EXP Goal 的搜索成本。

最终报告仍以 Anchor → Direction 为主结构，不新增独立“环境价值对象”。参考环境
必须绑定到一个有效 Direction，说明它用于复现哪个 baseline、验证哪个主要修改，
以及需要怎样适配。与任何有效 Direction 都无法建立关系的环境资产不属于本次
Learning Flow 的最终结论。

当前采用以下两个具体边界：

1. `PASS` 要求 handoff 足以定位最接近的 baseline/reference experiment、覆盖边界
   和必要适配；不要求在 Learning Turn 内形成精确可执行实现；
2. 有界检索确认不存在直接可复用环境时，允许明确记录“EXP Goal 从最近 baseline
   自建”后 `PASS`，避免外部资产客观不存在导致无限 `REVISE`。

## 14. 已明确：实验执行就是一个 EXP Goal

Learning Flow 在某个关键性能事实无法由已有证据确定时，可以启动一个独立
EXP Goal。它是一个 Goal-style Agent 的完整生命周期，可以在同一个 Goal 中连续
尝试环境、修改代码、运行实验、阅读结果和继续诊断；它不是单 Turn，但也不拥有
另一套 Worker、Reviewer、Decision、Controller 或状态机。

顶层关系只有：

```text
Learning Decision → EXP Goal → Learning Decision
```

Script 启动 EXP Goal 时注入当前对象和明确经验问题：

```text
EXP Goal Prompt
  ├─ current Anchor：当前候选或 active Anchor
  ├─ current Direction：当前 Direction；Anchor 诊断时允许为空
  └─ experiment objective：Decision guidance 中的有界实验目标
```

Decision 不需要在 guidance 中设计具体模拟器、代码、命令、完整实验步骤或最终
环境。它只需要解释：要通过实际观察回答什么，以及这个结果用于判断当前 Anchor
或 Direction 的什么问题。

资源、权限、超时和轮次预算仍由 Script 作为 Controller 运行配置管理，不要求
Decision 在 experiment objective 中重复，也不与上述三个主要语义输入混合。

具体模拟器、代码库、容器、实现路径和完整测量过程不在启动前一次冻结；这些由
EXP Goal Agent 根据真实错误和数据逐步尝试、修订并保存在自己的工作目录中。

本设计不调用 `draft/experiment_workflow_agent_loop_plans/01_experiment_flow_core_strategy.md`
中定义的第二套多 Agent Flow。该文档后续应单独修订或归档，不参与本设计实施。

## 15. 已明确：只保留两个实验触发位置

### 15.1 候选 Anchor 的 baseline/headroom 缺口

当 Worker 找到候选性能矛盾，但本地来源和已有运行产物不能确定当前算法
baseline、baseline 实现、baseline 系统或 baseline 硬件是否仍有优化空间时，
Reviewer 可以提出阻塞性的 `experiment` query gap；该 gap 是 Decision 的判断
依据，不直接启动实验。

典型问题包括：

- 实际瓶颈是否位于声称的阶段或资源；
- baseline 在目标 workload 下是否已经饱和；
- 延迟、吞吐、通信、容量或利用率 headroom 是否真实存在；
- 来源中的结果是否能迁移到当前实现、系统或硬件条件。

此时优先启动最小诊断实验，如 profiling、负载 sweep、trace、roofline、微基准或
模拟器诊断。结果回流后由 Learning Worker 修订候选 Anchor，再由 Reviewer 判断
该性能区域能否成立。

### 15.2 方法 baseline 已检查后的 Direction 机制缺口

Worker 已经查询最近方法 baseline，并说明当前 Direction 不是无差别重复后，如果
仍有一个必须通过实际观察才能回答的机制事实，Reviewer 可以提出阻塞性的
`experiment` query gap；是否实际启动仍由 Decision 决定。

典型问题包括：

- 声称可重叠的两段执行是否在目标硬件上真正并行；
- 资源竞争是否会抵消理论收益；
- 某个 runtime 信号能否稳定预测应选择的实现；
- 模拟器是否覆盖决定结论的硬件机制；
- 最小修改能否在不改变 guardrail 的条件下产生可区分效果。

此时优先执行最小 ablation、机制微基准、模拟器敏感性实验或最小原型，不直接
扩张为完整系统实现。

### 15.3 明确移除的触发点

`FINISH_WORKFLOW`、即将完成 Learning Flow、Round 即将耗尽或最终报告即将渲染，
都不是实验触发条件。Learning Flow 完成前不设置统一的确认性实验阶段，也不因
完成候选已满足而自动启动端到端 A/B 实验。

参考实验 handoff 不足只触发普通 `DEEPEN_DIRECTION` 知识补充；除非它同时暴露
第 15.1 或 15.2 节所述的决策关键经验缺口，否则不启动实际 EXP Goal。

## 16. 已明确：EXP Goal 选项和目标由 Decision 决定

Script 在进入 Learning Decision 时，可以把 `RUN_EXP_GOAL` 作为本次允许
决策之一。Decision 结合 Goal、当前 Anchor/Direction、全部 Agent 结论、已有实验
结果和运行状态，决定是否选择它。

Script 只用机械条件决定能否提供该字面量，例如：存在可注入的当前或 pending
Anchor、当前没有正在运行的子 Goal、且仍有用户授权的实验预算。Script 不要求先
出现某个 Reviewer query gap，也不判断实验是否有研究价值。

一次实验启动建议满足：

1. 当前存在会改变 Anchor/Direction 或 Workflow 判断的经验问题；
2. 该问题不能由已有论文、笔记、日志或已保存实验产物可靠回答；
3. Decision 能用简短 guidance 把它表达成一个有界实验目标；
4. 当前 Anchor 以及可选 Direction 提供了足以开始迭代的 baseline 上下文；
5. 实验结果会影响下一步选择，而不只是补充无关信息；
6. 已有必要资源、时间和权限授权；
7. 目标允许 EXP Goal 优先寻找能够区分当前假设的最低成本实验。

角色分工为：

```text
Learning Reviewer
  → 审阅 Anchor/Direction，指出性能事实和经验缺口
  → 不启动 EXP Goal

Learning Decision
  → 结合 Goal、全局价值、已有结论和本次允许项决定是否实验
  → 选择 RUN_EXP_GOAL 时必须在 guidance 中解释实验目标

Learning Script
  → 接受 Decision 字面量和原始 guidance
  → 冻结父状态
  → 将 Anchor、可选 Direction 和 experiment objective 注入 EXP Goal Prompt
  → 直接启动 EXP Goal Agent
  → 不理解或改写实验目标
```

Reviewer 的 `queryGap.dimension=experiment` 是重要输入，但不是选择实验的必要
协议条件：Decision 也可能从 Worker 结论、已有实验冲突或全局 Workflow 状态中
识别需要经验判断的问题。Script 不从 query gap 自动生成目标，也不校验 guidance
的专业充分性。

`RUN_EXP_GOAL` 的 guidance 与普通 guidance 不同：它是启动 EXP Goal 所需的
`experiment objective`，因此该分支下必须非空。Script 只检查存在性并原样保存、
注入，不解析其中的机制、指标或方法。

以下情况不启动实验：

- 对于 Direction 机制实验，尚未查清最近方法 baseline；
- 只是缺少参考论文、开源代码或环境说明；
- 只是开放式 Topic 扩展或关键词覆盖不足；
- 没有明确 baseline、经验问题或结果的决策用途；
- 实验无论得到什么结果都不会改变当前 Learning 分支；
- 仅因为 Learning Flow 即将完成或预算即将用尽。

## 17. 已实施：实验结果回流

顶层转换建议为：

```text
Learning Worker → Learning Reviewer → Learning Decision
  → Learning Decision: RUN_EXP_GOAL
       guidance = 有界 experiment objective
  → Script 提交当前 pending 结论并冻结父对象 revision 和父状态
  → Script 注入 Anchor + optional Direction + experiment objective
  → EXP Goal Agent 在一个 Goal 生命周期内迭代环境、代码、测量和诊断
  → 返回不可变 EXP Goal Result
  → Script 保存结果引用并更新 Learning observation
  → 新的 Learning Decision
       ├─ RUN_WORKER：让 Worker 整合结果或修订对象
       ├─ RUN_EXP_GOAL：以新的有界目标启动另一个 EXP Goal
       └─ 其他本次允许分支
```

因此 EXP Goal 是 Learning Loop 中 Decision 之后的可选环节：

```text
... → Decision → [optional EXP Goal] → Decision → ...
```

EXP Goal 完成后不自动进入 Worker，也不自动选择 Reviewer。新的 Decision 读取
实验结论和父 Workflow 状态，决定下一步。若需要把数据写入 Anchor/Direction，Decision
选择 `RUN_WORKER`，之后仍按正常 Worker → Reviewer → Decision 闭环处理。

EXP Goal Result 可以是：

- 当前条件下支持该性能机会或机制；
- 当前条件下不支持；
- 只在更窄条件下支持；
- 环境、实现或数据不足，无法得到可信结论；
- 暴露出新的 baseline 瓶颈或对象定义问题。

任何结果都只作为与父对象 revision 绑定的证据返回。EXP Goal 不静默修改 Anchor、
Direction 或 Topic；Learning Decision 决定是否调用 Worker 解释并形成新 revision。
新发现若构成另一个 Anchor，也必须回到正常 Learning Loop 中形成。

若子 Goal 暂停、需要新增资源授权或无法继续，父 Learning Flow 同步保持暂停；
Script 不把子 Goal 的负结果、不确定结果或环境失败解释成自动 `PASS`、`REJECT`
或 workflow completion。

## 18. 实现映射

- 具体 6L：`.codex/skills/learning-loop-worker/references/learning_6l_v1.md`；
- Reviewer 价值专家知识：
  `.codex/skills/learning-loop-reviewer/references/optimization_value_questions_v1.md`；
- Learning roles：`learning-loop-worker`、`learning-loop-reviewer`、
  `learning-loop-decision`；
- 持久实验 Goal：`.codex/skills/learning-exp-goal/`；
- Controller、状态与 Goal transport：`scripts/simple_semantic_loop/refactor/`；
- `RUN_EXP_GOAL` 只在 `--max-exp-goals` 大于零且存在可冻结 Anchor 时由 Script
  提供；Decision guidance 必须非空；
- EXP Goal 结果保存到 `experiments/<id>/` 并回到新的 Decision。只有 Decision
  随后选择 `RUN_WORKER`，Script 才把尚未整合的结果机械绑定到原对象深化 Task；
- Agent 内容 JSON 未增加 6L 价值分数、baseline 类型或实验结果回显字段；
  `experimentResults` 是 Script 生成的 T01 输入引用，实验记录是 Controller 内部
  JSON。

## 19. 已实施：EXP 是信息增益分支，不再是异常兜底

此前实现虽然向 Decision 提供 `RUN_EXP_GOAL`，但 Decision Skill 把它描述成仅在
现有证据完全不能回答时使用的稀有分支，Reviewer 又可能让“纸面完整但未实测”的
Direction 直接 `PASS`。实际 v8 运行因此形成 14 个已通过 Direction，却没有启动
任何 EXP Goal。

当前策略改为比较下一次纸面扩展与最小经验判别的边际信息量：

```text
Reviewer 识别来源报告的性能 gap
  ↓
若最小实验会改变 Direction 是否成立
  → REVISE + BLOCKING finding + experiment queryGap
  ↓
Decision 读取当前 R01、对象、Goal、全局轨迹和允许字面量
  ↓
比较：继续 Worker/Reviewer 纸面深化 vs RUN_EXP_GOAL
  ↓
优先选择能够低成本确认、否定或收窄主要假设的分支
```

Decision Skill 的具体规则为：

- Reviewer `REVISE` 且存在 verdict-changing `experiment` query gap，是选择
  `RUN_EXP_GOAL` 的最清晰信号；
- Saved evidence 无法回答、最接近方法 baseline 已检查、且 trace/profile/
  microbenchmark/最小复现/ablation 能区分假设时，优先 EXP；
- 优先 cheap go/no-go observation，不提前要求完整系统实现；
- EXP 不是临近完成时的仪式，也不因为预算尚有余额就自动执行；
- `FINISH_WORKFLOW` 前，Decision 必须显式审计：高价值 Direction 是否仍依赖一个
  可被低成本判别的 source-reported gap；
- Script 不解析 R01 的实验语义，也不机械强制每个 Direction 运行实验。

这仍符合最小 Controller 原则：Script 只决定 `RUN_EXP_GOAL` 是否机械可用，Reviewer
提供对象局部经验缺口，Decision 做全局语义取舍。

新运行默认 `maxExperimentGoals=5`，因此未显式传 `--max-exp-goals` 时，Decision
仍能看到实验分支。用户可传 `--max-exp-goals 0` 完全关闭 EXP。

## 20. 已实施：EXP 无 token budget，使用独立的 15 分钟无进展超时

原初始化逻辑把通用 `--idle-timeout-ms`、`--hard-timeout-ms` 和
`--interrupt-grace-ms` 应用到全部角色，导致示例中的 5 分钟 idle、15 分钟 hard
同时覆盖 EXP Goal。这样表达的是“实验总共最多 15 分钟”，不是“15 分钟没有进展
才暂停”。

当前 timeout profile 分离为：

| 角色 | 默认 idle timeout | 默认 hard cap | 说明 |
|---|---:|---:|---|
| Decision | 360000 ms | 900000 ms | fresh control Turn |
| Worker | 300000 ms | 900000 ms | fresh content Turn |
| Reviewer | 300000 ms | 900000 ms | fresh review Turn |
| EXP Goal | 900000 ms | 21600000 ms | 15 分钟无进展；总 hard cap 6 小时 |

CLI 参数边界为：

```text
--idle-timeout-ms
--hard-timeout-ms
--interrupt-grace-ms
  → 只覆盖 Decision / Worker / Reviewer

--exp-idle-timeout-ms
--exp-hard-timeout-ms
--exp-interrupt-grace-ms
  → 只覆盖 EXP Goal
```

EXP idle timer 由有意义的 Agent 消息、tool activity、Goal status 或 usage activity
重置；这些活动不能延长 hard cap。十五分钟无活动时，Runtime interrupt 当前 Turn、
把 Goal 置为 `paused`、保存已有输出和 workspace，然后让父 Learning Flow 暂停。

Controller 创建 Codex Goal 时明确设置 `tokenBudget: null`。单个 EXP Goal 不再因
累计 token 数达到阈值而在环境部署、实现或配对测量中途返回 `budgetLimited`。
实验的机械边界改为三层：一次 Learning Run 最多启动五个 EXP Goal、单个 Goal
连续 15 分钟没有真实进展则暂停、单个 Goal 总运行时间最多 6 小时。Provider 自身
的 usage limit、用户暂停、需要新增授权或真实 blocked 状态仍可终止或暂停 Goal。

CLI 不再提供 `--exp-goal-token-budget`。新 Run 的
`experimentGoalTokenBudget` 固定记录为 `null`；旧 format version 8 Run 中已经存在
的正整数仅保留为历史配置，Controller 不再把它发送给 Codex，因此旧暂停 Run 恢复
后启动的新 EXP 也没有 token budget。

EXP Skill 同时要求：长操作前把命令和 checkpoint 写入 `workspaceRef`，把不透明的
长任务拆成可观察的有界步骤，或让真实进展进入日志；不得用空 heartbeat 掩盖停滞。

## 21. 已实施：从 FINISHED 正式结果创建 continuation

已经 `FINISHED` 的正式目录保持不可变，不能用普通 `resume` 复活。新增
`continue` 命令创建一个新的审计分支：

```text
FINISHED source run
  ↓ copy complete history and canonical objects
new continuation workDir
  ├─ new runId and current Skill/Ref pins
  ├─ source run/state/final immutable snapshots and SHA-256
  ├─ copied Tasks, Turns, Results, Objects, observations and events
  ├─ lifecycle = RUNNING
  ├─ next Round = source round + 1
  └─ sequence = [DECISION]
```

源运行的 `run.json`、`state.json` 和 `final/` 被复制到：

```text
continuation/<source-run-id>/
  ├─ source.json
  ├─ run.json
  ├─ state.json
  └─ final/
```

`source.json` 和新 `run.json.continuation` 保存 source run ID、源目录、源 state
revision、继续时间、三个 SHA-256 和快照引用。Validator 检查快照存在、哈希一致、
源 state 确实为 `FINISHED`，并检查 source record 与新 Run metadata 一致。

Continuation 从 Decision 开始而不是重新运行 Worker。Decision 读取已复制的完整
Portfolio、轨迹和当前实验上下文，选择 EXP、Worker、Reviewer 或再次完成。默认
实验上下文是最新的未拒绝 Direction；当前多模态 v8 正式结果默认指向最新的 PPL
response-gate Direction。

如果源运行已有 EXP，`--max-exp-goals` 表示 continuation 中包含历史记录后的总
上限；不能小于已复制 EXP 记录数。源运行没有 EXP 时，`--max-exp-goals 5` 提供五次
新实验机会。

## 22. 当前正式结果的继续命令

```bash
node scripts/simple_semantic_loop.ts continue \
  --from-work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v8_20260802 \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v8_exp_continue_20260803 \
  --max-rounds 6 \
  --max-exp-goals 5 \
  --idle-timeout-ms 300000 \
  --hard-timeout-ms 900000 \
  --exp-idle-timeout-ms 900000 \
  --exp-hard-timeout-ms 21600000

node scripts/simple_semantic_loop.ts validate \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v8_exp_continue_20260803

node scripts/simple_semantic_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v8_exp_continue_20260803
```

第一条命令只建立分支和快照，不启动 Agent；第二条验证机械一致性；第三条才启动
新的 Decision Turn。源正式目录不会被写入。

## 23. 实施与验证结果

本轮修改涉及：

- `learning-loop-reviewer`：纸面/实测 gap 区分、最小判别实验 finding/query gap；
- `optimization_value_questions_v1.md`：将 gap 证据状态加入价值问题；
- `learning-loop-decision`：把 EXP 提升为信息增益分支并增加完成前审计；
- `learning-exp-goal`：说明 15 分钟无进展行为与可观察 checkpoint；
- `run_setup.ts` 和 CLI：EXP `tokenBudget: null`、独立 timeout、默认最多五次 EXP、`continue`；
- `types.ts`、`validation.ts`：continuation provenance 和哈希校验；
- README 与 E2E tests：继续命令、timeout 边界和源不可变性。

验证结果：

- Simple Semantic Loop 测试 `48/48` 通过；
- Decision、Reviewer、EXP Goal 三个 Skill 均通过 `quick_validate.py`；
- `doctor --no-provider` 通过；
- 用真实 `multimodal_inference_latency_first_v8_20260802` 建立临时 continuation，
  新分支 `validate` 全部通过，允许决策含 `RUN_EXP_GOAL`，默认目标为最新 Direction；
- 测试 continuation 已清理，正式源目录未修改；
- FINISHED 历史运行遇到项目 Skill 后续升级时，pin drift 作为审计 advisory 报告，
  不再把已经完成的历史结果误判为结构无效；RUNNING/PAUSED/FAILED 运行仍严格检查
  当前 Skill/Ref pin，防止活动运行在未记录的实现漂移下继续。
