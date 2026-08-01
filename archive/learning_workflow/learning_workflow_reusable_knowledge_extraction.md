# 新版 Learning Workflow：可复用知识提炼

> 归档状态：旧版工作流知识提炼稿，仅供设计追溯。

## 0. 文档边界

本文只提炼旧脚本、旧 skill、人工 `review_draft` 和已验证 `review_notes` 中可复用的专家规则与编排范式。旧实现保持只读；新版不会 import、调用或修改旧的 `learning_scheduler.ts`、`idea_review_orchestrator.ts` 及其配套 skill。

新版的目标不是继续压缩已有 `learning-output`，而是从 topic 出发构造：

```text
可验证的场景与 baseline
  → 每个 Anchor 的 L1–L6 候选修改空间
  → entry-level 跨层关系
  → 可实验的 Direction
  → Direction 级专家深审
  → 实验方向、baseline/reference 与证据缺口
```

## 1. 从旧脚本提炼的编排范式

### 1.1 外层脚本负责确定性，Agent 只负责语义判断

可复用：

- 脚本显式维护 phase、task、round、状态迁移、预算、超时和输出路径。
- Agent 输出必须经过协议解析和结构校验后才能进入下一阶段。
- 每个任务有 `pending → running → done|failed` 状态；中断的 `running` 在恢复时回滚为 `pending`。
- 完成不能只看进程退出码，还要检查可验证的产物和完成条件。
- 原始输入、原始输出、结构化结果和状态 checkpoint 分开保存。

新版强化：

- Agent 不直接写主数据库；只返回结构化 action/result，由脚本集中落盘。
- ID、哈希、引用完整性、同 Anchor 约束、图连通性、baseline 覆盖等由确定性代码检查。
- 校验和接纳以原子对象为单位：错误 entry/edge 被隔离，不连带丢弃同一 Anchor 中已经通过验证的 baseline 和 entry。
- Markdown 只是 JSON/JSONL 对象的渲染视图，不再是唯一事实来源。

### 1.2 高召回阶段使用短会话 worker pool

旧 learning scheduler 的并行 question/answer worker 适合扩展知识空间。新版保留“独立小任务 + worker pool”，但任务单元改为：

```text
DiscoveryTask = Topic × Layer(L1–L6) × ValueAxis
ValueAxis = exploration | implementation_reuse | method_reference
```

每个 worker：

1. 只处理一个粗筛切片；
2. 接收有限的、带行号的证据片段；
3. 输出原子 `EvidenceClaim` 和待补 query；
4. 不总结其他 worker 的输出；
5. 不生成最终方向。

这样可以并行扩展，同时避免让一个 Agent 在巨大上下文中同时完成检索、去重、锚定、跨层推理和价值判断。

### 1.3 深审阶段使用角色隔离的 loop

旧 idea review 中最值得保留的不是特定 marker，而是两类认知角色的隔离：

- `Question/Judge`：看规范化的 Experiment Bundle，负责发现反例、缺口、替代解释和停止条件；不自行检索。
- `Evidence/Answer`：可访问该 Direction 的证据账本，负责回答当前问题、列出证据与 information gaps；不做最终价值裁决。

新版每轮只允许一种动作：

```text
ask | request_evidence | complete
```

编排器保存：

- 当前任务块；
- 当前 round；
- 已问问题；
- 已加载 evidence/reference；
- 已确认与未确认的判断；
- 下一次允许的输入类型。

结构化输出不合格时，只允许一次“协议修复”；修复不得重做检索或改变既有状态。再次失败则保留 checkpoint，任务进入 `failed_retriable`，不能把半成品当作结论。

`complete` 也不能只由 Agent 自报。脚本必须确认场景机会、baseline
公平性、entry、跨层关系、实现复用、实验测量六个维度均留下证据问答；
不适用的维度也要显式判为 `not_applicable`。

### 1.4 专家 reference 按需注入

旧 idea review 的“类别白名单 + 每类最多加载一次”可以直接抽象为：

- reference 不在每轮重复注入；
- Judge 只有发现相关维度后才请求对应 reference；
- 编排器校验 reference key，防止 Agent 自行扩大资料范围；
- reference 只提供评判问题和阈值，不替代被审 Direction 的事实证据。

新版 reference 类别调整为：

1. 场景、瓶颈与可探索性；
2. baseline 与公平对比；
3. L1–L6 修改对象和实现入口；
4. entry-level 跨层接口、协同与冲突；
5. 实验设计、工具、指标与复现边界。

## 2. 从旧 skill 与人工结果提炼的专家知识

### 2.1 L1–L6 是修改空间坐标，不是强制流水线

保留六层语义：

| Layer | 新版中的修改对象 |
|---|---|
| L1 算法/Pipeline | 计算图、负载分解、动态参数、算法近似与可并行性 |
| L2 Serving/Runtime | 请求、batch、stage、资源和执行单元的运行时组织 |
| L3 Compiler | IR、依赖表达、pass、fusion、multiversion、codegen |
| L4 Kernel | tile/warp/instruction pipeline、同步、数据搬运和 kernel 组合 |
| L5 Architecture | 计算/控制单元、存储层次、调度器、NoC 和硬件原语 |
| L6 Chip/System | chiplet、PIM、wafer-scale、封装/互联和芯片级资源边界 |

修改：

- 一层可以有零到多个 entry。
- Direction 可以跨一层、两层或多层；不要求补齐 L1→L6。
- 缺层只有在该 Direction 的因果链需要该层时才是缺口。
- 同一方法通过 `entity_id` 复用，但在不同 Anchor 中可有不同 `LayerEntry`。

### 2.2 证据先于总结

旧 Answer skill 的可复用原则：

- 先把问题拆成对象、场景、机制、约束、实现、实验等逻辑节点。
- 搜索从长语义 query 逐级降级到短语和关键词。
- 保留 query → source → claim 的反向索引。
- 区分直接证据、合理推断和无证据缺口。
- 定量值、baseline 名称、实现入口、适用条件和退化条件不能被摘要省略。

新版将其固化为不可变 `EvidenceClaim`：

```text
claim_id
statement
claim_type
source_path + line_start + line_end + quote
direct|inferred
applicable_scope
```

所有 Anchor、Baseline、LayerEntry、Edge、Direction 和 Judgment 的非 `unknown` 事实字段都必须引用 `claim_id`。后续 Agent 可以重新组织 claim，但不能用新摘要覆盖原 claim。

### 2.3 水平去重和垂直协同的正确保留方式

旧 horizon skill 中可保留：

- 同名/同义实体去重；
- `substitutes`、`complements`、`depends_on` 关系；
- 实现、实验和来源不能因去重被丢掉。

旧 vertical skill 中可保留：

- 跨层接口必须解释数据/控制如何传递；
- 优先构造兼容性明确的组合；
- 缺证据必须显式标注。

不保留：

- 用单层 summary 取代原子答案；
- 只从 horizon summary 推导跨层关系；
- 强制每条路径覆盖 L1–L6；
- 把每层“最像”的方法拼成一条全栈链。

新版用对象图替代两级自然语言总结：

```text
EvidenceClaim → LayerEntry
LayerEntry --CrossLayerEdge--> LayerEntry
兼容 entry 子图 → Direction
```

### 2.4 价值优先级与 baseline 规则

新版价值顺序固定为：

```text
可探索场景 / 潜在加速机会
  > 实现、代码、工具、软件的可复用性
  > 论文方法参考
```

但 baseline 是独立、强制保留的轨道，而不是低探索价值后的垃圾桶：

- `exploration_value=low` 不会删除一个有效 baseline。
- 当前实践、strong baseline、测量工具 baseline、可复用实现 baseline 都进入最终输出。
- baseline 必须参与 Direction 的后续比较、消融与专家判断。
- 只有“证据无效、对象重复且无增量、或与 Anchor 无关”的项才可真正 reject。

每个 Anchor 的 `BaselineSet` 至少尝试覆盖：

```text
B0 current practice
B1 strong comparison
B2 tool/evaluation
B3 reusable implementation
```

不存在的角色写成 gap，不能由 Agent 补造。

### 2.5 专家深审应检查什么

从人工 `review_draft`、idea review references 和已验证 review 输出中可归纳为六组问题：

1. 场景是否具体：workload phase、shape/request regime、backend、bottleneck、metric 是否固定？
2. 加速机会是否真实：独立性、动态性、资源正交性、额外开销和退化边界是否有证据？
3. baseline 是否公平：执行路径、输入、精度、资源预算和指标是否可比？
4. 实现是否落地：modifiable object、代码入口、编译/运行时分工、同步和资源竞争是否明确？
5. 跨层边是否真实：两个 entry 的接口、方向、条件、冲突和协同收益能否单独验证？
6. 实验是否能证伪：单层 baseline、组合 baseline、消融、关键指标、测量粒度和误差是否足够？

AI 给出的“confidence”只作为待检查提示。最终可信度由证据类型、引用可验证性、独立来源和反例检查共同决定。

## 3. 新版对象与视图

事实对象：

- `GlobalEntity`
- `EvidenceClaim`
- `Anchor`
- `Baseline`
- `LayerEntry`
- `CrossLayerEdge`
- `Direction`
- `ExperimentBundle`
- `ExpertReview`

用户视图：

1. `Global Layer Catalog`：按 L1–L6 展示整个 topic 的 entry。
2. `Anchor Layer Map`：每个 Anchor 一张 L1–L6 表，每行一个 entry。
3. `Direction Bundle`：只包含一组兼容 entry、entry-level edge、BaselineSet、证据与实验计划。
4. `Baseline/Reference Registry`：包括低探索价值但有效的 baseline、实现资产、工具和方法参考。

## 4. AI 原生误差的工程约束

| AI 风险 | 新版约束 |
|---|---|
| 长上下文遗漏 | 切成独立 DiscoveryTask；事实写入 append-only claim ledger |
| 自由文本漂移 | API Structured Outputs / JSON Schema；本地二次校验 |
| 把替代方案拼成协同 | Direction 只能选择 entry-level 兼容边；冲突边禁止入同一子图 |
| 引用幻觉 | 精确 path/line/quote 校验；校验失败的 claim 隔离 |
| 高置信错误 | 不把模型 confidence 当最终证据等级 |
| 重试时改变历史 | checkpoint + event log；协议修复不能重做语义工作 |
| 总结损失 | JSONL 原子对象为事实源；Markdown 只渲染并回链 ID |
| prompt 过大 | 只在 task、claim batch、Anchor、Direction 边界切分；不从中间截断 JSON、证据片段或 Experiment Bundle |
| 单个 AI 条目不精确 | 原子接纳与隔离；坏 entry/edge 不污染也不抹除同批有效对象 |
| 过早收敛 | exploration/implementation/method 三轴粗筛；不按论文声望筛选 |
| 无限追问 | 每个 loop 有 round、query、token/费用和 evidence-gap 上限 |

## 5. 新旧策略映射

| 旧策略 | 新版吸收方式 |
|---|---|
| 六层 question agents | L1–L6 × 三价值轴的粗筛 DiscoveryTask |
| answer workers | 有限证据包上的原子 claim extractor |
| horizon summary | `entity_id` 去重 + 保真 claim/entry catalog |
| vertical summary | entry-level edge + Direction 子图 |
| human-draft | 可审计的 Anchor/Direction shortlist 与 baseline registry |
| idea review 双 Agent | Direction 级 Judge/Evidence loop |
| Markdown DONE marker | schema validation + task state + artifact hash |
| session marker/LOOP | 显式 state machine action 与 checkpoint |

## 6. 不从旧实现继承的内容

- 不调用或修改旧脚本、旧 skill。
- 不使用未验证的 `idea_brainstorm` / ideastorm 知识。
- 不沿用旧的 `relevance/reference_value/depth_value` 作为新版最终排序。
- 不让论文方法质量压过可探索场景和实现复用价值。
- 不把 `exploration_value=low` 自动降为仅供丢弃的 `reference_only`。
- 不在任何阶段用自然语言 summary 覆盖原子 evidence。
