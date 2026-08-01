> 归档状态：旧版 Learning Workflow 讨论稿，仅供设计追溯。

是的，而且需要再区分“锚点的问题空间”和“最终方向”。准确的数据层级应是：

```text
Topic
├── Anchor A1：场景 S1 × Baseline Set B1
│   ├── L1: [E1, E2, E3]
│   ├── L2: [E4, E5]
│   ├── L3: [E6, E7, E8]
│   ├── ...
│   ├── Direction D1: E2 → E5 → E7
│   └── Direction D2: E1 → E6 → E10
│
└── Anchor A2：场景 S2 × Baseline Set B2
    ├── L1: [...]
    └── ...
```

所以：

- 一个 topic 有多个 `Anchor`；
- 每个 Anchor 有一张 L1–L6 Intervention Map；
- 每层是零到多个 `LayerEntry` 的集合；
- 一个 Anchor 可以进一步形成多个探索方向；
- 一个具体方向只选择其中一部分 entry 和跨层关系。

## 1. 需要区分三种视图

### 全局 Layer Catalog

这里按 L1–L6 查看整个 topic 的所有条目，不同场景、baseline、方向可以共存：

```text
L4 Catalog
├── E041：Anchor A1，MoE grouped GEMM persistent kernel
├── E042：Anchor A1，MoE multi-stream baseline
├── E043：Anchor A2，DiT attention/MLP overlap
└── E044：Anchor A3，Video-LLM dynamic-shape kernel
```

它回答：“整个 topic 在 L4 有哪些已知修改点、baseline 和实现资产？”

### Anchor Layer Map

只包含与同一个“场景 × baseline”相关的条目。每层仍然可以有多个 entry：

| Layer | Entry | 角色 | 内容 |
|---|---|---|---|
| L3 | E31 | baseline behavior | 当前编译器为动态 shape 生成通用 kernel |
| L3 | E32 | opportunity | shape 分布集中，可生成少量特化版本 |
| L3 | E33 | method reference | profile-guided multiversioning |
| L3 | E34 | implementation asset | TorchInductor shape guard/codegen |
| L3 | E35 | constraint | code cache 与编译时间可能增长 |

它回答：“针对这个场景和 baseline，各层有哪些可能修改的对象？”

### Direction Bundle

从 Anchor Layer Map 中选出一组兼容 entry，构成一个具体方向：

```text
D1:
  E12(L1: 动态负载特征)
    → E24(L2: runtime shape bucket)
    → E32(L3: kernel multiversion)
    → E47(L4: 两类 tile 配置)
```

它回答：“哪些 entry 可以组成一个可实验验证的方向？”

这三者不能混为一体。尤其不能把 Anchor 中所有 entry 都当成同一个方向，否则 AI 很容易把相互替代、甚至互不兼容的方法拼在一起。

## 2. Anchor 应该比“加速场景 + baseline”定义得更精确

推荐定义为：

```text
Anchor =
  Workload Phase
  × Request/Shape Regime
  × Backend Context
  × Target Bottleneck
  × Primary Baseline
  × Target Metric
```

例如：

```yaml
anchor_id: A017
workload: Mixtral-8x7B
phase: single-request decode
backend: H100
regime: small-token expert imbalance
bottleneck: dispatch + grouped-GEMM tail wave
primary_baseline: default sequential dispatch/compute
target_metrics:
  - layer_latency
  - SM_utilization
  - HBM_traffic
```

判断是否应拆成不同 Anchor 的规则是：

- workload 阶段改变；
- 请求模式或 shape regime 改变；
- 主要瓶颈改变；
- baseline 执行路径改变；
- 后端能力边界改变；
- 目标指标改变到无法公平比较。

只要这些发生实质变化，就应该拆成新 Anchor。相同方法可以通过全局 `entity_id` 被多个 Anchor 引用，不必复制方法原文。

## 3. 每个 LayerEntry 必须保持原子性

不要让一个 entry 同时描述五个方法和三个结论。建议结构为：

```json
{
  "entry_id": "E-L3-032",
  "entity_id": "profile_guided_multiversioning",
  "anchor_id": "A017",
  "layer": "L3",
  "role": "baseline_behavior|opportunity|method|implementation|constraint|evaluation",
  "claim": "根据运行时 shape bucket 选择预编译 kernel variant",
  "modifiable_object": "Inductor shape guard and code cache",
  "applicable_baselines": ["B0"],
  "preconditions": ["shape distribution is concentrated"],
  "expected_effect": "reduce tail-wave latency",
  "evidence_refs": ["claim-182", "claim-206"],
  "confidence": "middle",
  "status": "candidate"
}
```

关键约束：

- 一个 entry 只有一个主要 claim；
- 方法本体通过 `entity_id` 全局复用；
- entry 表达该方法在当前 Anchor 中的具体角色；
- baseline、opportunity、实现和约束都可以作为独立 entry；
- 所有非 `unknown` 字段必须能回指原子证据。

因此，“一个表格条目有多个 entry”更准确的表达是：

> 每个 Layer 是一个 entry 集合；展示时一行一个 entry，不把多个 entry 塞入同一个 Markdown 单元格。

## 4. 跨层关系必须连接 Entry，而不是只连接 Layer

不能只记录：

```text
L2 → L4
```

因为 L2、L4 都可能有十几个候选条目。必须记录：

```text
E-L2-014 → E-L4-037
```

例如：

```json
{
  "from_entry": "E-L2-014",
  "to_entry": "E-L4-037",
  "relation": "controls",
  "interface": "scheduler-selected token count determines kernel tile variant",
  "compatibility": "conditional",
  "condition": "variant switching overhead < expected tail-wave saving",
  "evidence_confidence": "middle"
}
```

一个 Anchor 内可能产生多条不同路径：

```text
D1 = E-L1-02 → E-L2-14 → E-L4-37
D2 = E-L2-18 → E-L3-22 → E-L4-41
D3 = E-L3-25 → E-L4-44 → E-L5-09
```

这些才是送入专家评阅的候选方向。

## 5. Baseline 也应分为 Anchor 级和 Entry 级

Anchor 维护完整的 `Baseline Set`：

```text
B0：current-practice baseline
B1：strong baseline
B2：tool/evaluation baseline
B3：可复用实现 baseline
```

LayerEntry 再说明它与哪些 baseline 有关：

- 描述 B0 当前在本层如何执行；
- 暴露 B0 在本层的限制；
- 提供 B1 的替代实现；
- 提供 B2 的测量工具；
- 提供可以修改的代码入口。

这样 baseline 就会自然进入每层修改空间以及后续方向评判，而不是只有 Anchor 名称。

## 6. 专家评阅的单位应是 Direction，而不是整个 Anchor Map

Anchor Map 是候选池，可能包含相互替代的方案。若直接把整个 Map 交给 `idea_review_orchestrator`，评阅 Agent 很可能把不兼容 entry 错误合并。

正确顺序是：

```text
Anchor Map
  ↓
Entry 去重和关系发现
  ↓
形成多个 Direction Subgraph
  ↓
每个 Direction + Baseline Set 构成 Experiment Bundle
  ↓
idea_review_orchestrator 分别深审
```

评阅需要分别判断：

1. 每个选中 entry 是否成立；
2. entry 之间的接口是否真实；
3. 是否遗漏必要层；
4. 是否把替代方案误当成协同方案；
5. 单层 baseline 和组合 baseline 是否能区分协同收益；
6. 未被选中的 entry 是备用方案、冲突方案还是无关条目。

## 7. 最终建议的数据对象

```text
Global Entity
    论文方法、工具、代码、硬件机制，只存一份

Evidence Claim
    原文证据、来源位置、可信度

Anchor
    场景 × Baseline Set × Evaluation Context

LayerEntry
    某个 Entity/Claim 在某个 Anchor、某一层中的具体作用

CrossLayerEdge
    两个 Entry 之间的依赖、约束、冲突或协同

Direction
    从一个 Anchor 的 Entry Graph 中选出的连通子图

ExperimentBundle
    Direction + Baselines + Assets + Ablation Plan + Expert Judgment
```

因此，你的理解基本正确，但需要增加一个关键约束：

> 每个 Anchor 的每层可以有多个 entry；这些 entry 共同构成“候选修改空间”，并不自动属于同一个探索方向。具体方向必须从中选择一组兼容 entry，并通过 entry-level 的跨层边组成独立子图。
