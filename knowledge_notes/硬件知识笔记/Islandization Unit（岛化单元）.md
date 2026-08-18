## Islandization Unit（岛化单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Islandization Unit 是 L-PCN 提出的、插入在 Data Structuring Unit（DSU）与 Feature Computation Unit（FCU）之间的硬件模块，作用是在运行时检测并利用 point subsets 之间的空间重叠（spatial locality），消除 PCN 流程中的重复访存与重复 MLP 计算。它由三部分组成：Partitioning Module（执行 Octree-based Islandization：选 Hub points、Octree 邻接收集成 Hub List、形成 Islands 与 Island Lists）、Overlap Detection Module（用 Hub Octree 搜索检出当前 non-Hub subset 与已处理点之间的重叠点，输出 Overlap Indexes）、以及附着的 Hub Cache（存每个点的 MLP 结果供复用）。它可以作为 plug-in 无缝集成进准确型（PointACC、HgPCN）与近似型（EdgePC、Crescent）两类现有 PCN 加速器，提供额外 1.2×–3.2× 提速与 38%–56% 节能；其延迟开销 <1%（cycle-accurate），ASIC（TSMC 28nm 1GHz）面积开销 ~14%、功耗 ~10%。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 中 Islandization Unit 的运转流程（Figure 5/13，PointNet++(c)，每 Island 32 subsets）：
```
DSU 输出 point subsets（含 ~90% 重叠）-> Islandization Unit:
# (1) Partitioning Module: 2 个 OSE 并行在 Sampled Octree 上
#     选 Hub points -> 逐轮 Octree 邻接收集 -> Hub Lists -> Islands -> Island Lists
# (2) Overlap Detection Module: 按 Island List 自上而下处理每个 non-Hub subset:
#     对 subset 的 32 点在 Hub Octree 搜索 -> Overlap Indexes
#     - 重叠点 K 个: 从 Hub Cache 取缓存 MLP 结果 (K,128)，经 delta 补偿 -> Pooling
#     - 非重叠点 32-K 个: 从内存取特征 -> 进 FCU MLP -> (32-K,128) 回写 Hub Cache + 更新 Hub Octree
# (3) Hub Cache: 首 32 项存 Hub point subset 结果，之后存新增非重叠点结果
#     no-replacement policy，仅下个 Island 时整体替换
```
  - 该单元把"空间局部性"转化为时间局部性：岛内处理顺序（Hub 优先、沿 Island List 由内向外）保证缓存结果在 Island 处理期间持续命中，避免每次 subset 都重访内存。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：FPGA 原型用 SystemVerilog/VHDL 在 Intel Arria 10 GX（250 MHz）实现；Partitioning/Overlap Detection 模块由两个流水化 OSE + BRAM 缓冲（Octree Buffer、Hub-Octree Buffer）+ BRAM Hub Cache 构成。资源（PointNet++(c) 原型，Table II）：14,361 ALMs / 10,140 registers / 0 DSP / 770,048 BRAM，延迟仅 1,497 cycles（对比 DSU 1,046,461 与 FCU 1,263,176 cycles），即岛化开销可忽略。默认配置：每 Island 32 subsets、Hub-Cache 容量 = 单 subset 最大 feature size 的 2×；敏感度：更小 Island 一般更好（更快、更省能），更大 Island 数据复用变少但精度更高（speedup 2.94×→1.83×，精度 +0.2%）。论文未提供开源 RTL。相关先例：I-GCN（MICRO'21）的 islandization 硬件（Island Locator + Island Consumer）面向 GCN。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
