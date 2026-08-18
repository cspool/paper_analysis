## Hub Cache（Hub 缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hub Cache 是 L-PCN Islandization Unit 中存储可复用结果的缓存，每个条目存一个点的 MLP 结果。组织方式：前 32 个条目存 Hub point subset 的 MLP 结果（假设 Hub subset 大小为 32），第 33 到 N 个条目存后续 non-Hub subset 中新出现的非重叠点的 MLP 结果。处理 non-Hub subset 时，Hub Cache 依据 Overlap Detection Module 提供的 Overlap Indexes 返回重叠点的缓存结果；缓存采用 no-replacement policy（岛内不淘汰），只在下一个 Island 开始时整体替换。容量默认配置为单个 subset 最大 feature size 的 2×（敏感度研究显示：小 Island 下更大缓存基本无收益即 overprovisioning，大 Island 下更大缓存能提升性能）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 中 Hub Cache 的运转流程（Figure 12，N 条目）：
```
# Island 开始: Hub subset 32 点全量进 FCU MLP -> 结果写 HubCache[0..31]
# 处理 non-Hub subset:
#   K 个重叠点: 用 Overlap Indexes 读 HubCache -> (K,128) 结果 + delta 补偿 -> Pooling
#   32-K 个非重叠点: 算完 MLP -> (32-K,128) 写 HubCache[33..] 空位 -> 更新 Hub Octree
# Island 结束: 整体替换（no-replacement policy 岛内不淘汰）
```
  - 与 CNN/GCN 加速器的 locality 缓存思想同源（如 Eyeriss 的片上数据复用、I-GCN 的片上局部性增强），但 Hub Cache 存的是"特征计算结果"而非中间激活，配合结果增量补偿实现跨子集复用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：FPGA 上为 BRAM 实现（Arria 10 GX 原型，Islandization Unit 总 BRAM 770,048 bits）；ASIC 版用 TSMC memory Compiler 生成 SRAM（Islandization Unit 约 14% 面积、10% 功耗开销中 SRAM 占主要部分）。论文未提供开源 RTL。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
