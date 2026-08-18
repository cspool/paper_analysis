## SSD 内部带宽与外部带宽（Internal vs External Bandwidth）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SSD 外部带宽指 host 与 SSD 经 PCIe 接口传输数据的带宽（GRAINS 评估：SSD-G4 = PCIe Gen4 顺序读 7 GB/s；SSD-G5 = PCIe Gen5 14.8 GB/s）；内部带宽指 SSD 控制器与 NAND flash 之间的带宽（16 channel × 每通道峰值，如 57.6 GB/s 内部 vs 14 GB/s 外部）。现代 SSD 过度配置内部带宽以保护用户可见的外部 I/O 性能，缓解 channel 争用、内部数据迁移（GC/磨损均衡）与 refresh 的影响。IFP 在 flash die 内处理时利用全部 die 的聚合带宽，该带宽随 die 数增长，显著超过控制器侧带宽与外部带宽。GRAINS 依赖此特性：把低复用大数据的处理搬进 SSD 后，带宽瓶颈从"外部接口整页搬运"变为"die 内处理+只回传结果"。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 动机分析（§3）：低复用图数据（659/822 GB）经外部接口整体搬运是端到端性能主导瓶颈——No-I/O 配置比 SSD-G4（G5）在 Fulgor/MetaGraph 上平均快 16.7×（9.3×）/7.5×（4.5×），且图越大 I/O 占比越高（2.7×→9.2×）。GRAINS 通过 SCC 消除该瓶颈：GRN-Ext（同优化、逻辑在 SSD 外、PCIe 16 GB/s 外部带宽）只能平均 3.4×/5.0× 超 FG/MG，因为局部性优化摊不薄"低复用数据跨存储-host 接口"的搬运成本；GRN（IFP/ISP 在 SSD 内）平均再快 2×，正是把访问留在 die 内、只回传结果对内部带宽的利用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 由 PCIe 代际（Gen4/Gen5）、channel 数与每通道 I/O 速率（SSD-G4/G5 为 1.2/2.4 GB/s）、die 数决定；评估中外部带宽由真实系统测量、内部带宽由 MQSim 忠实建模（Table 1 配置：16 channel、8 die/channel、4 plane/die、4-KiB 页）。用于解释 SCC 收益的量化依据：低复用数据的搬运成本（外部带宽受限）vs 存储内并行处理（内部/die 聚合带宽充裕）之间的带宽鸿沟。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
