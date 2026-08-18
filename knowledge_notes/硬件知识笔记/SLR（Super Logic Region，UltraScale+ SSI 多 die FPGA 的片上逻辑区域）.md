## SLR（Super Logic Region，UltraScale+ SSI 多 die FPGA 的片上逻辑区域）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SLR 是 AMD/Xilinx UltraScale+ 器件基于 stacked silicon interconnect（SSI）技术把多个硅 die 拼接成单一器件时，每个 die 对应的片上逻辑区域；die 间经 SSI 互连通信。Alveo U55C 与 Alveo U200 均为 3 SLR。跨 SLR 数据移动引入额外延迟与布线资源争用，因此 SLR 拓扑直接影响分区、布局与时序收敛（Web 佐证：U55C 的 HBM2 栈直连 SLR0，内存密集逻辑应靠近 SLR0 以减少跨 SLR 穿越；Alveo U200 同样 3 SLR）。
- 在 Graph.hls 中的角色：L3 参数"分区到 FPGA 资源（SLR 与 HBM 通道）的映射"把 SLR 当作一等设计维度；GH-Architect 产出的 system.cfg 写入 SLR/HBM 绑定；GH-Scope 静态对比 cross-SLR 连接数，识别劣质分区（如"新分区策略比 golden reference 多 40% cross-SLR 传输"）——因为跨 SLR 拥塞是 late-stage 布线失败（>3h 综合后才暴露）的根源。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Graph.hls 图加速器中的运转流程：U55C 的 3 SLR 容纳 14 个 pipeline slot（L3 启发式的容量前提）→ GH-Architect 把 11 little + 3 big pipeline 分配到 SLR/SLR 组合，HBM 通道绑定进 system.cfg → 综合/布线时跨 SLR 信号走 SSI 互连 → 若分区使同类 pipeline 的目的集跨 SLR 频繁交换数据，cross-SLR 连接数上升 → GH-Scope 在综合前用静态分析报告该指标并与 golden reference 对比 → 开发者据此调整分区策略或 pipeline 数量。
- 硬件架构视角要点：SLR 是"die 级并行 + die 间有限互连"的约束源；图处理的 big/little 异构 pipeline 天然适合按 SLR 分组（每 SLR 内聚一组 pipeline 减少穿越），这也是 U55C 上 14 slot 分组问题的物理背景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Vitis 中经 system.cfg / SLR 分配约束把 kernel/接口绑定到指定 SLR（Graph.hls 自动生成该绑定）；RTL 视角 SLR 对应 die，跨 SLR 用 SSI 互连。
- 使用：Graph.hls 把"分区→SLR/HBM 通道映射"提升为 L3 可配置参数，GH-Architect 生成一致绑定、GH-Scope 在综合前验证跨 SLR 代价，避免"综合 3 小时后布线失败"的 late-stage 失败。跨论文复用：任何多 die FPGA 加速器的分区/布局决策都可借用"cross-SLR 连接数作为质量指标 + 综合前静态检查"的方法。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
