## Instruction Chain Bus / Mesh Bus / Data Ring Bus（ICB / Mesh / DRB，M100 三级片上互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- M100 NPU 的三种片上互连：① Instruction Chain Bus（ICB）——CCB 到 TPB cluster 的 daisy-chain 指令分发总线（64 bits/cycle）；TPB 指令数千 bit、传输数百 cycle，但执行数十万 cycle，故派发非瓶颈。② 2D Mesh Bus——可扩展高带宽点对点互连，连接 TPB cluster、CCB、CPU、DMA、Block SRAM，每节点对最高 256 GB/s，低拥塞下扩展良好。③ Data Ring Bus（DRB）——确定性、高效广播路径，聚合最高 256 GB/s，适合跨 TPB 多播与权重广播（CCB DMA 经 DRB 广播权重，匹配 DDR 读带宽）。软件按通信需求在 Mesh（点对点）与 DRB（广播）间动态选择。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：CCB 固件经 ICB 广播 TPB 指令（destination mask 指定目标 TPB 组，含 tensor shape/通信需求元数据）→ 数据搬运：小规模/一对多广播走 DRB（确定性高带宽）、点对点/可扩展通信走 Mesh Bus；权重由 CCB DMA 经 DRB 以 256GB/s 广播到全部 TPB。cluster 内 4 TPB 用 cluster NoC 低延迟通信，跨 cluster 走 Mesh（效率低于 cluster 内，需编译器注意）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ICB 为 daisy-chain 总线 + cluster 指令队列；Mesh 为 2D 网络（每节点对 256GB/s）；DRB 为环形广播网络（聚合 256GB/s）。使用：编译器/固件选择互连路径；对跨 cluster 通信做优化（降低通信效率影响）。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
