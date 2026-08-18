## VM 放置策略（Random-fit / Best-fit / Topology-aware）

术语解释
DDC 中把 VM/任务的计算与内存需求分配到 compute/memory 节点的三类代表性放置策略；R2D2 用它们分析 disaggregation 流量的空间稀疏度与时间稳定性，并揭示策略如何加剧或缓解网络过配。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Random-fit：随机配对 compute 与 memory 节点，优先调度速度（FastSwap [3]）或负载均衡（Farm [17]）——论文分析中为稀疏/稳定的最坏情况（512 节点 98.8% 链路无流量、每节点 6 条活动链路）。
- Best-fit：最小化碎片化（Infiniswap [30]），对所有 compute-memory 对一视同仁（99.1% 稀疏、每节点 5 条活动链路）。
- Topology-aware：优先分配到已有亲和性的 compute-memory 对（Clio [31]、Hermit [57]、LegoOS [61]），稀疏与稳定最优（99.5% 稀疏、每节点 3 条活动链路；时间稳定性 >0.997）。
- 关键结论：即使 worst-case 的 Random-fit 也只需 ≤6 条/节点并发链路，远低于全连接的最坏 511/2047 条——这是 R2D2"按需只建 ~4-6 条链路"的可行性依据。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 放置策略决定流量矩阵：给定 VM trace（Protean，2064 机器、48 核/384GB），调度器把每个 VM 的 compute 与 memory 需求落到节点 → 形成 compute-memory 流量热力图（TVMPP/ECHO 方法）与时间稳定性曲线（Infiniswap/Sinbad 方法）。topology-aware 把配对集中到已连接节点 → 更稀疏更稳定；random 打散配对 → 更多活动链路但仍在 6 条内。
- 在 R2D2 runtime 中，联合算法（见上一条目）即"拓扑感知的放置"升级版：不仅优先已连接节点，还主动重构拓扑以维持长期稀疏稳定，并把重构成本计入 fitness——对应解决 baseline 放置策略（在静态网络上放置）无法控制网络形态的缺陷。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 datacenter 编排器/资源管理器中作为分配启发式实现（Protean 等生产系统用 best-fit 类策略）；R2D2 论文用三类策略驱动流量分析（§II-C）与运行时算法对比 baseline（best-fit on R2D2 硬件，10-20× 更高分配延迟）。
- 使用：任务/VM 分配、负载均衡、碎片最小化；Topology-aware 在 Clio/Hermit/LegoOS 中通过保留 affinity 信息实现。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
