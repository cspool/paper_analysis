## Vector Multicast Network（向量多播网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vector multicast network 是 SegFold 的全局片上互联（源自 vector multicast [50]），负责把 B 行从内存后端多播到各 PE 行：因为 SELECTA 可能同时选多个 k，多 B 行需在同一周期活跃，网络必须支持"一次从内存取多条 B 行、分发给多个 PE 行"。实现为向量化 crossbar（vectorized crossbar），关键设计参数是一次能并行多播的 B 行数（每 PE 行同一时刻至多处理一条 B 行）；论文选 4 路多播平衡面积/能耗与性能（敏感性实验 1→16 路：1→4 收益显著，>4 边际收益骤减，d=0.1 高密度大矩阵下网络争用更敏感）。配套部件：每 PE 行一个 row shifter，把到达的 B 元素重对齐到 IPM 指定的连续 merger 起点（同一 B 行的元素必须映射到连续 merger 集，头位置动态决定）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：内存控制器（SELECTA）选出 (m,k) 对后，对应 B 行（DCSR 读）经 vector multicast crossbar（≤4 行/cycle）路由到目标 PE 行 → 各 PE 行的 row shifter 按 IPM 给出的注入点把元素重对齐到连续 merger → 进入 merge network 做 SEGMENTBC。跨 PE 行通信与 PE 级 mesh（四邻居）互补：mesh 提供细粒度本地通信，vector multicast 提供高带宽全局再分发（B 段路由到合适 PE 行、无冲突）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：向量化 crossbar + per-PE-row row shifter；网络端口流式多条 vector 增加复杂度，故以"最多 4 路 B 行/cycle"为硬件上限（Table II 参数）。使用：带宽是敏感性研究轴之一（Fig.12 crossbar width sweep，normalized to BRL=4，BRL=Broadcast Row per cycle），高密度下更宽网络缓解争用但面积/布线成本高，默认取 4。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
