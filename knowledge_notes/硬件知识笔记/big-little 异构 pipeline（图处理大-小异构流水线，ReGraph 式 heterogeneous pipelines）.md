## big-little 异构 pipeline（图处理大-小异构流水线，ReGraph 式 heterogeneous pipelines）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- big-little 异构 pipeline 是 ReGraph（MICRO 2022，Xinyu Chen 等）提出的 HBM FPGA 图处理架构：把处理 pipeline 分为两类资源形态——little pipeline（面向高密度分区，吞吐高）与 big pipeline（面向稀疏分区，容量大），配合 graph-aware 任务调度把图分区调度到正确 pipeline 类型并生成最高效的 pipeline 组合、平衡负载（Web 佐证：ReGraph 论文摘要）。在 ReGraph 中：dense 分区由全部 little pipeline 处理、sparse 分区由全部 big pipeline 处理，且同类 pipeline 共享同一目的节点集——所有 little pipeline 处理含 65,536 目的节点的分区、所有 big pipeline 处理含 524,288 节点的分区。
- 在 Graph.hls 中：big-little 是 L3 pipeline 分组策略的代表性 baseline 设计；论文指出其固定 2-class 结构在多样图拓扑下负载失衡（稀疏图 big pipeline 全部处理同一受限目的集而利用率低；大稠密图 65,536 节点分区不够、若半条 little pipeline 处理共享分区则另一半空闲）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件架构中的运转流程（ReGraph）：预处理把图按度分布切成两类分区（dense/sparse）→ 分区调度到对应 pipeline 类（little 处理 dense、big 处理 sparse）→ 每轮迭代各 pipeline 流式处理自己的边集、更新共享目的节点集 → 收敛后聚合。L1 活跃顶点过滤阈值（degree 阈值）决定顶点路由到 little 还是 big 路径，直接影响两类 pipeline 的负载分布。
- Graph.hls 的泛化（硬件架构视角）：把固定 2-class 泛化为任意 N-class 分区——每类独立目的节点集、可配置 pipeline 数、N 个独立 merger 替代 2 个共享 merger，host 侧多类调度与独立 HBM buffer 分配；U55C 3 SLR 的 14 个 pipeline slot 可分组为 (11 little, 3 big) 等任意组合。效果：L3 全 DSE 使 SSSP 从 L1+L2 的 2.95× 提到 4.48×（pipeline 形态适配新吞吐画像）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ReGraph 为手工 HLS C++（固定 2-class 深度嵌入代码）；Graph.hls 把分组策略作为 L3 配置参数，GH-Architect 按度分布启发式选择分组（如 rmat-21-32 幂律分布→11 little+3 big）并自动生成对应多分区架构（N 个 merger、每类独立 buffer/host 调度），修改成本从 >1000 行降为配置声明。
- 使用：作为图加速器 pipeline 组织的通用模板——"异构 pipeline 按度分布分组处理异构图数据"可复用于其它 HBM/DDR FPGA 图处理设计；Graph.hls 用 GH-Scope 把 ReGraph/ThunderGP/GraphLily 的架构策略作为 golden reference 组合对比，在综合前评估。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
