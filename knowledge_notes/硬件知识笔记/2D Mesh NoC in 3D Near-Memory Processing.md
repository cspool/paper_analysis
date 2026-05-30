## 2D Mesh NoC in 3D Near-Memory Processing

术语解释
2D mesh Network-on-Chip 是 3D NMP 加速器中连接分布式计算节点的片上互联网络拓扑。每个节点通过路由器连接到其东、西、南、北四个邻居，数据包通过 XY routing（先沿 X 轴后沿 Y 轴）计算 Manhattan 最短路径进行传输。

术语是什么？
2D mesh NoC 是一种规则网格拓扑的片上网络，每个 tile（计算节点+内存 bank）位于网格交叉点上，通过双向链路连接相邻 tile。在 3D NMP 架构中，NoC 替代了 GPU 中的 shared memory/crossbar 互联，负责传输 token hidden states（all-to-all dispatch/combine）和 expert 权重（pre-broadcast）。HD-MoE 论文中 NoC 链路带宽为 25-75 GB/s per link，mesh 尺寸包括 4×4 (16 nodes)、4×8 (32 nodes)、8×8 (64 nodes)。通信模式分为：(1) Intra-Expert Communication（TP 模式下同一 expert 分片在不同节点间的 all-reduce 同步）；(2) Inter-Expert Communication（EP 模式下不同 expert 间的 all-to-all token dispatch/combine）。

从硬件架构角度拆解术语
HD-MoE 论文使用 discrete-event simulator 精确建模 2D mesh NoC 的通信延迟：
1. **通信任务生成**：对每个 token 的 activated expert group，确定 expert 物理节点（src）和聚合目标节点（dst）。
2. **XY Routing 路径计算**：缓存 XY routing 算法生成 Manhattan distance 最短路径上的每一跳。
3. **事件调度与链路管理**：通信任务按 chunk 切分，进入 priority queue 按时间戳调度。link schedule dictionary 追踪每条链路的占用时间表，新任务在链路空闲时才开始传输。传输时间 = chunk_size / BW。
4. **性能度量**：取最后一个通信任务完成时间作为 t_comm。
HD-MoE 的关键优化（Link Balance）正是基于这一模拟器，通过 Bayesian Optimization 搜索逻辑集群到物理节点的映射，最小化链路级拥塞。

术语一般如何实现？如何使用？
2D mesh NoC 在学术界广泛使用 BookSim2、Gem5-Garnet、ASTRA-sim 等模拟器进行评估。HD-MoE 使用自建 Python 离散事件模拟器，验证工具为 ASTRA-sim（用于 ring all-reduce 延迟校验：模型预测 673µs vs ASTRA-sim 仿真 668µs，误差 <1%）。商用 3D NMP 芯片（如 Samsung HBM-PIM）的内部互联使用类似 mesh/torus 拓扑但具体实现未公开。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
