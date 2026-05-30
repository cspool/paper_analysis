## α-β Communication Model（Alpha-Beta 通信模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
α-β 模型（也称 postal model）是并行计算中最基础的通信成本模型，将发送大小为 s 字节的消息所需时间建模为 `T(s) = α + β·s`。其中 α 是延迟项（latency，单位秒），表示每次通信的固定开销（启动延迟、协议握手等）；β 是带宽项的倒数（`β = 1/bandwidth`，单位秒/字节），表示每字节传输时间。NetMoE 将 All-to-All 通信中的三类通道（intra-device 内存拷贝、intra-node NVLink、inter-node InfiniBand）分别用 α-β 模型建模，由于 intra-device 带宽极高（~2 TB/s），其通信时间被忽略。最终通信时间取两类通道的最大值：`t = max(t_intra, t_inter) = max(α_intra + β_intra·s_intra, α_inter + β_inter·s_inter)`。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 NetMoE 的系统架构中，α-β 模型用于：
1. **Profiling 阶段**：训练前对硬件环境进行 profiling，获取各通道的 α（latency）和 v=1/β（bandwidth）。论文使用 A800 集群：`v_intra = 400 GB/s`（NVLink），`v_inter = 100 GB/s`（InfiniBand）。
2. **优化目标**：由于 `α_{inter}` 和 `α_{intra}` 是常数，优化目标简化为最小化通信量——通过 KM 算法求解的边权重仅依赖通信量 `c_{i,n}` 和 `c'_{i,j}`（以 token 数为单位），而非带宽。这是因为在 α-β 模型下，给定 profiled 带宽，最小化通信时间等价于最小化瓶颈通道的通信量。
3. **两层建模**：Stage 1 优化仅针对 `s_inter`（跨节点通信量），因为 `v_intra >> v_inter` 意味着 inter-node 几乎总是瓶颈。Stage 2 对每个 node 内优化 `s_intra`，不改变 Stage 1 的 inter-node 分配。
4. **局限性**：α-β 模型假设理想通信（无路由冲突、无拥塞），NetMoE 的实测加速略低于理论值（论文 Fig. 7 显示实际加速 < 理论加速），因为实际硬件中存在 NCCL 内部的路由冲突和协议开销未被建模。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在分布式训练中，α-β 模型通常通过 ping-pong 测试（两个 GPU 间发送不同大小的消息，线性拟合得到 α 和 β）来 profiling。
- NCCL 的实际性能偏离简单 α-β 模型的原因：消息大小阈值切换不同的传输协议（eager vs rendezvous）、NVLink 和 InfiniBand 的多通道并行、以及 ring/chunked 算法对带宽利用率的影响。
- 更精确的通信模型包括 LogP、LogGP、LogGPS 等，它们额外建模了 CPU overhead（o）、gap（g）等参数，但在 MoE 训练场景中，简单的 α-β 模型已足够指导优化方向。
- NetMoE 利用 α-β 模型的线性性质将通信时间最小化转化为简单的通信量最小化（加权二分图匹配），显著降低了优化问题的复杂度。

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
