## α-β Communication Model for Optimal Expert Broadcast

术语解释
α-β 通信模型是分布式计算中描述 point-to-point 消息传递延迟的经典模型：发送大小为 m 的消息所需时间 = α + β·m，其中 α 是消息启动延迟（latency per message），β 是每字节传输时间的倒数（1/bandwidth）。HD-MoE 使用该模型推导 MoE expert 预广播的最优 chunk size c，以在给定 runtime window 内最大化广播效率。

术语是什么？
α-β 模型（也称 Hockney 模型或 postal model）将通信分解为两个独立成本：(1) α（latency term）：每次通信的固定开销（包括软件协议栈开销、路由建立延迟等），与消息大小无关；(2) β（bandwidth term）：每单位数据的传输时间 = 1/bandwidth。总延迟 T(m) = α + β·m。该模型适用于消息大小适中的场景，对于极小消息（α 主导）和极大消息（β 主导）需要更复杂的 LogP/logGP 模型。

从kernel调度角度拆解术语
HD-MoE 使用 α-β 模型推导 expert 预广播的最优 chunk size：
```
# 给定：expert 大小 = h·IS, mesh sqrt(D)×sqrt(D), 可广播时间窗口 k iterations
# latency  = α · (2√D + h·IS/c)  # α: per-hop + per-chunk overhead
# bandwidth = β · (h·IS + 2c√D)   # β: 1/BW
# t_pre_b = latency + bandwidth

# 下界（当 chunk size c 最优时）：
# t_pre_b ≥ h·IS·β·k + 2·α·√D + 2·√(2√D·β·k·α·h·IS)

# 最优 chunk size：
c* = √(α·h·IS / (2·β·k·√D))
```
在 batch=512, 5 TFLOPS/50 GB/s 配置下，上层推理时间允许预广播 2 个 expert；在 2.5 TFLOPS/75 GB/s 配置下可预广播 5 个 expert。预广播的 expert 被分成 c* 大小的 chunk，利用多跳路径并发传输以最小化链路拥塞。

术语一般如何实现？如何使用？
α-β 模型广泛应用于 MPI collective 通信建模（Hockney 1994）、GPU 间通信建模（NCCL 性能模型）和分布式训练通信优化（DeepSpeed、Megatron）。α 和 β 值通过 microbenchmark 测量得到：发送不同大小消息，拟合 T(m) = α + β·m 线性回归。HD-MoE 的 α 对应 NoC hop latency (~0.1-5µs per hop)，β = 1/BW (~0.013-0.04 ns/byte for 25-75 GB/s)。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

HierMoE 将 α-β 模型扩展到多维度分层 AlltoAll 通信建模。对于 D 维分层 AlltoAll：t_d = Σ_{i=1}^{d-1} (n_inter_i · β_inter(i) + α_inter(i)) + n_intra · β_intra(d-1) + α_intra(d-1)。通过 nccl-tests 一次性测量 7 种 AlltoAll 变体的 α, β（r² > 0.997, <300s），训练期间无需重新校准。该模型驱动最优维度 d* 选择和 expert swap 决策矩阵 Q_d*。
