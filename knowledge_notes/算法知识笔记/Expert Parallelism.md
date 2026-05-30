## Expert Parallelism

术语是什么？
Expert Parallelism（专家并行）是 MoE 模型训练中一种专门针对 expert 参数的并行策略。核心思想：将 MoE layer 中的 E 个 expert 均匀分布到 N 个 GPU 设备上（每个设备持有 E/N 个 expert），每个 token 通过 Router 被分配到目标 expert 所在的设备上进行计算。与 Data Parallelism（每个设备持有完整模型副本）和传统 Model Parallelism（按层切分）不同，Expert Parallelism 按 expert 粒度切分模型参数，配合 All-to-All 通信原语实现 token 的跨设备路由。从 MoE 架构提出（Shazeer et al. 2017）到 Switch Transformer（Fedus et al. 2021），Expert Parallelism 已成为 MoE 训练的标准分布式策略。

从算法pipeline角度拆解术语：
Expert Parallelism 的 MoE layer 前向计算流程（以 top-2 gating 为例，每层 N 个设备）：

```
输入: T_I ∈ R^{B, M}   // B tokens, model dim M
输出: T_O ∈ R^{B, M}

每层执行:
1. Router: G(T_I) = softmax(W_g · T_I) ∈ R^{B, E}
2. Top-K gating: 选择每 token 的 top-2 expert
3. 统计每个 expert 收到的 token 数 → input_split_sizes[N], output_split_sizes[N]
4. // 第一个 All-to-All: Dispatch
   T_DI = All-to-All(T_I, input_split_sizes, output_split_sizes)
5. // Expert FFN 计算（每个 device 独立执行其 local experts）
   for each received chunk c:
       T_M[c] = GeLU(Linear1(T_DI[c]))
       T_DO[c] = Linear2(T_M[c])
6. // 第二个 All-to-All: Combine
   T_O = All-to-All(T_DO, output_split_sizes, input_split_sizes)
```

关键内存分析（MPMoE Equation 1-3）：
- Model States: M_ms = 4 * (E*M + 2*H*M)
- Activations: M_act = 4*B*M + B*H
- Temporary Buffers: M_buf = B*M + B*H

Expert Parallelism 通过分布式存储 expert 参数解决了 model states 的内存瓶颈，但 All-to-All 通信成为新的性能瓶颈。

术语一般如何实现？如何使用？
- 实现框架：FastMoE（PyTorch 原语）、DeepSpeed-MoE（分层 All-to-All + 自定义 CUDA kernel）、FasterMoE（pipeline + expert shadowing）、MPMoE（微批次 pipeline + 内存复用）。
- 关键考量：(a) Router 计算量极小但需全局同步 token 分布；(b) 不均匀的 expert 负载导致 All-to-All 出现 straggler；(c) Expert Parallelism 通常与 Data Parallelism 组合。
- MPMoE 的改进：沿 batch 维度切分 micro-batch 进行 pipeline（保留 NCCL All-to-All 优化），从固定 granularity 升级为自适应 granularity。

MixNet 从网络架构视角进一步揭示了 EP 通信的三个关键特性（基于生产环境 128 H800 GPU 的 Mixtral 8×7B 训练测量）：
1. **时间非确定性**：每个 training iteration 中 token-specific expert activation 导致 all-to-all 通信矩阵在 iterations 间显著变化。即使使用 load balancing loss，traffic matrix 的 sparsity 始终存在。
2. **空间非均匀性**：每个 traffic matrix 是非均匀的——仅有少数 GPU 对（heavy hitters）之间有大流量通信，大部分 GPU 对之间通信量很小或为零。
3. **强局部性**：仅同一 MoE block 内的 expert 层需要 all-to-all 通信——不同 PP stage 的 expert 层不直接通信。

这三项特性是 MixNet 设计区域可重构 OCS 高带宽域的理论基础。EP 的 all-to-all 通信量占比显著：Mixtral 8×7B 中 EP 占 30%（TP 占 60%），LLaMA-MoE 和 Qwen-MoE 中 EP 超过 80%。EP 的 all-to-all 通信占据了 33%-55% 的总训练迭代时间（400 Gbps 网络）。

- **MoE Parallel Folding 中的 EP**：该论文将 EP 从 DP 的子组中解放出来，允许 EP 折叠到 Attention 层的 TP/CP/DP 任意子组中。这使得 EP 的 All-to-All 通信可以限制在 NVLink 域（节点内 450 GB/s）而非 InfiniBand 域（节点间 400 Gbps），显著降低通信开销。同时通过统一 token dispatcher 支持 EP 与 ETP 的任意组合。配置示例（Mixtral 8x22B, 128 H100）：Attention TP=2, CP=1, DP=8, PP=8；MoE ETP=1, EP=8, PP=8（MoE 层纯 EP 不做 TP）。EP=8 时 8 个完整 expert 分布在 8 GPU，GEMM 效率最高。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD 的 EP+SD 兼容性分析**：在 EP 配置下，expert 分布到多 GPU，N(t) 和 Texp 不受影响（仅影响每 GPU 持有的 expert 子集），因此 MoESD 的理论分析仍然有效。MoE FFN 仍占显著处理时间 → memory-boundness 效应在端到端性能中可观测。值得注意的是，在大量 EP GPU 配置下，小 batch 时的 SD 低效问题可能消失——因为 EP 提供的额外聚合内存带宽使验证阶段计算增量更容易被吸收。

---
