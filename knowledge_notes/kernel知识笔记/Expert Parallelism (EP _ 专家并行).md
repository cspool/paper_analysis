## Expert Parallelism (EP / 专家并行)

术语是什么？
Expert Parallelism (EP) 是一种 MoE 模型训练/推理中的分布式并行策略。将不同的 expert 的权重分布到不同的计算设备（GPU/加速器）上，每个设备持有若干完整 expert 的权重。Router 在运行时将 token 按 top-k routing 结果发送到持有对应 expert 的设备，计算完成后将结果传回。EP 的核心收益是每个设备只需存储和处理部分 expert，使得总参数量可超过单设备内存限制，同时各 expert 的 matmul 保持了较大的 kernel size。

从kernel调度角度拆解术语：
在 MoE 训练的一个 iteration 中，EP 的 kernel 调度流程为：

```
// 假设 4 个 GPU，每个持有 2 个 expert（共 8 experts, top_k=2）

// Step 1: Router (每个 GPU 独立执行)
logits = Router(local_tokens)                     // (B_local*S, E)
topk_vals, topk_idx = topk(softmax(logits), k=2)

// Step 2: All-to-All Dispatch (通信 kernel)
// 在每个 GPU 上：
expert_tokens = {}  // 按目标 device 分组
for token in local_tokens:
    for expert_id in topk_idx[token]:
        target_device = expert_to_device[expert_id]
        expert_tokens[target_device].append(token)
// All-to-All scatter: 将 token 发送到持有对应 expert 的设备

// Step 3: Expert Compute (计算 kernel)
// 收到来自各 GPU 的 token 后：
for expert_e in my_experts:
    if has_tokens_for(expert_e):
        output = SwiGLU_FFN(expert_e, tokens)
        
// Step 4: All-to-All Combine (通信 kernel)
// 将 expert 输出送回 token 原始所在设备

// Step 5: Token Reorder (reorder kernel)
// 将返回的输出按原始 token 顺序排列
Y = reorder_by_token_index(returned_outputs)
```

BTA 论文指出，在 wafer-scale 处理器上，EP 解决的是跨设备 expert 分布问题，但不能解决同一设备内 attention 与 expert 的 batch size 冲突。BTA 与 EP 是互补的：EP 跨晶圆/设备分布 expert，BTA 在同一设备内解耦 attention 和 expert 的 batch size。

术语一般如何实现？如何使用？
主流框架实现：
- Megatron-Core：通过 `moe_ep_size` 参数配置 EP 并行度，与 TP/DP/PP 混合使用。推荐 Mixtral-8x7B 在 64 GPU 上用 TP=1, EP=8, PP=4。
- DeepSpeed-MoE：基于 DeepSpeed 的 EP 实现，支持 expert 到 GPU 的灵活映射和 All-to-All 通信优化。
- Tutel：自适应 MoE 框架，支持动态 expert 分配和 load balancing。
- FineMoE/MoE-Infinity：使用 hash map 做 expert→GPU 映射，round-robin 确保 GPU 间 expert 数均衡。

主要挑战：(1) Load imbalance — 不同 expert 被选中的 token 数不均衡；(2) All-to-All 通信开销 — 尤其在跨节点场景；(3) 小 top_k 时每个 expert 的有效 batch 过小，计算密度低；(4) Checkpoint 侧——EP 将 expert 分布在不同 ranks 上，现有 baseline（如 Megatron-DeepSpeed）仅用 EP-Group-0 保存所有 expert checkpoint，造成 bottleneck rank 负载过高而其他 EP groups 闲置。MoC-System (ASPLOS '25) 的 Fully Sharded Checkpointing 将 expert checkpoint 按 expert 切分在所有 EP groups 间均分，消除此瓶颈（bottleneck workload 降低 22%-29%）。

涉及论文标题：
- Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- ReXMoE Reusing Experts with Minimal Overhead in Mixture-of-Experts
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
- Sparse Upcycling Training Mixture-of-Experts from Dense Checkpoints（TPU v4 上使用 expert partitioning 分布 32 experts；Base/Large 用 64 chips，XL 用 256 chips + 4-way model partitioning）
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
- Upcycling Large Language Models into Mixture of Experts

Sem-MoE 论文揭示了 EP 在 MoE **推理**场景中的严重通信瓶颈：即使在高带宽互联（>400GB/s）下，all-to-all 占 DeepSeek-V2-Lite MoE layer forward latency 的 59.2%。Sem-MoE 通过 semantic-aware model-data collaborative scheduling 提升 Local Activation Rate (LAR) 从 25% 到 62-68%，从而直接减少 all-to-all 通信量 49-57%。

Pre-gated MoE (ISCA '24) 从另一个角度解决 EP 的问题：不使用 multi-GPU EP，而是将所有 expert 参数 offload 到 CPU，仅通过单 GPU 推理。通过 pre-gate function 提前知道下一个 block 需要的 experts，利用 CUDA stream 将 CPU→GPU expert migration 与 GPU expert computation 重叠，避免了 EP 中的 All-to-All 通信开销和 load imbalance 问题。

**ScMoE (ICML '25)** 从架构-调度协同设计角度优化 EP 通信瓶颈：通过shortcut连接使gating和All-to-All dispatch可以基于前一层表示提前启动，与当前层的attention+shared expert计算重叠。当通信时间 ≤ overlap_window（约50%总MoE时间）时实现100%通信隐藏（pipeline策略因其prologue/epilogue bubble无法达到）。在8×A30-PCIe（通信占60%）下实现1.49×训练加速和1.82×推理加速。

ScaleMoE 论文揭示了 EP 中 All-to-All 通信的 zero padding 问题：由于 expert selection 高度不均衡，Tutel/DeepSpeed 等框架为统一 all-to-all message size 而加入大量 zero padding（zero ratio 从训练初期 88% 升至后期 98%），导致通信量膨胀。ScaleMoE 提出 Adaptive All-to-All Communication 通过 all-gather 聚合 per-expert 选择计数后使用精确 slice size 的 NCCL alltoallv，消除 zero padding——all-to-all 通信开销减少 up to 81%。此外，Dynamic Expert Clustering 通过 K-means 聚类 tokens + expert replication + cold expert offload 重新均衡 EP 下的 expert-to-GPU 映射；Topology-aware Expert Remapping 使用遗传算法在异构网络中搜索近最优 cluster-to-GPU 映射。

涉及论文标题：
- Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
- ReXMoE Reusing Experts with Minimal Overhead in Mixture-of-Experts
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（首次在 Mesh TensorFlow 中系统提出 Expert Parallelism 作为独立并行维度，与 Data/Model Parallelism 组合；Switch-C 1.6T 参数模型使用 2048 experts 纯 EP+DP，Switch-XXL 395B 使用 EP+MP+DP）

**ES-MoE (ICML '24)** 将传统 EP 中 experts 常驻 GPU 的假设打破，通过 expert offloading + dynamic placement 实现 on-demand EP：experts 不再静态分配给 GPU，每 iteration 根据 gating output 动态决定 expert→GPU 映射。GPU 仅持有 non-expert params + active expert params + activations。Expert placement 由 greedy scheduling 算法在 CPU 执行（<2.69us），按 token load 均衡分配。

PopFetcher (USENIX ATC '25) 在 EP 基础上引入 popularity-based expert-wise prefetching：利用滑动窗口（s=10 iterations）预测下一层热门 expert，在 Attention 层（非 MoE 计算）期间通过独立 CUDA stream 异步预取 remote expert 参数到本地 GPU。已预取的 expert 的 token 直接本地计算——消除该部分 token 的 All-to-All dispatch。采用 hybrid push-pull 范式：当 token 传输量 > 2048 tokens（H=1024, ~16MB expert 参数）时 pull expert，否则 push token。在 8×RTX 4090 (100Gbps InfiniBand) 上，token 传输量减少 14.85%（MoE-GPT）、13.46%（MoE-BERT），per-iteration time 加速 1.28-2.4×。

- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

Pro-Prophet (NUDT) 提出 lightweight expert placement 优化 EP 中的 expert-to-device 映射：每个 expert 仅传输到有其 input 的 device 子集（而非全部 devices），通过 Trans 原语（传输 parameters）和 Agg 原语（聚合 gradients）替代全局 model states 传输，显著降低通信量。Pro-Prophet 的 planner 通过 greedy algorithm + performance model 在 runtime 搜索 communication-efficient placement。

ScheMoE (EuroSys '24) 从任务调度角度重新审视 EP：将 EP 中 MoE layer forward/backward 的 7 类任务（compress、A2A dispatch、decompress、expert compute、compress、A2A combine、decompress）形式化为带数据依赖约束的调度问题，并通过数学证明给出了给定输入分区度 r 下的最优 CompTask 执行顺序（OptSche 算法）。此外，ScheMoE 提出 Pipe-A2A 通信算法——将 EP 中 A2A 的 intra-node SR 和 inter-node SR 分配到两个独立 CUDA stream 并发执行，使 EP 的通信阶段同时利用 intra-node 和 inter-node 带宽。ScheMoE 的 AbsCompressor/AbsAlltoAll/AbsExpert 三层抽象接口使得 EP 中的压缩算法和 A2A 算法可插拔替换而无需修改调度逻辑。

UCCL-EP (2025) 从通信系统可移植性角度解决 EP 的 vendor lock-in 问题：现有 EP 通信系统（DeepEP）通过 IBGDA 实现 GPU-initiated token-level 通信，但每个 (GPU vendor, NIC vendor) 组合需独立开发（O(m×n) 成本）。UCCL-EP 通过 CPU-proxy 架构解耦 GPU 通信发起与 NIC 通信执行——GPU 通过 FIFO channel 将 TransferCmd 传递给多线程 CPU proxy，CPU 通过 libibverbs（可移植 RDMA 库）执行所有 NIC 操作——仅需 O(m+n) 开发成本。在 EFA（无序传输、无硬件 atomics）和 Broadcom NIC 上首次实现 GPU-initiated token-level EP 通信，性能达 DeepEP 可比水平（NVIDIA-only）或更优（EFA 上优于 PPLX 2.1×）。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
- UCCL-EP Portable Expert-Parallel Communication
