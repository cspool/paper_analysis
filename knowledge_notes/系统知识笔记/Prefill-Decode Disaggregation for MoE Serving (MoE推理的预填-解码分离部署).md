## Prefill-Decode Disaggregation for MoE Serving (MoE推理的预填-解码分离部署)

术语解释
Prefill-Decode Disaggregation 是一种 MoE 大模型推理部署策略，将 prefill 阶段（处理 prompt，compute-bound）和 decode 阶段（逐 token 生成，memory-bound）部署到不同 GPU 集群上，分别针对各自的瓶颈进行独立优化。DeepSeek-V3 是首个在 MoE 模型上系统描述 prefill-decode 分离部署生产实践的工作。

术语是什么？
DeepSeek-V3 的部署策略：(1) **Prefill 集群**：最小部署单元 4 nodes × 8 H800 = 32 GPUs。Attention 部分使用 TP4（Tensor Parallelism 4）+ SP（Sequence Parallelism）+ DP8（Data Parallelism 8）；MoE 部分使用 EP32（Expert Parallelism 32）。冗余专家：32 个额外的 expert 副本。(2) **Decode 集群**：最小部署单元 40 nodes × 8 H800 = 320 GPUs。Attention 部分使用 TP4+SP+DP80；MoE 部分使用 EP320，每个 GPU 仅承载 1 个 expert，64 个 GPU 专门承载冗余专家和共享专家。All-to-all 通信通过 direct point-to-point IB 传输（IBGDA 降低延迟）。(3) **Micro-batch 双流水线**：Prefill 重叠 attention+MoE of batch-A 与 dispatch+combine of batch-B；Decode 重叠 attention of batch-A 与 dispatch+MoE+combine of batch-B（decode 阶段 attention 占主导）。

从系统架构角度拆解术语：
```
=== DeepSeek-V3 Prefill-Decode 分离部署流程 ===

┌─────────────────────────────────────────────────────────┐
│                    Prefill Cluster (32 GPUs)              │
│  TP4+SP+DP8 (Attention) + EP32 (MoE)                    │
├─────────────────────────────────────────────────────────┤
│  Input: User prompt tokens [t1, ..., tn]                 │
│                                                          │
│  // Micro-batch double pipeline                         │
│  Batch-A: [Attn layers]  [Dispatch][MoE layers][Combine] │
│  Batch-B:                [Attn layers]  [Dispatch]...    │
│           ↑ overlap ↑                                    │
│                                                          │
│  Redundant experts: 32 extra replicas of hot experts     │
│  Routing: K_r=8, node-limited M=4                        │
│                                                          │
│  Output: KV cache (c^{KV} + k^R, FP8/E5M6)              │
└──────────────────────┬──────────────────────────────────┘
                       │ IB transfer KV cache
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Decode Cluster (320 GPUs)              │
│  TP4+SP+DP80 (Attention) + EP320 (MoE)                  │
├─────────────────────────────────────────────────────────┤
│  // Autoregressive generation loop                       │
│  for step in 1..max_tokens:                             │
│      // Micro-batch overlapping                         │
│      Batch-A: [Attention (dominant)]                     │
│      Batch-B: [Dispatch][MoE][Combine] (fewer SMs)       │
│      // Shared expert → treated as always-selected       │
│      // 9 experts total activated per token              │
│      // All-to-all via direct P2P IB (IBGDA)             │
│                                                          │
│  MTP speculative decoding (optional):                   │
│      Accept rate: 85-90% → 1.8× TPS                      │
└─────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？
Redundant expert 检测周期：每 10 分钟基于 online deployment statistics 更新。Prefill: 32 redundant experts，每个 GPU 除 8 个 original experts 外额外承载 1 个 redundant expert。Decode: 64 GPUs 专用于 redundant+shared experts。正在探索 dynamic redundancy：每 GPU 承载 16 experts 但仅动态激活 9 个，all-to-all 前实时计算全局最优路由。Decode 阶段 batch size per expert 较小（≤256 tokens），bottleneck 在 memory access 而非 computation，因此可分配较少 SMs 给 MoE 部分。

涉及论文标题：
- DeepSeek-V3 Technical Report
