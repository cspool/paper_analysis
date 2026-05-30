## Soft Expert-Tensor Parallelism (S-ETP)

术语解释
Soft Expert-Tensor Parallelism (S-ETP) 是通过算法层面的 expert partition（partial transformation）实现 TP-like 效果的并行策略，替代传统系统层面的 Expert-Tensor Parallelism (ETP)。S-ETP 将 ETP 中多轮通信（"AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll"）简化为单次 AlltoAll，减少 kernel launch 和同步开销。

术语是什么？
传统 ETP 在 EP 基础上叠加 TP 对 expert weight 做跨 GPU 分片，通信模式为两阶段 AlltoAll + AllGather（或 ReduceScatter + AlltoAll）。S-ETP 从算法层面将每个 original expert 用 partial transformation 分为 P 个 sub-experts，配合 EP 实现等价 TP 效果：weight 分片由分片后的 sub-expert 自然实现（而非系统层面切分 weight 参数）。通信仅需一次 AlltoAll 做 token dispatch + result gather，消除了 AllGather/ReduceScatter 的额外 round-trip、kernel launch 和 synchronization barrier。

从系统架构角度拆解术语：
```
=== ETP (Traditional Expert-Tensor Parallelism) ===
Configuration: EP=4, TP=2, 8 GPUs
Communication per layer:
  Step 1: AlltoAll — dispatch tokens to expert-owning GPUs
  Step 2: AllGather or ReduceScatter — share/reduce within TP groups
  Step 3: FFN computation on sliced weights
  Step 4: AllGather or ReduceScatter — reconstruct full outputs
  Step 5: AlltoAll — return results to token-originating GPUs
  Total: 3+ collective operations

=== S-ETP (Soft Expert-Tensor Parallelism, DualSparse-MoE) ===
Configuration: EP=8 (4 original experts × P=2 partition), 8 GPUs
Communication per layer:
  Step 1: AlltoAll — dispatch tokens (single operation!)
  Step 2: FFN computation on partitioned experts (no TP slicing needed)
  Step 3: AlltoAll — return results (single operation!)
  Total: 2 collective operations

Bandwidth improvement (real H20, EP=4 TP=2): 3.0-29.9%
Bandwidth improvement (simulated NVL72, EP=9 TP=8): 10.2-80.4%
```

术语一般如何实现？如何使用？
- Partial Transformation as foundation：将每个原 expert 均分为 P 个 sub-experts，保持 gating network 不变，仅重映射 expert indices
- 通信优化：S-ETP 仅需 AlltoAll（vs ETP 的 "AlltoAll+AllGather" 双阶段），减少 collective ops 和 barrier sync
- 适用场景：(a) 原使用 ETP (EP+TP) 的部署场景直接受益；(b) 需要 scale up EP 的场景——partition 产生更多 experts 可分布到更多 EP devices；(c) 全互联高带宽系统（NVL72、CloudMatrix384）受益最大（因消除了 inter-node bandwidth bottleneck）
- 实现位置：SGLang framework 的 EP 通信模块 + expert partition 预处理
- 局限性：(a) 需要 partial transformation 作为前置步骤；(b) gating scores 重复处理增加 trivial overhead；(c) 过细 partition (P>>2) 的 communication benefit 递减

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
