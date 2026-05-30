## Expert Capacity / Capacity Factor / Token Dropping in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert capacity 是标准 MoE 实现中的核心约束概念：为满足 batched matrix multiplication 的等大小输入约束，限制每个 expert 最多处理的 token 数量。具体定义为：expert_capacity = (num_tokens / num_experts) × capacity_factor。capacity_factor 是一个超参数乘数：capacity_factor=1 表示每个 expert 刚好能处理均匀分配下的 token 数；capacity_factor>1 增加容量以降低丢 token 概率。Token dropping 发生在某 expert 被分配超过其 capacity 的 token 时——超出部分直接被丢弃，不参与 expert 计算，依赖 residual connection 传递信息（图 1）。为避免丢 token，Tutel 引入 dynamic capacity factor（运行时设为刚好不丢 token 的最小值），但 MegaBlocks 实验显示可能需要 capacity_factor 高达 11（Hwang et al. 2022），且 capacity_factor 尖峰在训练中不可预测地出现。

从算法pipeline角度拆解术语：
标准 MoE 的 token dropping/padding 流程：
```
输入: indices (num_tokens,)  # 每个 token 的 expert 分配
      capacity_factor       # 超参数
输出: padded_batches        # 可用于 batched GEMM 的 expert inputs

# 1. 计算 capacity
expert_capacity = ceil(num_tokens / num_experts * capacity_factor)

# 2. 按 expert 分组 tokens
for expert e in 1..num_experts:
    batch[e] = tokens[indices == e]

# 3. Dropping & Padding
for expert e in 1..num_experts:
    if len(batch[e]) > expert_capacity:
        batch[e] = batch[e][:expert_capacity]  # Truncate/DROP
        dropped_tokens += len(batch[e]) - expert_capacity
    elif len(batch[e]) < expert_capacity:
        batch[e].pad_to(expert_capacity)  # Zero-padding

# 4. Batched GEMM
# All experts computed with same batch size = expert_capacity
outputs = batched_gemm(batches, expert_weights)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 典型实现：GShard (Lepikhin et al. 2020) 引入 capacity_factor 概念；Switch Transformer (Fedus et al. 2022) 广泛使用 token dropping；Tutel (Hwang et al. 2022) 使用 dynamic capacity factor 在运行时自适应调整容量。
- 核心 tradeoff：(a) capacity_factor 小 → 更多 token 被丢弃 → 模型质量下降（MegaBlocks 实验显示 capacity_factor=1 时 loss 改善仅 0.15 vs dense，而不丢 token 改善 0.26）；(b) capacity_factor 大 → 大量 zero-padding → 计算和内存浪费（某些模型需要 capacity_factor 高达 11，MoE 层计算量增加 >2×）。
- MegaBlocks 通过 block-sparse 重表述从根本上消除 capacity_factor 参数和 token dropping/padding 问题。
- 在分布式 expert parallelism 中，token dropping 还影响 All-to-All 通信效率：不均匀的 expert 负载导致 straggler 问题。

- **Sub-sequence Dropping（子序列丢弃）**：MoE Parallel Folding 论文提出的一种 token dropping 优化策略。在进行 token dropping 决策时，仅基于当前 rank 处理的子序列（sub-sequence）的本地 logits 做决策，而非跨所有 rank 收集完整序列的 logits（full-sequence dropping）。这避免了 AllGather 通信开销。论文经验验证：sub-sequence dropping 不影响模型收敛（training/validation loss 曲线与 MCore v0.9 对齐）。对 token-dropless 训练范式（如 MegaBlocks），Dispatcher 直接按 expert 分配所有 token，无容量约束。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- Mixture-of-Experts with Expert Choice Routing
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

---
