## Dropless Routing (无丢弃路由 / Capacity-Free MoE Routing)

术语解释
Dropless Routing 是一种 MoE token 路由策略，在 token-to-expert 分配过程中不通过 capacity factor 强制丢弃超出 expert 容量上限的 token。与标准 GShard 风格 routing（设置 capacity factor，超限 token 被 drop 或通过 residual connection 绕过 expert）相反，dropless routing 保证每个 token 都被其选中的 top-K expert 处理，无需 token dropping。

术语是什么？
传统 Token-Choice Routing（GShard, Switch Transformer）为每个 expert 设置 capacity = capacity_factor × (total_tokens / N_E)，当路由到某 expert 的 token 数超过 capacity 时，多余的 token 被丢弃（不经过该 expert 计算，仅通过 residual connection 传递）。这虽然保证了计算负载可控（无 straggler），但丢弃 token 会损失信息。

Dropless Routing（由 MegaBlocks/dMoE, Gale et al., 2023 引入）移除 capacity 限制，所有 token 都被其选中的 expert 处理。实现方式不是通过 padding（padding 会导致无效计算），而是通过 block-sparse matrix multiplication——将 t token × E expert 的分配矩阵按固定 block size 划分，仅计算非空 block，避免冗余 FLOPs。

从算法pipeline角度拆解术语：
```
# Standard Routing with Capacity Factor (GShard)
for each expert e:
    capacity_e = capacity_factor * total_tokens / N_E
    tokens_e = argtopk(gate_scores, k=min(topK, capacity_e))
    # 超出 capacity 的 token 被丢弃
    output[tokens_e] += expert_e(tokens[tokens_e])

# Dropless Routing (dMoE / MegaBlocks)
# 无 capacity 限制，依赖 block-sparse GEMM
token_expert_map = topK(gate_scores, k)  # 所有 token 都被路由
# 使用 block-sparse matrix multiply: [t, h] x [E, h, 4h]
# 仅计算非零 block，避免 padding 开销
output = block_sparse_gemm(input, experts, token_expert_map)
```

术语一般如何实现？如何使用？
- **MegaBlocks (dMoE)**：Gale et al. (2023) 的 dropless MoE 实现，通过自定义 block-sparse GEMM kernel 高效执行 token-expert 间的稀疏计算。训练时无需 capacity factor 调参，token 不被丢弃。
- **Demons in the Detail (Qiu et al., 2025)**：论文采用 dropless routing 策略（类似 dMoE），以避免 token drop 对不同 Balance BSZ 方法的影响相互混淆。由于使用 dropless 策略，不同 Balance BSZ 设置间的 FLOPs 计算量一致（所有 token 都被处理），但 global-batch balance 可能导致局部负载不均（某些 GPU 处理远超均值的 token 数），引起 ~5.8% 的速度下降。通过额外加微量 micro-batch LBL 可缓解。
- **与 capacity-based routing 的权衡**：Dropless routing 避免了信息损失（无 token 被丢弃），但可能在某些 GPU 上产生局部计算热点，导致训练速度波动（straggler effect）。Capacity-based routing 保证计算负载可预测（有利于 Expert Parallelism），但牺牲信息完整性。

涉及论文标题：
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 dropless MoE 训练，基于 gpt-neox + MegaBlocks 实现）
