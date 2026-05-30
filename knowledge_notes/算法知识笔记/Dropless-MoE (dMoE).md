## Dropless-MoE (dMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dropless-MoE (dMoE) 是 MegaBlocks (MLSys 2023) 提出的 MoE 层计算方法，核心思想是将 MoE 层的 expert 计算从 batched matrix multiplication 重新表述为 block-sparse matrix multiplication，从而**从根本上消除 token dropping** 的需求。传统 MoE 实现（GShard, Switch Transformer, Tutel）为了满足 batched GEMM 的形状约束（要求所有 expert 分配相同的 token 数量），在 token 分配不均衡时强制丢弃超出 expert capacity 的 token 或 zero-padding 不足的 expert batch。dMoE 将 expert 计算视为 variable-size block diagonal matrix multiplication（图 3C）：每个 expert 的 token batch 被分解为多个 128×128 固定 block，仅计算实际分配的 token 行（sparse non-zero blocks），天然支持负载不均衡的 token 分配。dMoE 从算法层面消除了 capacity_factor 超参数和 token dropping/padding 的 tradeoff，已被用于训练 Mixtral 8×7B 和 DeepSeek V2 等模型。

从算法pipeline角度拆解术语：
dMoE 的 forward pass（图 4）：
```
输入: x (num_tokens, hidden_size)
输出: y (num_tokens, hidden_size)

# (1) Router: Assign tokens to experts (与标准 MoE 相同)
indices, weights = router(x)  # top-k greedy selection

# (2) 构造 block-sparse matrix topology（关键差异）
# 将 variable-size expert batches 分解为 128×128 blocks
topology = make_topology(indices)
# topology 描述图 3C 的 block-sparse matrix:
#   - row_offsets[i]: expert i 的 blocks 在 non-zero list 中的起始偏移
#   - column_idxs[b]: block b 对应的 expert (决定使用 w1 的哪一列)
#   - row_idxs[b]: block b 在输出中的行坐标 (用于 SDD)

# (3) 按 expert 分组 tokens + padding 到 128 倍数
x_permuted = padded_gather(x, indices)  # (total_tokens_padded, hidden_size)

# (4) Expert 计算: Sparse = Dense × Dense (第一层)
# w1.shape: (hidden_size, ffn_hidden_size * num_experts)
intermediate = sdd(x_permuted, w1, topology)  # block-sparse output

# (5) Dense = Sparse × Dense (第二层)
# w2.shape: (ffn_hidden_size * num_experts, hidden_size)
y_permuted = dsd(intermediate, w2)  # dense output

# (6) Un-permute + scaling
y = padded_scatter(y_permuted, indices)
return y * weights
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- dMoE 通过自定义 block-sparse GPU kernels 实现（§5.1）：扩展 CUTLASS 2.5 实现 SDD、DSD、DDS 操作，使用 Hybrid Blocked-CSR-COO 编码（§5.1.3）和 Transpose Indices（§5.1.4）。
- 开源实现：https://github.com/databricks/megablocks (Apache-2.0)，通过 `pip install megablocks` 安装。集成于 Megatron-LM，支持 data/expert/pipeline parallelism。
- 两种计算后端：Sparse MLP（block-sparse via STK，Ampere GPU A100）和 Grouped MLP（grouped GEMM，Hopper GPU H100 推荐）。
- 已被工业界广泛采用：Mistral AI 的 Mixtral 8×7B 训练使用 MegaBlocks，vLLM 集成 MegaBlocks 进行 MoE 推理，DeepSeek V2 训练也基于此技术栈。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 从不同角度实现 dropless MoE——通过轻量级索引数据结构（expert_token_indices, token_expert_indices 等）替代 per-expert materialized buffer，天然支持 variable-length expert batches（每个 expert 处理的 token 数量任意，由实际路由决定而非固定 capacity 限制）。与 MegaBlocks 的 block-sparse 方法（将 variable-size batch 分解为 128×128 fixed blocks）不同，MoEBlaze 的索引方法无需 block decomposition 和 padding。

---
