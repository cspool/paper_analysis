## Mixture of Block Attention (MoBA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MoBA（Mixture of Block Attention）是一种将 Mixture of Experts (MoE) 的专家路由原理从 FFN 层迁移到 attention 层的稀疏注意力架构。核心思想：将长上下文的 Key-Value 序列划分为等大小的 blocks，每个 query token 通过 parameter-free gating（query 与 mean-pooled K per block 的内积）计算与每个 block 的 affinity score，再用 top-k gating 选择最相关的 k 个历史 blocks 进行 attention 计算。MoBA 与 full attention **参数等价**（0 参数增量），支持训练中 MoBA↔Full Attention 无缝切换。

从算法pipeline角度拆解术语：MoBA 位于 Transformer 的 attention 层，直接替换标准 scaled dot-product attention，不影响其他层（FFN、LayerNorm）。计算流程为 Block Partitioning → Mean Pooling Key Representation → Gating Scores → Causal Top-k → Block-wise FlashAttention Varlen → Online Softmax Combining。

具体例子（1M context prefill, B=4096, k=12）：
```
输入：Q, K, V ∈ R^{N×h×d}, N=1M, h=32, d=128
n = N/B = 1M/4096 ≈ 244 blocks

# Step 1-2: Block partition + mean pool
K̄ = mean_pool(K.reshape(n, B, h, d), dim=1)  # [n, h, d] = [244, 32, 128]

# Step 3: Gating scores
S = einsum('nhd,mhd->nhm', Q, K̄)  # [1M, 32, 244], O(N·n·d) ≈ 31G FLOPs

# Step 4: Causal mask + topk
M[pos, :, i] = -inf if pos < i*B  # mask future blocks
G = topk(S + M, k=12, dim=-1)    # [1M, 32, 12] sparse indices

# Step 5: Block-wise varlen FA
# Current block (always causal): 1 block × 4096 tokens = 4K tokens
# Selected history blocks: 12 blocks × 4096 tokens = 49K tokens
# Total attention tokens = 53K per query (5.3% of 1M)

# Step 6: Online softmax combine
lse_s, lse_m = logsumexp from each partial attention
O = (exp(lse_s-lse_total)·O^s + exp(lse_m-lse_total)·O^m)
```
复杂度从 O(N²·d) 降至 O(k·B·N·d)，sub-quadratic。Sparsity = 1-(k+1)B/N = 94.7%。

术语一般如何实现？如何使用？

基于 PyTorch + FlashAttention + DeepSpeed-MoE 实现。开源：https://github.com/MoonshotAI/MoBA。核心 CUDA-level 优化包括：block-split + index_select（MoE-style token dispatch）、varlen FlashAttention（不同 block 的 query 数量不同）、online softmax combine（tiling 保证数值等价）。已部署于 Kimi 长上下文请求。适用于 continued pre-training 扩展已有模型 context length（如 Llama-8B→1M）。典型超参：block_size=4096, top-k=12, layer-wise hybrid（最后 3 层 full attention）。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs
