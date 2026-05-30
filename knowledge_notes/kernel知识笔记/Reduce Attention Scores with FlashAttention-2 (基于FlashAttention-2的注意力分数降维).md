## Reduce Attention Scores with FlashAttention-2 (基于FlashAttention-2的注意力分数降维)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Reduce Attention Scores 是 NACL 实现的与 FlashAttention-2 兼容的 CUDA kernel，用于在 encoding 阶段高效计算 per-key 的累积 attention scores（column-wise reduced attention），作为 KV Cache 淘汰的 token 重要性评分依据。

FlashAttention-2 forward 为节省显存不将完整 attention matrix S ∈ R^{N_q×N_k} 写入 HBM，而是分 tile 在 SRAM 中计算。但 KV Cache 淘汰需要 per-key 累积 attention scores。该 kernel 利用 FlashAttention-2 forward 输出的 log-sum-exp (LSE) vector L ∈ R^{N_q}，按 backward pass 方式重算 attention matrix 并做 column-wise sum。

两种实现：(1) 完整重算——每个 (Q_i, K_j) tile 重算 S → P = exp(S-L) → column-wise reduce；(2) 小矩阵重算——仅对 proxy tokens（~10% of N_q）重算，开销可忽略。

从kernel调度角度拆解术语：

```
输入: Q∈R^{N_q×d}, K∈R^{N_k×d}, L∈R^{N_q}(FA2 LSE), B_c,B_r
输出: O∈R^{N_k}(per-key reduced scores)

Step 1: T_r=ceil(N_q/B_r), T_c=ceil(N_k/B_c), O=zeros(N_k)
Step 2: for j=1..T_c:
   Load K_j from HBM→SRAM, R_j=zeros(B_c)
   for i=1..T_r:
     Q_i,L_i from HBM→SRAM
     S_i^{(j)} = Q_i@K_j^T                       # on-chip
     P_i^{(j)} = exp(S_i^{(j)} - L_i)            # online rescale
     R_j += columnwise_sum(P_i^{(j)})            # reduce over query dim
   atomicAdd(O_j, R_j)
```

复杂度与 FA2 同阶 O(N_q·N_k·d)。小矩阵方式仅 O(|P|·N_k·d)，~10× 加速。

术语一般如何实现？如何使用？

基于 FlashAttention-2 的 CUDA tiling 实现。NACL 128K context 下 evict 20% 维持 ~15GB 稳定显存。小矩阵方式可纯 PyTorch 实现（仅对 proxy tokens 做 matmul + softmax + sum）。代码：https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---
