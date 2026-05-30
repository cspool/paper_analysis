## Intra-Sequence Sharing (DES-Seq)

术语解释
DES-Seq 是 DES 的一种直接 coreset selection 策略：对每个并行 token 取 Top-k 个最 salient experts（k < K），取所有 token 的并集作为共享 coreset。

术语是什么？
DES-Seq 的 coreset 构建：C_DES-Seq = ∪_{n=1}^N TopK(I_n, k)，其中 k 是超参数（满足 k < K，K 为 vanilla routing 的每 token expert 数）。这一策略最早在 AR 模型的 batch-level optimization（OEA, Oncescu et al., 2025）中探索，DES 将其适配到 dLLM 的 intra-sequence level。

从算法pipeline角度拆解术语：
```
# DES-Seq Algorithm (Algorithm 2 from paper)
Input: I (router logits N×M), k (local selection count, k < K)
Output: C (coreset expert indices)

C = ∅
for n = 1 to N:
    topk_n = TopK_indices(I[n], k)  # per-token top-k experts
    C = C ∪ topk_n                   # union across tokens

return C
```

局限性：
1. **不显式最大化 sharing**：仅减少 local budget（k < K），不寻求跨 token 共识 → 可能产生低效 coreset
2. **固定 k 忽略 expert 重要性差异**：第 2 名 expert 对 token A 可能比对 token B 关键得多，uniform k 无法捕捉

术语一般如何实现？如何使用？
- k 值：典型配置 k=2, k=3（vs vanilla K=8）
- coreset size 下限：每 token 至少 1 expert，即 |C| ≥ 1（当所有 token 共享同一 top-1 expert 时）
- 与 DES-Vote 对比：DES-Seq 在 accuracy-efficiency Pareto frontier 上处于 DES-Vote 之下
- 适用场景：当 voting overhead 需要避免时作为简单 baseline；极低 latency 场景
- k=2 配置（LLaDA2.0-Mini）：unique experts 84→34 (-60%), relative accuracy 95.7%

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
