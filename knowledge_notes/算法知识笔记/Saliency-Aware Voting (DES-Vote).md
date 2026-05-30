## Saliency-Aware Voting (DES-Vote)

术语解释
DES-Vote 是 DES 的 coreset selection 策略。所有并行 tokens 按加权 router saliency 投票选举共享 expert coreset，克服 DES-Seq 的两大局限：无法显式最大化 expert 共享，以及使用固定阈值 k 忽略 token 间 expert 重要性差异。

术语是什么？
DES-Vote 核心流程（Algorithm 3）：
1. **Mask**: 对每个 token 的 router logits，保留 local Top-K 权重，其余置零 → I_m
2. **Aggregate**: 跨所有 token 聚合加权投票 → V_i = Σ_{n=1}^N I_{m,n,i}，即 expert i 从所有 token 收到的总 saliency
3. **Select**: Top-M_core experts by total vote → C = TopK(V, M_core)

关键洞察来自 **Expert Importance Map**（Figure 4）：raw gating weights 与实际 expert 重要性高度相关，因此用 router scores 作为 voting weights 比 uniform voting 更有效。

DES-Vote 优于 DES-Seq 的原因：
- 解决局限 (1)：全局 voting → 自然形成跨 token 共识 → 最大化 expert 共享
- 解决局限 (2)：weighted voting → collective importance 自然决定保留哪些 expert → 自动处理 token 间 expert 重要性差异
- 连续 β → 绕过 DES-Seq 每 token 至少 1 expert 的下限 → 支持更小的 coreset

从算法pipeline角度拆解术语：
```
# DES-Vote Algorithm (Algorithm 3 from paper)
Input: I (router logits N×M), M_core (target coreset size), K (local top-k)
Output: C (coreset expert indices)

# Step 1: Keep only local top-K weights, mask others
I_m = zeros_like(I)
for n in 1..N:
    topk_indices = TopK_indices(I[n], K)  # per-token top-K
    I_m[n, topk_indices] = I[n, topk_indices]  # keep weights, rest = 0

# Step 2: Aggregate weighted votes across sequence
V = sum(I_m, dim=0)  # V ∈ R^M: total saliency per expert

# Step 3: Select top M_core experts by total vote
C = TopK_indices(V, M_core)  # C ⊂ {1..M}, |C| = M_core

# Subsequent Constrained Local Routing (same as DES):
for each token n:
    S_n = TopK(I[n, C], K)  # Route within C
    ...
```

M_core 由 budget factor β 控制：M_core = β × M。例如 β=0.15, M=128 → M_core=19。

实验证据：DES-Vote 在相同 coreset size 下比 DES-Seq 实现更高 Top-K recall（保留更多 ground truth expert selections）和更低 residual reconstruction loss。

术语一般如何实现？如何使用？
- 参数选择：β ∈ (0, 1]，需按模型/任务 tuning。典型配置：LLaDA2.0-Mini β=0.15, LLaDA-MoE-7B β=0.6
- Mask 操作是必需的：不去除 low-rank experts 的 noise weights 会降低 voting 质量
- 与 DES-Seq 对比：DES-Vote 在 Top-K hit rate、reconstruction loss、最终 accuracy 上全面优于 DES-Seq
- 适用场景：并行度越高（block size 大），DES-Vote 优势越明显（more tokens = better voting consensus）

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
