## Expert Co-activation Matrix (专家共激活矩阵)

术语解释
Expert Co-activation Matrix M_ℓ(i,j) 是 BuddyMoE 中量化 MoE expert 间功能关系的数据结构，记录在 profiling corpus 中 expert i 和 j 被同一 token 同时选中的频率。从 M 导出的条件共激活分布 q_{j|i} = M_ℓ(i,j) / Σ M_ℓ(i,j') 是 buddy selection 的基础。

术语是什么？
给定 layer ℓ 的 E 个 experts：对 profiling corpus 中的每个 token x，router 选中 top-k expert set S_ℓ(x)，对于 (i,j) ∈ S_ℓ(x)×S_ℓ(x), i≠j，M_ℓ[i][j]++。条件分布 q_{j|i} 量化"给定 expert i 被选中时 j 也同时被选中的条件概率"。高 q_{j|i} 暗示 i 和 j 处理相似的 token 子流形（功能相似）。关键实证性质：(1) A_ℓ(·) heavy-tailed——少数 popular expert 占多数激活；(2) q_{j|i} mass 集中在少数 peers——top-r peers (r≪E) 覆盖大量 co-activation；(3) layer-wise heterogeneity——早期层 diffuse，后期层 tight clusters。

从算法pipeline角度拆解术语：
```
M = zeros(E, E)
for x in profiling_data:
    S = TopK(Router(x), k)
    for i in S:
        for j in S, j != i:
            M[i][j] += 1
q[i][j] = M[i][j] / sum(M[i])
buddy_ranking[i] = argsort_descending(q[i])
B[i] = buddy_ranking[i][:t] where cumsum(q) >= alpha
```

术语一般如何实现？如何使用？
- 可选概率加权变体：Σ 𝟙{i,j∈S} · min(p̃(i), p̃(j)) 利用 router probability 作为 soft weight
- Laplace smoothing (M←M+ε) 防零概率，down-weight early warm-up steps 减冷缓存 artifact
- 是 buddy construction、expert pruning、expert merging 的通用预处理
- **CoE 的 intra-layer 视角**：CoE 中的 co-activation 矩阵记录同一 token 在不同 iteration（t vs t+1）的 expert 选择对，衡量 expert transition pattern。对角线低 = flowing nature（token 倾向于跨步切换 expert）；非对称分布 = role differentiation

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models
- Continual Pre-training of MoEs How robust is your router（CPT 分析：Granular PBTk MoE 在 CPT 过程中 layers 0-1 的 co-activation 变化最大，layer 18 出现一致的 spike。0% replay checkpoint 的 co-activation 变化最大且遗忘最严重 → 更显著的 co-activation 变化与更高遗忘相关。SB Granular MoE 在 pre-training 时 co-activation 高度集中在 expert 15，CPT 后分散化）

---
