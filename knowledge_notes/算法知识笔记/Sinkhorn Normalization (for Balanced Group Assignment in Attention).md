## Sinkhorn Normalization (for Balanced Group Assignment in Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sinkhorn Normalization 是一种将任意非负矩阵迭代变换为双随机矩阵（doubly stochastic matrix）的算法，在 Focus 论文中被用于强制注意力分组均衡。算法流程：给定 token-group 得分矩阵 S ∈ R^{T×K}（T 个 token，K 个 group），首先 Q ← exp(S/τ)（temperature τ 控制软硬程度），然后交替进行行归一化（Q ← Q / sum(Q, dim=tokens)，使每个 token 的总 assignment 为 1）和列归一化（Q ← Q / sum(Q, dim=groups)，使每个 group 的总 mass 均衡），迭代 N 次后 Q 近似双随机——所有行和列的和均为 1。与 softmax 归一化（仅行归一化，无列约束）不同，Sinkhorn 阻止任何单个 group 吸收所有 token（group dominance），同时仍允许 LM 梯度学习哪个 token 属于哪个 group。在 Focus 中 N=10 次迭代足以产生平衡分组，τ=0.1 控制 assignment 的软硬程度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Focus 的每个 attention 层，Sinkhorn 归一化作为 group assignment 的核心步骤，位于 token-to-centroid 亲和度计算之后、门控注意力之前：

```
# 输入: h [T, d] hidden states, C [K, d_g] centroids, W_g [d, d_g]
# 输出: g [T, K] doubly-stochastic group assignments

def sinkhorn_group_assignment(h, C, W_g, tau=0.1, N=10):
    # Step 1: 投影到 centroid 空间
    S = (W_g @ h.T).T @ C.T    # [T, K] token-centroid 亲和度得分
    
    # Step 2: Sinkhorn 迭代
    Q = exp(S / tau)            # [T, K] 指数化
    for i in range(N):
        Q = Q / Q.sum(dim=0, keepdim=True)    # 列归一化: 均衡 group mass
        Q = Q / Q.sum(dim=1, keepdim=True)    # 行归一化: 每个 token sum=1
    
    # Q 现在是近似双随机矩阵
    # 每行: token i 对各 group 的软分配
    # 每列: 各 group 的 token 质量均衡 (≈T/K)
    return Q  # [T, K]
```

Pipeline 中 Sinkhorn 的位置：
1. token hidden states → W_g 投影 → centroid 空间 (d_g=16)
2. 计算 token-centroid 亲和度得分 S
3. **Sinkhorn 归一化** → 双随机 group assignment g
4. g 用于门控注意力: s_ij = q_i^T k_j · (1_local + (1-1_local) · σ(λ · g_i^T g_j))
5. 仅同组远距离 token 参与注意力

Sinkhorn 阻止三条 group dominance escape pathway:
- Path A (centroid drift): 即使 centroid 漂移导致所有 token 偏向同一 centroid，列归一化强制重新分配
- Path B (representational bypass): 即使 hidden states 偏移，行归一化保持 per-token assignment 分布
- Path C (projection bypass): 即使 W_g 映射所有 token 到同一方向，双随机约束仍强制均衡

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sinkhorn 归一化最初来自最优传输（optimal transport）理论，用于高效近似计算 Wasserstein 距离。在深度学习中的使用方式：
- Focus 论文中实现为 Python/PyTorch，每次前向传播在 attention 层内执行 10 次迭代的行/列归一化
- 温度 τ 控制 assignment 的置信度：τ=0.1 时 assignment 接近 hard（高置信度），τ 增大则趋于均匀
- N=3 次迭代不足以均衡（尤其在低 τ 下 exp(scores/τ) 分布极尖锐），论文推荐 N≥10
- 与 softmax + entropy loss 方法对比：Sinkhorn 是结构约束（非软损失），因此不依赖梯度来学习均衡——即使梯度推动 collapse，Sinkhorn 迭代仍强制重新分布
- 超参稳健性：Table 9 显示 fine-tuned PPL 在 16 种配置下仅波动 0.6

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)
