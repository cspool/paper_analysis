## Joint Selection Strategy for KV Cache Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Joint Selection Strategy 是 R-KV 提出的 KV cache token 保留策略，通过线性组合 importance score（I_i^h）和 redundancy score（R_i^h）来决定每个 token 是否保留：Z_i^h = λ·I_i^h − (1−λ)·R_i^h。这是 R-KV 区别于纯 attention-based 方法（如 SnapKV，仅使用 I_i^h）的核心机制。λ∈[0,1] 控制两项目标的权衡：(a) 高 I_i^h → token 对后续解码重要，应保留；(b) 高 R_i^h → token 语义与大量其他 token 相似（冗余），应淘汰。两者通过相减在 joint score 中融合：当 token 重要性高且冗余性低时 Z_i^h 最高，最优先保留；当 token 重要性低且冗余性高时 Z_i^h 最低，最优先淘汰。

λ 的选择关键：R-KV 消融实验（§5.1）发现 I_i^h 分布极度稀疏（少数 outlier 主导），而 R_i^h（经 softmax 归一化）分布相对均匀。λ=0.1 时 redundancy 项的权重 (1−λ)=0.9 足以有效抑制冗余，同时 λ=0.1 的 importance 项保证 attention sink/初始 token 不被错误淘汰（λ=0 时初始四个 token 不保证保留，会严重损害生成能力，如 Fig. 5 所示）。λ≥0.5 后 selection 退化为近似纯 attention-based。最优 λ∈[0.01, 0.1]，论文所有实验使用 λ=0.1。

从算法pipeline角度拆解：

```
# Joint Selection 跨head聚合流程
# 输入: I ∈ R^{H × N_c} (per-head importance scores)
#       R ∈ R^{H × N_c} (per-head redundancy scores)
# 参数: λ=0.1

for h in range(H):  # 每个attention head
    for k in range(N_c):  # 每个候选token
        # 线性组合
        Z[h][k] = λ * I[h][k] - (1-λ) * R[h][k]

# 跨head聚合 → 均值
AggScore[k] = (1/H) * Σ_h Z[h][k]  for k in 0..N_c-1

# Top-B_budget选择
selected_indices = ArgSort(AggScore, descending=True)[:B_budget]
```

R-KV 的跨 head 聚合使用 mean（而非 max 或 sum），确保每个 head 的 joint score 对最终选择有均等贡献（因不同 head 可能关注不同类型的 token——某些 head 关注语法、某些关注语义、某些关注 attention sink）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Joint Selection 的核心实现是纯 PyTorch tensor 操作：per-head Z = λ*I - (1-λ)*R 用 element-wise arithmetic；跨 head 聚合用 tensor.mean(dim=head_dim)；Top-K 用 torch.topk。计算开销为每 compression step O(H·N_c) 的标量操作，相比 attention 计算 O(α·N_c·d) 和 similarity matrix O(N_c²·d) 可忽略不计。Joint Selection 的 λ 选择需针对不同模型/数据集做 sensitivity analysis——不同模型、不同数据集的 attention score 稀疏度和 redundancy 分布可能不同。R-KV 建议从 λ=0.1 出发做 grid search over {0.01, 0.05, 0.1, 0.5, 1.0}。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
