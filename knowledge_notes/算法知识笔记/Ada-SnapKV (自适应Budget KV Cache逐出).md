## Ada-SnapKV (自适应Budget KV Cache逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Ada-SnapKV（Feng et al., 2024）是 SnapKV 的增强版，核心创新在于**跨 head 的动态 token budget 分配**：不同于 SnapKV 给所有 attention head 分配相同数量的 KV token，Ada-SnapKV 允许各 head 获得不同数量的 token——关键 head 保留更多 token，非关键 head 保留更少，在相同总 budget 下提高信息保留率。

Sparse Frontier 的实现使用 **max-aggregation**（而非 SnapKV 的 mean-aggregation）跨 query positions 和 heads 进行分数计算，经验证明这对自适应分配更有效（但对均匀分配无影响）。每 head 最低 budget 设为 20% 容量——消融显示 10-50% 范围内性能良好，但接近 100%（等价均匀 SnapKV）时退化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Ada-SnapKV prefill阶段 KV 选择（自适应 budget 版本）
Input: Q, K, V, total_budget=2048, n_heads=32, min_ratio=0.2

# Step 1: 使用 observation window 计算 per-head per-token importance
for each head h:
    Q_obs = Q[h, -256:, :]                         # observation window
    attn_h = Q_obs @ K[h]^T / sqrt(d)              # [256, S]
    imp_h = max(attn_h, dim=0)                      # max-aggregation [S]
    imp_h = AvgPool1d(imp_h, kernel_size=21)         # smoothing

# Step 2: 跨 head 聚合得全局 importance（用于 adaptive budget 分配）
global_imp = max_pool_over_heads(imp_1..imp_H)      # [S]

# Step 3: Adaptive budget 分配
for each head h:
    # 该 head 对 top 全局重要 token 的覆盖度决定 budget
    overlap = intersection(topk_global_indices, topk_h_indices)
    budget_h = max(total_budget * 0.2,  # 最低 20%
                   total_budget * |overlap|/total_budget)
    budget_h = min(budget_h, total_budget * n_heads)  # 上限

# Step 4: 每 head 独立选择 TopK
for each head h:
    selected_h = sort(preserved ∪ TopK(imp_h, budget_h))
    K_compress[h] = K[h, selected_h]
    V_compress[h] = V[h, selected_h]
```

术语一般如何实现？如何使用？

Ada-SnapKV 开源（MIT 许可证）。与 SnapKV 共享 infrastructure，差异仅在于 budget 分配策略。使用方式：设置 token_capacity（同 SnapKV）、min_budget_ratio=0.2、kernel_size=21、observation_window=128。Sparse Frontier 评估表明 Ada-SnapKV 始终优于均匀 SnapKV（尤其 multi-query 任务），但两者均弱于 Quest（full-cache 方法）因 eviction 的不可逆信息损失。推荐在内存受限场景（无法保留全 KV cache）使用。

涉及论文标题：
- Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
