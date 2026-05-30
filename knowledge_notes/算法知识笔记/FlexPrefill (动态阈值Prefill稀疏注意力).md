## FlexPrefill (动态阈值Prefill稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FlexPrefill（Lai et al., ICLR 2025）是一种针对 prefilling 阶段的 training-free 稀疏注意力方法。其核心创新在于**阈值驱动（threshold-based）的动态 budget 分配**：不同于 Vertical-Slash 给每个 head 分配固定数量的 vertical columns 和 slashes，FlexPrefill 通过设置 coverage 参数 α（如 α=0.8 表示覆盖 80% attention mass）和一个 min_budget 参数，让每个 head 自动决定需要保留多少 QK 交互对来达到目标 coverage。当动态分配在高稀疏度下失效时，回退到 α=0（等价于均匀 Vertical-Slash 分配）。

关键机制：(1) 首先用近端 query window（256/512 tokens）估计注意力分布；(2) 对每个 head，按 attention score 降序选择 top tokens 直至累积覆盖率 ≥ α；(3) 每 head 最低保留 min_budget 个 tokens 作为连通性保证。Sparse Frontier 实验发现：(a) FlexPrefill 在多数任务中 matching 或略低于 Vertical-Slash 的均匀分配——threshold-based 选择捕获高 attention token 但漏掉 attention 分布长尾中的重要信息（"attention sink phenomenon"效应）；(b) min_budget=512 显著改善性能；(c) 高压缩比下动态分配失效需回退。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# FlexPrefill prefilling 稀疏注意力
Input: Q, K, V ∈ R^{S×d_h}, α=0.7, min_budget=512, window=256

# Step 1: 使用近端 query 窗口估计注意力
Q_recent = Q[-window:, :]
S_approx = Q_recent @ K^T / sqrt(d)             # [window, S]

# Step 2: 沿 query 维度聚合得 per-token importance
importance = S_approx.sum(dim=0)                 # [S]
importance = softmax(importance)                  # normalize to dist

# Step 3: 保留 attention sinks (prefix + local)
preserved = [0:4] ∪ [S-64:S]                     # 固定保留

# Step 4: Threshold-based 动态选择（关键差异）
sorted_imp = sort(importance[4:S-64], descending=True)
cumsum = cumsum(sorted_imp)
num_selected = max(min_budget, argmin(cumsum >= α))  # 至少min_budget
i_vs = top_indices(sorted_imp, num_selected)

# Step 5: 稀疏 attention 计算（仅 selected QK pairs）
O = sparse_attention(Q, K, V, indices = preserved ∪ i_vs)
```

术语一般如何实现？如何使用？

FlexPrefill 开源（Apache-2.0）：https://github.com/xxxx。实现基于 Vertical-Slash 基础设施，增加 coverage-based selection 逻辑。配置参数：(α, min_budget)——低 α 等价均匀分配，高 α 保留更多 attention mass。Sparse Frontier 推荐使用场景：中等稀疏度（0.5-0.7），min_budget=512，此时动态分配略有优势。高稀疏度下建议退回到均匀 Vertical-Slash（α=0）。

涉及论文标题：
- FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
