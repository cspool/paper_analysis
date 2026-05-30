## Minimum Threshold Prediction (DP-based)（基于动态规划的最小阈值预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Minimum Threshold Prediction 是 XAttention 的第三个组件（可选），通过动态规划（Dynamic Programming）为每个注意力头离线搜索最优的稀疏阈值 τ_h。不同头有不同的稀疏度-准确率特性，统一阈值无法充分利用这种差异性。

问题建模：对 H 个注意力头，DP table D[h][m] 表示在前 h 个头中进行了 m 次阈值调整时的最佳性能。递推关系：$D[h][m] = \max(D[h-1][m], P(h, m))$，其中 P(h,m) 是将第 h 个头的阈值降低一步后的模型性能。

从kernel调度角度拆解术语：

```
# Minimum Threshold Prediction 动态规划
Input: H attention heads, max adjustments M, base τ_0 = 0.9
Output: per-head optimal thresholds τ*_h

# Step 1: Initialize
D = zeros(H+1, M+1)  # DP table
for h in 1..H:
    for m in 1..M:
        # Option 1: skip head h (keep current threshold)
        skip = D[h-1][m]
        # Option 2: adjust head h threshold down by 10%
        # t_h(m) = t_h(m-1) * 0.9
        adjust = evaluate_performance(h, threshold=t_h(m-1)*0.9)
        D[h][m] = max(skip, adjust)

# Step 2: Backtrack to get per-head thresholds
τ* = backtrack(D, H, M)

# Step 3: Runtime use
# 每个 head h 在推理时使用 τ*_h 而非统一 τ
# 推理过程不变——仅 τ 值不同
```

搜索结果：从 τ_0=0.9 开始，M=1000 步搜索后，平均阈值降至 0.8。该平均阈值在 RULER benchmark 上同时实现更低密度（S=4: 21.09% vs 23.06%）和更高准确率。

术语一般如何实现？如何使用？

这是离线（offline）优化过程，在模型部署前执行。实现方式：使用校准数据集（如 RULER 子集）迭代评估不同 τ 组合下的模型准确率，利用 DP 避免穷举搜索（M=1000 时穷举 H^{1000} 不可能，DP 将复杂度降至 O(H×M)）。搜索出的 per-head τ 值保存为配置文件，推理时直接加载使用。论文强调这是可选组件——不使用 DP 而使用固定 τ=0.9 也能取得良好效果。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring
