## Usage-Frequency Weighted Expert Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Usage-Frequency Weighted Expert Merging 是以每个 expert 被 router 选中的相对频率作为合并权重的策略。M-SMoE 首次采用此策略但仅基于经验；MergeMoE 通过 Theorem 1 严格证明了在独立假设下（router logits 与 expert 输出独立），使用频率 f_j / Σ f_k 作为簇内权重是 Frobenius 输出误差下界的最优解。证明思路：目标函数 Σ f_j (v_i - e_j)^T W (v_i - e_j) 在每个 cluster 内是独立的二次函数，W = Y_0^T Y_0 为准正定矩阵，设 v_i[j] = f_j/Σf_k 使一阶导数为零，二阶导数 ≥0 保证全局最优。

从算法pipeline角度拆解术语：
```
// 统计频率: calibration 数据前向推理一次，f_i = count(expert_i 被 top-K 选中)/total
// 归一化权重: B_{ji} = f_j / Σ_{k∈C_i} f_k (和为1)
// 在 MergeMoE 框架中: T2/T3 列权重 = B_{ji} (式4), T1 单独最小二乘优化
// Theorem 1 证明: min Σ f_j(v_i-e_j)^T W(v_i-e_j) → v_i[j] = f_j/Σf_k 为全局最优
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 频率统计只需一次无梯度前向推理，开销极小。
- 理论局限：独立性假设在实际模型中可能不完全成立（router 输出与 expert 参数通过训练耦合），但实践效果良好。
- 其他权重方案：均匀权重（Average）、参数幅值加权、路由 logits 加权——MergeMoE 实验验证使用频率加权最优。
- 对比 M-SMoE：两者均使用频率加权，但 MergeMoE 提供了理论最优性证明。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---
