## Dare Merging for MoE (Drop and Rescale)

术语是什么？
Dare（Drop and Rescale, Yu et al. 2024）通过随机 drop + rescale 解决参数干扰：(1) 随机将 task vector 中 (100-p)% 参数置零；(2) rescale 保留参数 × 1/(0.01·p) 补偿幅值损失；(3) 求和得到 τ_m。MergeME 首次将 Dare 从 dense 模型合并扩展到 MoE 合并——仅对非 FFN 层应用 Dare，FFN 层保持独立。MergeME 设置 p=80%, λ=1/3。Table 1: Dare merging MoE avg 12.86 vs BTX 11.72 (+9.72%)，尤其显著提升在 TriviaQA（30.68 vs 25.10）。

从算法pipeline角度拆解术语：
```
输入: task vectors [τ₁,...,τₗ], p=80%, λ=1/3
for each τᵢ:
    mask = random_bernoulli(prob=p/100)
    τᵢ[mask==0] = 0                 // random drop 20%
    τᵢ = τᵢ / (0.01 * p)            // rescale
τ_m = Σ τᵢ
θ_m = θ_b + λ · τ_m                // 仅应用于非 FFN 层
```

术语一般如何实现？如何使用？
- 开源：mergekit 库提供 `dare_linear` 方法。随机 drop 的 seed 可固定以保持可复现性。
- 在 MergeME MoE 场景仅应用于非 FFN 层。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
