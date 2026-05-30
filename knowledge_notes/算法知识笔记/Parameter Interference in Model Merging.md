## Parameter Interference in Model Merging

术语是什么？
Parameter Interference（参数干扰）是合并多个 fine-tuned 模型时，因 task vector 间的冲突导致的性能下降。MergeME Figure 2 识别三类干扰：(1) **Sign Conflict**：τ₁[j] > 0, τ₂[j] < 0 → 平均后抵消；(2) **Magnitude Disparity**：大 magnitude 被小值稀释；(3) **Redundancy**：接近零的参数不携带信息但占用参数空间。Dare 和 Ties 通过 drop 和 sign alignment 缓解。MergeME Figure 4 验证 attention 层 task vector 的余弦相似度也较低（~0.1-0.3），说明 BTX 的"attention 层可直接平均"假设不成立。

从算法pipeline角度拆解术语：
```
// Sign Conflict 示例:
τ_math[j] = +0.8, τ_code[j] = -0.6
Average: 0.1（几乎归零）  // Ties 方案: 保留 +0.8, 丢弃 -0.6

// Magnitude Disparity 示例:
τ_math[j] = 0.9, τ_know[j] = 0.01
Average: 0.455（大参数被稀释）  // Dare 方案: drop 0.01 后 rescale
```

术语一般如何实现？如何使用？
- 检测：task vector 余弦相似度分析（MergeME Figure 4）。
- 缓解：(a) Dare: random drop + rescale；(b) Ties: trim + elect sign + disjoint merge。
- MoE 特殊性：仅共享层（非 FFN）受干扰，范围比 dense 合并更受控。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
