## Least-Squares Optimization for Expert Merging (T1 Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 MergeMoE 中，T1 矩阵是用于减少 merged expert 中间维度的线性变换矩阵。当 N 个 expert 被合并为一个时，扩展后的中间维度为 N×E，T1 ∈ R^{E×E}（在合并后 expert 内部）将其压缩回 E。MergeMoE 使用最小二乘法而非简单拼接来求解 T1：固定 T2/T3 后，对采样输入 X̂ 计算 P = σ(T2 W'_G X̂) ⊙ (T3 W'_U X̂)（经压缩路径的中间激活）和 Q = σ(W'_G X̂) ⊙ (W'_U X̂)（原始路径），利用 Moore-Penrose 伪逆求 T1 = Q P^† 的闭式解，最小化 ||T1 P - Q||_F。这与 M-SMoE 形成对比——M-SMoE 等价于 T1 = [I; I; ...; I]（仅拼接不做优化）。

从算法pipeline角度拆解术语：
```
// 步骤1: 获取中间激活 (torch hooks)
P = σ(T2 · W'_Gi · X̂) ⊙ (T3 · W'_Ui · X̂)    // 压缩路径
Q = σ(W'_Gi · X̂) ⊙ (W'_Ui · X̂)               // 原始扩展路径

// 步骤2: 最小二乘闭式解
T1 = Q @ pinv(P)   // Moore-Penrose 伪逆, shape (E, E)

// 步骤3: 构造最终权重
W^final_Di = W'_{Di} @ T1; W^final_Gi = T2 @ W'_{Gi}; W^final_Ui = T3 @ W'_{Ui}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch `torch.linalg.pinv(P)` 或 `torch.linalg.lstsq(P.T, Q.T)`；BFloat16 精度容纳更多样本。
- 样本量临界阈值 ~32 samples，低于此值性能崩溃（≈random guessing）；高于阈值后逐步提升。
- 跨数据集泛化好：即使单一数据集（如 WinoGrande）样本计算的 T1 在其他 benchmark 上表现良好（<1-2% drop）。
- 从后往前逐层压缩：后层压缩不影响前层激活，每层获取 hooks → 计算 T1 → 释放内存。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---
