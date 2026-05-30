## SVD Entropy (SVD熵 / 奇异值分解熵 / Spectral Entropy Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SVD Entropy（奇异值分解熵）是一种量化矩阵奇异值分布"平坦度"的度量。给定权重矩阵 W 的奇异值 σ = (σ₁, σ₂, ..., σ_n)（降序排列），SVD entropy 定义为：
$$H(σ) = -\frac{1}{\log n} \sum_{i=1}^{n} \frac{\sigma_i^2}{\sum_{j=1}^{n} \sigma_j^2} \log \frac{\sigma_i^2}{\sum_{j=1}^{n} \sigma_j^2}$$
值域 [0, 1]。H=1 表示所有奇异值相等（分布最平坦、最均匀），H→0 表示仅少数奇异值主导（分布最集中、rank 最低）。在深度学习优化中，SVD entropy 用于评估优化器是否使模型权重矩阵学习到更多样化、更平坦的奇异值谱——高 SVD entropy 意味着权重在更多方向上具有表达能力，而非集中在少数主导方向（后者可能导致过拟合或容量浪费）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Muon vs AdamW 的谱分析实验中 (Sec 3.4)：

```
# 对每个权重矩阵 W：
U, Σ, V^T = SVD(W)                     # 计算奇异值分解
σ = diag(Σ)                             # 提取奇异值向量

# 归一化为概率分布
p_i = σ_i² / Σ_j σ_j²                   # 用平方奇异值（对应能量/方差）

# 计算归一化 SVD entropy
H = - Σ_i p_i * log(p_i) / log(n)       # n = min(A, B)，归一化到 [0,1]

# 分组平均
groups = {AttnQO, AttnKV, Experts, SharedExperts, Router, Dense}
H_group = mean(H over all matrices in group)
```

实验结果 (图 4)：
- 在 1.2T tokens 训练过程中的所有 checkpoint、所有 6 组权重矩阵上，Muon 的 SVD entropy 均高于 AdamW
- Router 权重的差异最大（Muon 显著高于 AdamW），说明 MoE 模型受益更大——更平坦的路由器权重谱意味着更差异化的专家选择
- 超过 90% 的独立权重矩阵在 Muon 下 SVD entropy 更高（Appendix F, 图 9-10）
- Singular value 分布可视化显示 Muon 训练的权重奇异值曲线更平坦（更少在少数大奇异值处集中）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 计算工具：任何支持 SVD 的数值库（PyTorch `torch.linalg.svdvals()`, NumPy `np.linalg.svd()`）
- 在训练中通常仅对 checkpoint 做离线分析（全 SVD 开销大），不必每步计算
- 应用场景：(a) 优化器对比——衡量不同优化器产生的权重多样性（如 Muon vs AdamW）；(b) 权重初始化质量评估——高 SVD entropy 的初始化可能更有利于训练；(c) 模型压缩——ARSVD (Adaptive-Rank SVD, Cherukuri & Lala 2025) 用 SVD entropy 指导每层 rank 分配
- 变体：某些工作用 σ_i 而非 σ_i² 计算 p_i，或使用非归一化 entropy H = -Σ σ_i log σ_i（此时非归一化到 [0,1]）
- 注意事项：SVD entropy 仅反映奇异值分布，不直接度量模型性能；高 entropy 不等于好性能，但结合 AdamW 和 Muon 的实验，高 entropy 与更好的下游性能相关

涉及论文标题：
- Muon is Scalable for LLM Training
