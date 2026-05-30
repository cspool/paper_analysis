## Amplified Attention Shift (放大注意力偏移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Amplified Attention Shift 是 QuantSparse 论文提出的描述量化与稀疏注意力组合时性能退化机制的核心概念。当模型权重和激活被量化时，量化噪声 ϵ 注入 QK dot product 产生系统性偏差 δ（‖ϵ‖_F ≤ δ）。当稀疏 mask M 同时应用时，mask 删除的低值 attention connection 中本包含的量化噪声被"锁死"进入最终 attention 分布，两者叠加产生超额偏移：Δ_total = Δ_sparse + Δ_quant + O(‖ϵ‖_F·‖M‖_0)。交叉项 O(‖ϵ‖_F·‖M‖_0) 是关键——它意味着总偏移比单独量化和单独稀疏化之和更大。论文通过定量实验验证：单独量化 attention MSE=0.216, 单独稀疏化 attention MSE=0.134, 而组合后 attention MSE=0.685 (远超 0.216+0.134=0.350 的简单相加)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Amplified Attention Shift 的形式化表达：

```
无压缩:    A_full = softmax(Q·K^T/√d_k)·V

仅量化:    A_quant ≈ softmax((Q+ϵ_q)(K+ϵ_k)^T/√d_k)·V
           = A_full + Δ_quant   (MSE ≈ 0.216)

仅稀疏化:  A_sparse = softmax(Q·K^T/√d_k ⊙ M)·V
           = A_full + Δ_sparse  (MSE ≈ 0.134)

量化+稀疏
(naive组合): A_sq = softmax(Q_q·K_q^T/√d_k ⊙ M)·V
           = A_full + Δ_sparse + Δ_quant + O(‖ϵ‖·‖M‖₀)
           (MSE ≈ 0.685, 远超 0.216+0.134=0.350)
```

交叉项 O(‖ϵ‖_F·‖M‖_0) 的来源：量化噪声 ϵ 对 QK 矩阵中所有元素都产生扰动，稀疏 mask 删除部分 attention connection 后，被删除位置的量化误差无法被 softmax normalization 中的其他 attention 值"稀释"，导致保留的 attention connection 承受了不成比例的量化和稀疏化双重扭曲。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuantSparse 通过两个技术对抗 Amplified Attention Shift：(1) MSAD——在 calibration 阶段直接监督 attention map 对齐，减少 Δ_quant 的幅度；(2) SSAR——在 inference 阶段通过二阶残差缓存恢复稀疏化丢失的低值 attention, 减小 Δ_sparse 的幅度。两者协同将 attention MSE 从 0.685 恢复至接近 FP 水平。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---
