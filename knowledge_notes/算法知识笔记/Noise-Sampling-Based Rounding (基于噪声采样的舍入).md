## Noise-Sampling-Based Rounding (基于噪声采样的舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise-Sampling-Based Rounding 是 HDRQ 提出的解决模型合并中舍入歧义（Rounding Ambiguity）的技术。问题源于量化模型合并时的整数域歧义：当两个量化值的整数表示 I₁、I₂ 之和为奇数时，midpoint averaging 的合并结果落在两相邻整数中间，导致舍入方向不确定。在浮点域合并公式 I_merged = ⌊(I₁·Δ₁ + I₂·Δ₂)/(Δ₁+Δ₂)⌉ 中，当 Δ₁≈Δ₂（域自适应场景的典型情况，因共享源模型且学习率小），步长项被约去，公式退化为 ⌊(I₁+I₂)/2⌉，又回到歧义问题。HDRQ 的解决方案：合并前对权重添加采样噪声 ε₁, ε₂ ∼ U[-Δ/2, Δ/2]，计算 I_merged = ⌊(I₁·Δ₁+ε₁ + I₂·Δ₂+ε₂)/(Δ₁+Δ₂)⌉。采样多组噪声（如 30 组），计算各组合并权重到目标域权重的向量与原始插值向量的 cosine similarity，选择最高相似度的样本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 两个量化模型的权重 w1_q = I1 * Δ1, w2_q = I2 * Δ2
#       噪声样本数 K=30

best_score = -inf
best_w_merged = None

for k in range(K):
    # 1. 对每个量化权重添加采样噪声
    ε1 ~ U[-Δ1/2, Δ1/2]
    ε2 ~ U[-Δ2/2, Δ2/2]
    
    # 2. 噪声辅助的整数舍入
    I_merged = round((I1 * Δ1 + ε1 + I2 * Δ2 + ε2) / (Δ1 + Δ2))
    w_merged_k = I_merged * ((Δ1 + Δ2) / 2)     # 反量化
    
    # 3. Cosine similarity 质量评估
    v_merged = flatten(w_merged_k - w_src)       # 合并后变化方向
    v_interp = flatten(w1_q - w2_q)              # 原始插值方向
    score = dot(v_merged, v_interp) / (||v_merged|| * ||v_interp||)
    
    # 4. 选最优
    if score > best_score:
        best_score = score
        best_w_merged = w_merged_k

return best_w_merged
```
当 Δ₁≈Δ₂ 时，噪声 ε₁, ε₂ 打破确定性舍入的对称性，提供多个候选舍入方向。Cosine similarity 筛选确保合并方向与原始无噪声插值方向保持一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该技术在 HDRQ 中仅用于合并阶段（推理前的一次性离线操作），不增加推理时计算开销。采样 30 组噪声的额外计算可忽略（仅需每层权重做一次加法+取整+sanity check）。噪声采样舍入的有效性在实验中被证实（Figure 3）：对比无 filter（随机采样）、不使用 cosine similarity filter 以及 HDRQ 的 Advanced（cosine similarity filter），Advanced 方法显著稳定合并质量，filter 掉低质量噪声样本。三目标域合并场景（Office-Home 三域）由于不再存在奇数歧义（三数平均天然落在连续区域），无需噪声采样。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
