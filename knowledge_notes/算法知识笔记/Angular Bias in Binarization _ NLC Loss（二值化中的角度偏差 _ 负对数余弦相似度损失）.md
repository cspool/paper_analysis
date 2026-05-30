## Angular Bias in Binarization / NLC Loss（二值化中的角度偏差 / 负对数余弦相似度损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
二值化中的角度偏差（Angular Bias）指二值化权重与原始 FP 权重在方向（非仅幅值）上存在偏差。传统方法仅用 MSE loss 最小化幅值差距，但方向不同也会导致与激活相乘后输出差异。NLC Loss（Negative Logarithm of Cosine Similarity Loss）= -log(cos_sim(f₁, f₂))，其中 cos_sim = (f₁·f₂)/(||f₁||x||f₂||)。PTQ1.61 联合 MSE + NLC 作为分块优化目标：E(f₁, f₂) = ||f₁-f₂||₂ + (-log(cos_sim(f₁, f₂)))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PTQ1.61 分块优化的损失计算
W_q_prime = (alpha_r1 x alpha_r2) * (alpha_s * sign(W))  # Eq. 9
out_fp = block_fp(X); out_q = block_q(X, W_q_prime)

# 联合损失 (Eq. 5-7)
loss_mse = ||out_fp - out_q||_2
cos_sim = (out_fp · out_q) / (||out_fp|| x ||out_q||)
loss_nlc = -log(cos_sim)
total_loss = loss_mse + loss_nlc  # 分支1
# 分支2: 同样联合损失，输入为量化激活 X_q
```
NLC 特性：cos_sim→1 时 NLC→0（方向一致）；cos_sim→0 时 NLC→∞（方向正交，强惩罚）；cos_sim→-1 时 MSE 主导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
消融（Table 7, LLaMA-1-7B）：带角度偏差 PPL=12.50 vs 不带 13.56（WikiText2）。NLC 在 block 输出级计算（非权重级），因为目标是对齐 block 输出方向。基于 CBQ 分块框架增强。RBNN (Lin et al. 2020) 首次指出角度偏差问题并使用旋转矩阵纠正；LRQuant (Zhao et al. 2024) 在 PTQ 中引入余弦相似度考量。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- RBNN: Rotated Binary Neural Network
- LRQuant: Learnable and Robust Post-Training Quantization for Large Language Models
