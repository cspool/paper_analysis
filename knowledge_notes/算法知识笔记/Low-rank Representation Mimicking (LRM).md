## Low-rank Representation Mimicking (LRM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Low-rank Representation Mimicking (LRM) 是 BinaryDM 提出的低秩表征模仿技术，用于辅助二值化 DM 的优化。核心流程：对全精度 DM 各 timestep embedding 模块的输出 ε̂ ∈ R^{h×w×c} 计算协方差矩阵 C = (hw)⁻² * ε̂ * ε̂^T → 特征分解取前 ⌈c/K⌉ 列特征向量 E（K=4, 降维4倍）→ 将全精度和二值化 DM 的中间表征投影到低秩空间 R^FP = ε̂^FP * E, R^bi = ε̂^bi * E → MSE 损失 ||R^FP - R^bi|| 驱动二值化 DM 沿主成分方向学习。投影矩阵 E 在首 batch 计算后固定，保证优化方向稳定性。LRM 避免高维空间直接对齐导致的模糊优化方向，使二值化 DM 集中于主方向学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Init: compute E_i once and freeze
for i, module in enumerate(timestep_embed_modules):
    ε_fp = fp_model.module_i(first_batch)          # [h,w,c]
    C = (ε_fp @ ε_fp.T) / (h*w)^2                  # [c,c]
    eigvecs = eigh(C).eigenvectors                  # descending
    k = ceil(c / 4)                                 # K=4
    E_i = eigvecs[:, :k]                            # freeze

# Training: low-rank alignment
for iter:
    for i, module in enumerate(timestep_embed_modules):
        R_fp = fp_model.module_i(x) @ E_i           # [h,w,k]
        R_bi = bin_model.module_i(x) @ E_i          # [h,w,k]
        L_LRM += MSE(R_fp, R_bi)
    L_total = L_simple + L_EBB + (1e-4/M) * L_LRM
```
消融：直接 MSE FID=7.36 vs LRM K=4 FID=6.99；K=8 FID=6.95。每 100 iter 更新 E FID=7.11 vs 固定 E FID=6.99（验证稳定性设计）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `torch.linalg.eigh` 首 batch 计算，E_i 作为 buffer 存储（不参与梯度）。应用于每个 timestep embedding 模块输出。消融（W1A32, LSUN-Bedrooms）：+EBB FID=7.39 → +EBB+LRM FID=6.99。训练 11.3h vs Q-Diffusion 校准 13.7h。

涉及论文标题：
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

---
