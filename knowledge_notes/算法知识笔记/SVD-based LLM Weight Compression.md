## SVD-based LLM Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SVD-based LLM Weight Compression（基于奇异值分解的 LLM 权重压缩）是一类利用低秩近似压缩 LLM 权重的后训练压缩方法。对 W ∈ R^{d1×d2}，SVD 分解 W = U Σ V^T，保留前 k 个最大奇异值：W ≈ U_k Σ_k V_k^T。压缩比 = (d1·d2) / (k·(d1+d2+1))。

演进路径：(1) Vanilla SVD：直接 SVD 截断，压缩误差大；(2) FWSVD（Hsu et al., 2022）：Fisher 信息加权，需梯度、大模型 OOM；(3) ASVD（Yuan et al., 2023）：激活敏感度选择通道，避免梯度；(4) SVD-LLM（Wang et al., 2024b）：whitening matrix S（S·S^T = cholesky(X^T X)）缩放权重后 SVD，使截断误差与输出误差对齐——当前 per-layer SVD SOTA；(5) Basis Sharing（本文）：在 SVD-LLM 基础上扩展跨层共享基向量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SVD-LLM 压缩流程（W, X, 压缩比 x%）：

```
X = collect_activation(W, calib_data)    # [L, d1]
S = cholesky(X^T @ X)^{1/2}             # whitening matrix
U, Σ, V^T = SVD(S @ W)
k = (d1 * d2 * x%) / (d1 + d2)
B = S^{-1} @ U[:,:k] @ Σ[:k,:k]         # 基矩阵 [d1, k]
C = V[:k, :]                             # 系数 [k, d2]
# 推理: Y ≈ X @ B @ C
```

关键数学：引入 S 后 ‖XW - XW_k‖_F² = ‖(SW) - (SW)_k‖_F²，SVD 截断直接对应输出 MSE 最小化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：SVD-LLM https://github.com/AIoT-MLSys-Lab/SVD-LLM; Basis Sharing https://github.com/TUDa-HWAI/Basis_Sharing。需 128-256 条校准样本，FP64 S 评估。LLaMA-7B 20% 压缩 SVD-LLM PPL=7.94 (dense=5.68)，Basis Sharing=7.74。限制：大模型需大显存，>50% 压缩误差急剧增大，加速来自内存带宽节省而非计算量减少，校准集分布影响 reasoning 效果。

涉及论文标题：
- Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

SLiM 将 SVD 压缩推广到显著性加权的误差补偿范式：不对原始权重 W 做 SVD，而是对压缩误差的显著性矩阵 S_C = diag(x)·(W^C - W) 做 SVD，再通过逆显著性变换 diag(1/x) 恢复低秩适配器。相比直接对原始权重或误差做 SVD，显著性加权确保低秩近似优先修正对模型输出影响最大的权重通道。该方法的数学基础是显著性函数 F(W)=diag(x)W 的可加性（F(W^C+LR)=F(W^C)+F(LR)）和可逆性，使误差隔离和适配器恢复成为可能。

---
