## Centered Kernel Alignment (CKA) for Attention Head Similarity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CKA 是 Kornblith et al. (NeurIPS 2019) 提出的表征相似度度量。给定两个 head 的 Key 投影子矩阵 W_i, W_j ∈ R^{d_model×d_k}，通过线性核 Gram 矩阵 G_i=W_i·W_i^T, G_j=W_j·W_j^T，centering 后计算 CKA(i,j) = HSIC(G_i_c, G_j_c)/√(HSIC(G_i_c,G_i_c)·HSIC(G_j_c,G_j_c))，其中 HSIC(A,B)=Tr(A·B)。CKA ∈ [0,1]，值越高表示 head 子空间结构越相似。ReCalKV 用 CKA 指导 HSR 的 head 分组。

从算法pipeline角度拆解术语：

```
// CKA for head similarity
G_i = W_i @ W_i.T  // [d_model, d_model]
G_i_c = G_i - G_i.mean(0) - G_i.mean(1) + G_i.mean()  // centering
CKA(i,j) = Tr(G_i_c @ G_j_c) / sqrt(Tr(G_i_c²)·Tr(G_j_c²))
```

术语一般如何实现？如何使用？

PyTorch: `torch.mm()` + `torch.trace()`。ReCalKV 在 offline 阶段对所有 head 对计算一次 CKA（O(h²·d_model²·d_k)），结果用于贪心分组。也可用于 layer pruning、head pruning 等需要表征相似度分析的场景。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 中的 CKA 使用**：xKV 将 CKA 用于衡量不同 Transformer 层 KV-Cache 之间的整体结构相似度（而非 head 间相似度）。对于层 ℓ 的 KV-Cache X_ℓ ∈ R^{n×d}（n 为 token 数），在 token 维度计算 centered Gram matrix G_ℓ = H X_ℓ X_ℓ^T H（H 为 centering matrix），CKA(X_ℓ1, X_ℓ2) = Tr(G_ℓ1 G_ℓ2) / √(Tr(G_ℓ1²)·Tr(G_ℓ2²))。高 CKA 值表明层间的主导左奇异向量高度对齐（详见 xKV 论文附录 A: CKA = Σ_{i,j} σ_i² σ_j² (u_i·v_j)² / ...），即使 token-wise cosine similarity 很低。xKV 利用此发现设计跨层 SVD 压缩——通过对齐的奇异向量共享压缩基。

---
