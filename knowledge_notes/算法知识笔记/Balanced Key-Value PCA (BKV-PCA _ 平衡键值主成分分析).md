## Balanced Key-Value PCA (BKV-PCA / 平衡键值主成分分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Balanced Key-Value PCA (BKV-PCA) 是 TransMLA 论文提出的对 K_nope 和 V 做联合低秩压缩时解决 norm 不平衡问题的技术。问题：经过 RoRoPE 后，K_nope（去除了 RoPE 的 key，不含第一 head）的 ℓ₂-norm 显著大于 V 的 ℓ₂-norm（因 key 保留了主要信息成分）。如果直接对 [K_nope; V] 拼接做 PCA，主成分方向会被 norm 更大的 K_nope 主导，导致 value 子空间信息在压缩中严重丢失。BKV 解决：计算平衡因子 α = E[||K_nope||₂] / E[||V||₂]，在校准数据集上将 K_nope 缩放 1/α 使两者 norm 对齐后再拼接做联合 PCA，得到平衡的低秩投影矩阵 R_KV。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**BKV-PCA 计算流程**：
```
// Step 1: 计算 norm 平衡因子
For each calibration sample x_t:
    k_nope_t = W_NoPE^{DK} @ x_t    // K_nope activation, [ (g-1)d ]
    v_t = W^{DV} @ x_t              // V activation, [ gd ]
α = mean(||k_nope_t||_2) / mean(||v_t||_2)  // 标量

// Step 2: 平衡后拼接
For each calibration sample x_t:
    k'_t = (1/α) · k_nope_t                          // 缩放 K_nope
    c_t = concat([k'_t; v_t])                         // [(2g-1)d]
// 此时 ||k'_t||_2 ≈ ||v_t||_2

// Step 3: 联合 PCA
Σ = covariance_matrix({c_t})                          // [(2g-1)d, (2g-1)d]
eigenvalues, eigenvectors = eig(Σ)                     // 按特征值降序
R_KV = eigenvectors[:, :r_kv]                          // [(2g-1)d, r_kv]

// Step 4: 低秩分解（应用于权重矩阵）
W^{DKV'} = R_KV^T @ [W_NoPE^{DK}; W^{DV}]              // [r_kv, D]
W^{UKV'} = [W_NoPE^{UK}, 0; 0, W^{UV}] @ R_KV           // [2hd, r_kv]
// 推理时仅缓存 c_t^{KV'} = W^{DKV'} @ x_t ∈ R^{r_kv}
```

**BKV 的等价性**：因 W^{UK} 相应缩放 α 倍：(1/α · W_NoPE^{DK}) × (α · W_NoPE^{UK}) = W_NoPE^{DK} × W_NoPE^{UK}，数学上等价于原始计算，不改变模型输出。BKV 仅改变 PCA 阶段的数据分布（使 K/V 的 norm 平衡），使主成分方向更均衡地捕获两者的方差。

术语一般如何实现？如何使用？

BKV-PCA 在校准数据集（WikiText-2 子集）上离线执行。实现要点：(1) α 基于校准集上期望 norm 比值计算，使用 running average；(2) BKV 后 PCA 可选择 weight-based（对 W 做 SVD）或 activation-based（对激活值做 PCA，TransMLA 证明效果更好，Figure 4b）；(3) r_kv 选择决定压缩率——KV cache 从 2gd 压缩到 r_kv + d（K_rope 的 d 维不参与压缩）。TransMLA 实验中 BKV 显著降低联合 PCA 的 perplexity 损失（Figure 4b），是 training-free 转换低损失的关键因素。与 MHA2MLA（直接联合 SVD，无 BKV）相比，BKV-PCA 在相同压缩率下性能显著更好。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---
