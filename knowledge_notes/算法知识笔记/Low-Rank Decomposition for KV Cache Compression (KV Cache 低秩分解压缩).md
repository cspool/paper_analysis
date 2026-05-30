## Low-Rank Decomposition for KV Cache Compression (KV Cache 低秩分解压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

低秩分解压缩 KV Cache 是通过对 Key/Value 权重矩阵 $W^K, W^V$ 做矩阵分解来减少 KV Cache 存储开销的技术。将 $W^K \in \mathbb{R}^{h_{in} \times h_{out}}$ 近似为两个更小矩阵的乘积 $W^K \approx A^K B^K$，其中 $A^K \in \mathbb{R}^{h_{in} \times h_{comp}}$，$B^K \in \mathbb{R}^{h_{comp} \times h_{out}}$。推理时存储 $X A^K$（维度 $h_{comp}$）而非 $X W^K$（维度 $h_{out}$），实现 $h_{out} / h_{comp}$ 倍压缩。

CSKV 发现 KV Cache 的奇异值呈长尾分布——大量小奇异值可移除而不显著影响性能。直接使用标准 SVD 分解 $W = U \Sigma V^T$ 保留前 k 个奇异值存在局限：它不考虑激活值分布。ASVD（Activation-aware SVD）通过缩放矩阵 S 使分解对高激活值维度更敏感。

从算法pipeline角度拆解术语。

```
// 标准 SVD 低秩分解
W ∈ R^{hin × hout}
U, Σ, V_T = SVD(W)
A = U[:, :hcomp] @ sqrt(Σ[:hcomp, :hcomp])  // (hin, hcomp)
B = sqrt(Σ[:hcomp, :hcomp]) @ V_T[:hcomp, :] // (hcomp, hout)

// ASVD 变体
S = diag(mean(|X|, dim=0)^α)  // 缩放矩阵, α=0.5
W_s = W @ S
U_s, Σ_s, _ = SVD(W_s)
A = inv(S) @ U_s[:, :hcomp] @ sqrt(Σ_s[:hcomp, :hcomp])
B = sqrt(Σ_s[:hcomp, :hcomp]) @ V_s_T[:hcomp, :] @ inv(S)
```

术语一般如何实现？如何使用？

PyTorch 中通过 `torch.linalg.svd()` 实现。ASVD 初始化需从标定数据集采样 256 样本，收集每层 Key/Value 激活，计算 Absolute Mean Value 作为 S（α=0.5）。初始化后的 A/B 作为可训练参数，通过逐层 MSE 损失微调。适用于 LLaMA、Mistral 等标准 Transformer 架构。

ReCalKV 将低秩 KV Cache 压缩进一步细化为不对称的 Key/Value 策略：(1) 对于 Keys，使用 HSR（Head-wise Similarity-aware Reordering）先通过 CKA 相似度将结构相似的 head 分组，再对每组做 grouped SVD；(2) 对于 Values，使用 OVC（Offline Value Calibration）用标定数据对 SVD 分解后的 L_v 和 R_v 做闭式校准，最小化 ||L_v R_v X - W_v X||_F^2，然后通过 Matrix Fusion 将 R_v 融合进 output projection W_o 消除推理时重建开销。ReCalKV 还引入 Fisher Information 引导的逐层压缩率分配，使重要层保留更多 rank。经 256 个 WikiText-2 标定样本在单张 A800 GPU 上完成离线压缩后，推理时 50% 压缩率下零样本 QA 仅降 ~2%。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
- TransMLA: Multi-Head Latent Attention Is All You Need
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 直接对 KV-Cache 做跨层低秩分解**：与上述方法对权重矩阵离线分解不同，xKV 直接对 prefill 阶段产生的 **KV-Cache（而非权重 W_K/W_V）** 做在线 SVD，且引入了**跨层维度**：将多层的 KV-Cache 水平拼接后做统一 SVD，提取跨层共享基。xKV 发现 CKA 分析表明跨层主导左奇异向量高度对齐（但 token-wise cosine similarity 很低），因此跨层 SVD 比单层 SVD 更高效——相同 rank 保留更多跨层共享信息，相同压缩比下精度更高（8× 压缩下 xKV avg=87.8% vs Single SVD avg=35.3%）。此外 xKV 还兼容 MLA 架构（对 non-RoPE latent representations 做跨层 SVD）。

TransMLA 提出 BKV-PCA——对 K_nope 和 V 做联合低秩压缩前，先计算平衡因子 α = E[||K_nope||₂]/E[||V||₂] 缩放 K 使两者 norm 对齐，避免 key 主导 PCA 主成分方向导致 value 信息丢失。与 weight-based SVD 相比，activation-based PCA（在标定数据激活值上做 PCA 而非在权重矩阵上做 SVD）在 TransMLA 实验中显著降低压缩损失（Figure 4b）。BKV-PCA 联合压缩 [K_nope; V]（(2g-1)d 维）到 r_kv 维 latent 空间，推理时仅缓存 r_kv 维 latent vector，而非 (2g-1)d 维的完整 NoPE key + value。
