## Smooth K (K矩阵平滑 / Attention Channel-wise Outlier Smoothing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Smooth K 是 SageAttention 提出的针对 Attention 中 K 矩阵 Channel-wise Outlier 的平滑方法。核心观察：K 矩阵存在 channel-wise large bias——每个 token 的 key 实际上是"所有 token 共享的大偏置 + 小的 token-wise signal"的叠加，即 outlier 源于大偏置而非 token 间的大方差。Smooth K 通过减去所有 token 的均值来消除该偏置: `K_smooth = K - mean(K)`，其中 `mean(K) = 1/N * Σ_{t=1}^{N} K[t,:]`，形状 1×d。该变换的数学关键性质：它不改变 attention score P，因为对于任意 query q，`σ(q(K - mean(K))^⊤) = σ(qK^⊤ - q·mean(K)^⊤) = σ(qK^⊤)`（softmax 对常数偏移具有不变性）。实验表明 smooth K 使 INT8 per-token 量化的 Cosine Similarity 从 62.24% 提升至 99.47%，且 speed overhead <0.2%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: K ∈ R^{N×d} (FP16)
K_mean = sum(K, dim=0) / N        # [d], 沿token维求均值
K_smooth = K - K_mean             # [N×d], 减去均值消除channel bias
δ_K, K̂_INT8 = ψ_K(K_smooth)       # per-token或per-block INT8量化
S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]  # INT8 FlashAttention
```
与 SmoothQuant (Xiao et al., 2023) 的区别：SmoothQuant 将量化难度从激活迁移到权重处理 linear layer activation outlier，但 attention 中无可迁移权重维度且 Q 也受 outlier 影响。Smooth K 利用 attention softmax 特有的常数偏移不变性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 FlashAttention-2 tiling 下，mean(K) 通过两次 pass 完成：累加所有 K tile sum → 计算全局 mean → 逐 tile 减去。除以 N 和逐元素减法为 element-wise 操作，开销 <0.2%。可在 Triton kernel 中融合于 K 加载阶段。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
