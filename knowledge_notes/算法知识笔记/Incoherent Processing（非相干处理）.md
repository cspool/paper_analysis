## Incoherent Processing（非相干处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incoherent Processing（非相干处理）是一种量化预处理技术，通过在量化前对数据施加随机正交变换来"均匀化"数据分布，减少outlier对量化精度的破坏。给定矩阵$\mathbf{Q}$，将其乘以随机正交矩阵$\mathbf{M}$（满足$\mathbf{M}\mathbf{M}^\top = \mathbf{I}$）得到$\mathbf{Q}' = \mathbf{Q}\mathbf{M}$。由于$\mathbf{M}$是正交矩阵，$(\mathbf{Q}\mathbf{M})(\mathbf{K}\mathbf{M})^\top = \mathbf{Q}\mathbf{K}^\top$，即attention的数学结果不变。但关键在于：$\mathbf{Q}\mathbf{M}$的每个元素是$\mathbf{Q}$一行中各元素的随机加权和（权重来自$\mathbf{M}$的列），这使原本集中在少数维度的outlier被"分散"到所有维度中——每个元素的大小趋于均匀（由中心极限定理），大幅降低量化时的动态范围差异。FlashAttention-3采用Chee et al. (QuIP) 和Tseng et al. (QuIP#) 的方法，取$\mathbf{M}$为Hadamard矩阵$\mathbf{H}$与随机对角符号矩阵$\mathbf{D}$的乘积：$\mathbf{M} = \mathbf{H}\mathbf{D}$，计算复杂度从$O(d^2)$降至$O(d \log d)$（由于Hadamard变换可用Fast Walsh-Hadamard Transform加速）。

从算法pipeline角度拆解术语：
Incoherent processing在FP8 FlashAttention-3中的位置（fuse到rotary embedding，零额外开销）：
```
// 原始Q, K (FP16/BF16)
1. Apply Rotary Position Embedding: Q_rope, K_rope = RoPE(Q), RoPE(K)
   // rotary embedding is memory-bound, fuse incoherent processing here
2. Multiply by random orthogonal matrix M = H × D:
   Q' = Q_rope × M    // = Q_rope × H × D, O(d log d) via Fast Walsh-Hadamard
   K' = K_rope × M    // = K_rope × H × D
   // Q'K'^T = (Q_rope M)(K_rope M)^T = Q_rope (M M^T) K_rope^T = Q_rope K_rope^T
3. Block quantize Q', K' to FP8 e4m3
4. Proceed with FP8 FlashAttention-3 on Q'_FP8, K'_FP8
```
数学验证：$\mathbf{M}$由随机对角矩阵$\mathbf{D}$（对角元为±1随机取值）和Hadamard矩阵$\mathbf{H}$组成。$\mathbf{H}$满足$\mathbf{H}\mathbf{H}^\top = d\mathbf{I}$（Hadamard矩阵是正交的，但scale by $\sqrt{d}$），归一化后为正交。$\mathbf{D}$是对角符号矩阵，满足$\mathbf{D}\mathbf{D}^\top = \mathbf{I}$。乘积$\mathbf{M}=\mathbf{H}\mathbf{D}$仍是正交矩阵。FlashAttention-3的数值实验验证：FP8 with block quant + incoherent processing 的RMSE比per-tensor FP8 baseline低2.6×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Incoherent processing源自QuIP (Chee et al., NeurIPS 2023) 和QuIP# (Tseng et al., 2024) 的LLM weight quantization方法。FlashAttention-3将其adapt到attention activation quantization场景。实现采用Fast Walsh-Hadamard Transform (FWHT)：`y = FWHT(x)`迭代式地将输入向量通过log2(d)层butterfly操作，每层O(d)，总O(d log d)。随机符号矩阵$\mathbf{D}$的生成使用固定seed的PRNG（每个head独立），存储开销仅O(d)而非O(d²)。在FlashAttention-3的implementation中，incoherent processing与rotary embedding融合在同一个preprocessing kernel中，不引入额外kernel launch开销。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
