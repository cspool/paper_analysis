## KV Cache Quantization (KV 缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Quantization 是一类将 LLM 推理中的 KV Cache 从高精度（FP16/BF16）压缩到低比特整数（INT4/INT2）以减少 GPU 内存占用的技术。与权重量化不同：(1) KV Cache 是流式数据结构——新 token 的 K/V 实时到达并追加，无法使用需要离线校准的优化方法（如 GPTQ 的 Hessian 补偿）；(2) KV Cache 的数值分布随序列长度动态变化，预计算的量化参数可能失效；(3) 量化误差不仅影响当前层还通过 residual 累积传播。

KIVI 提出了一种免调优的非对称 2bit KV Cache 量化：基于对 key/value cache 元素分布的深入分析，发现 key cache 应 per-channel 量化（隔离 outlier channel），value cache 应 per-token 量化（保护重要 token 不受干扰）。同时通过 grouped+residual split 实现流式兼容和局部全精度滑动窗口。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 的 KV Cache 量化与 attention 计算融合流程：

```
# KV Cache 结构: {Q(X_K_g): int2, X_K_r: FP16, Q(X_V_g): int2, X_V_r: FP16}
# Q(X_K_g): per-channel group-wise 2bit quantized
# Q(X_V_g): per-token group-wise 2bit quantized
# X_K_r, X_V_r: 最近R个token保持FP16 (full precision sliding window)

# 每 decoding step:
t_Q = t @ W_Q, t_K = t @ W_K, t_V = t @ W_V

# --- 更新 KV Cache ---
X_K_r = Concat([X_K_r, t_K])
X_V_r = Concat([X_V_r, t_V])
if len(X_K_r) == R:
    Q(X_K_r_new) = KeyQuant(X_K_r)         # per-channel quant
    Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r_new)])
    X_K_r = empty

# --- 混合精度 Attention ---
A_g = t_Q @ Dequant(Q(X_K_g))^T  # grouped部分 (fused dequant+matmul)
A_r = t_Q @ X_K_r^T              # residual部分 (FP16)
A = Concat([A_g, A_r])
A_g_sm, A_r_sm = Softmax(A)[:-R], Softmax(A)[-R:]

t_O = A_g_sm @ Dequant(Q(X_V_g)) + A_r_sm @ X_V_r  # 混合精度加权求和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
早期 FlexGen 使用 4bit per-token group-wise RTN 量化 KV Cache，但未区分 key/value。SmoothQuant 可通过 equivalent transformation 迁移量化难度使 KV Cache 可 8bit 量化，但 4bit 精度大幅下降。KIVI 是第一篇系统研究 KV Cache 元素分布并设计非对称量化策略的工作（与 KVQuant 同期独立发现）。KIVI 代码开源：https://github.com/jy-yuan/KIVI，基于 HuggingFace Transformers，使用 CUDA (fused dequant+matmul) + Triton (group-wise quantization) 实现。QuaRot 从不同角度解决 KV Cache 量化问题：通过 head-wise Hadamard 旋转消除 Key 和 Value 中的离群值（与消除激活值离群值相同的原理），使简单的 asymmetric group-wise INT4 量化（group=128）即可在 4-bit KV cache 下实现近乎无损的困惑度（+0.04 on 7B）。QuaRot 使用 Post-RoPE Caching（在 RoPE 后做在线 head-wise Hadamard 旋转再量化缓存），避免了解码时需对大量缓存 key 做逆旋转的开销。

ResQ 将混合精度量化扩展到 KV Cache：通过 U_B 和 U_C 两个投影矩阵分别处理 value 和 key 的 KV cache 量化。(1) U_B 后乘 value 投影层 W_v，将 value 向量投影到 PCA+PCA 基上，离线融合 U_B^T 到 o_proj 权重；(2) U_C 利用 computational invariance 同时投影 query 和 key（对称投影），因 RoPE 无法融合而在运行时显式计算，但量化为 8-bit 以降低开销。最终 KV cache 以 4/8-bit 混合精度存储（1/8 通道 8-bit）。key/value cache 均采用 per-head asymmetric 量化。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

---
