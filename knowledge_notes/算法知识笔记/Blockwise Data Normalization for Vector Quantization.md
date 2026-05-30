## Blockwise Data Normalization for Vector Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Blockwise Data Normalization 是 GPTVQ 在 EM codebook 初始化前对权重数据进行的预处理。目的：降低 VQ 的量化误差。方法：对每个 group 对应的权重子矩阵 W_i，按每 sub-row（block，通常 16/32/64 个元素）执行逐元素除法 W_i ⊘ S_i，其中 scale s^{(i)} = max_j |w_j^{(i)}| 是该 sub-row 的最大绝对值。为覆盖多个数量级，scale 在 log 空间量化到 4-bit：s^{(i)}_{int} = ⌈(log₂(s^{(i)}) - z)/a⌋·a。解码时逆操作：w = w_decoded · 2^{-a·s_{int} - s_0}。消融（Table 16）显示：scaling block size 越小（8→128），perplexity 越低但 overhead 越大；2D 3-bit VQ 时 BS=16 的 PPL=5.66 vs 无 scaling 的 PPL=5.91。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Blockwise Data Normalization (codebook 初始化前的预处理)
# 输入: 权重子矩阵 W_i ∈ R^{r × m}, block size BS=32
# 输出: 归一化后的 W_i_normalized, scales S, log-offset z

for each sub-row w in W_i:  # w ∈ R^{BS}
    # Step 1: 计算 scale (max abs)
    s = max(|w_j| for j in range(BS))
    
    # Step 2: Log-space 量化到 4-bit
    s_int = ceil((log2(s) - z) / a) * a  # a = shared quantization step
    # z = floating point offset, shared per-column
    
    # Step 3: 归一化
    w_normalized = w * 2^{-a * s_int - s_0}  # s_0 = log2(z)
    
    store s_int  # 4-bit per block

# 解码时逆归一化:
# w_decoded_fp = w_vq_decoded * 2^{a * s_int + s_0}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 在 codebook 初始化阶段应用 normalization，在 VQ 解码 kernel 中应用 inverse normalization（高效乘 2 的幂次）。Scale overhead：BS=32 时每个 weight 增加 4/32=0.125 bits overhead。Log-space 量化的优势：(1) 覆盖跨越多个数量级的权重值（LLM 权重常见范围 10^{-3} 到 10^{1}）；(2) 乘法逆归一化在硬件上高效（乘 power-of-two ≈ bit-shift）。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
