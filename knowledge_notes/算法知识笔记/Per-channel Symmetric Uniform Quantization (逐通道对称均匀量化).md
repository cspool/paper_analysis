## Per-channel Symmetric Uniform Quantization (逐通道对称均匀量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-channel Symmetric Uniform Quantization 是一种对权重矩阵按输出通道（行）独立计算量化参数的对称均匀量化方法。对称（symmetric）意味着 zero-point = 0，量化范围关于原点对称；均匀（uniform）意味着量化步长恒定；per-channel 意味着每行（输出通道）有独立的 scale 因子。

数学表达：对权重矩阵 W ∈ R^{o×c}，每行 i 的 scale s_i = max(|W[i,:]|) / (2^{bit-1} - 1)，量化后 W_q[i,j] = clamp(round(W[i,j] / s_i), q_min, q_max)，反量化 W_hat[i,j] = W_q[i,j] * s_i。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Per-channel Symmetric Uniform Quantization (4-bit)
# W: FP16 weight matrix of shape [out_features, in_features]
bit = 4
q_min = -2**(bit-1)      # -8
q_max = 2**(bit-1) - 1   # 7

scales = []
for row in W:
    max_abs = max(abs(row))
    scale = max_abs / q_max  # 或 max_abs / (2^{bit-1}-1)
    scales.append(scale)
    
W_q = clamp(round(W / scales.reshape(-1,1)), q_min, q_max)
# W_q ∈ Z^{o×c}, scales ∈ R^o

# Dequantization at runtime:
# W_hat[i,:] = W_q[i,:] * scales[i]
# Then compute: output = input @ W_hat^T (or fused dequant+matmul)
```

相比 per-tensor 量化（整个矩阵一个 scale），per-channel 量化为每行独立选择 scale，更好地适应不同输出通道的权重分布差异；相比 group quantization（每 32/128 个元素一个 scale），per-channel 的粒度更粗但内存开销更小。论文 MoEQuant 采用 per-channel 对称均匀量化（见 Equation 5-6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ、AWQ 等主流 LLM PTQ 方法默认使用 per-channel 对称均匀量化。在 GPU 推理时，dequantization 由 GEMM kernel 内的向量指令完成：每个 warp/block 加载对应的 scale 值，在 INT4→FP16 dequant 后执行 FP16 matmul。llama.cpp 的 GGUF 格式中，Q4_0/Q4_1/Q5_0 等为 group-wise 量化（比 per-channel 更精细），Q8_0 为 per-channel 对称量化。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
