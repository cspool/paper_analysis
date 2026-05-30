## GPTQ (Generative Pre-Trained Transformer Quantization)

术语解释
GPTQ 是一种基于近似二阶（Hessian-based）信息的 one-shot post-training weight-only 量化方法，能够在 3-4 比特精度下将 LLM 权重压缩且保持较高精度，广泛用于 MoE 模型的 expert 量化以降低参数传输和计算开销。

术语是什么？
GPTQ 的核心思想是逐列（column-wise）量化权重矩阵，利用 Hessian 矩阵的逆（用 calibration 数据的激活值估计）作为重要性度量，依次量化每一列权重并补偿之前列的量化误差，在保证整体 MSE 最小的前提下完成所有列的量化。相比于 naive rounding (RTN)，GPTQ 能在 3/4-bit 下保持接近 FP16 的 perplexity。

MoE 推理中的 GPTQ 应用场景：
- **Expert 量化减小传输**：在 expert offloading 场景中，将 expert 权重量化为 3/4-bit 后，PCIe 传输数据量减少 4-5×
- **NDP 计算加速**：在 GPU-NDP 系统中，NDP 设备使用量化 expert 权重执行 FFN 计算，降低 NDP 计算压力
- **多精度缓存**：预计算每个 expert 的 1/2/3/4-bit GPTQ 量化版本，推理时根据重要性动态选择精度

从算法pipeline角度拆解术语：
GPTQ 在 context-aware MoE-NDP 推理中的使用流程：

```
=== 离线阶段：多精度 GPTQ 量化 ===
输入: expert weights W_{l,e} ∈ R^{d_ff × d} (FP16), calibration data D_cal
输出: 预缓存的 1/2/3/4-bit 量化 replicas + per-bitwidth loss table L_{l,e}(b)

for each expert e in MoE model:
    for b in {1, 2, 3, 4}:
        # GPTQ column-wise quantization
        H = 2 * X_cal^T @ X_cal       # Hessian from calibration activations
        H_inv = Cholesky(H)^(-1)       # Inverse Hessian
        
        for col_j in range(d):
            # Quantize column j to b bits
            w_q[j] = quantize(W[:,j], b)  # round to nearest b-bit level
            error = (W[:,j] - w_q[j]) / H_inv[j,j]
            
            # Compensate remaining columns
            for col_k in range(j+1, d):
                W[:,k] -= error * H_inv[j,k] / H_inv[j,j]
        
        W_q(b) = group_and_scale(w_q)  # 应用 per-group scaling
        L_{l,e}(b) = MSE(W_q(b)(X_cal), W_fp16(X_cal))
```

```
=== 在线阶段：根据重要性选择精度 ===
解码前:
    # 从 pre-cached replicas 中按分配的 bitwidth 选择
    for expert e in NDP_resident_experts:
        b_e = bitwidth_assignment[l][e]  # 1/2/3/4 bit
        W_active[e] = load_cached(W_q_{l,e}, b_e)

解码时:
    # NDP 使用量化权重执行 FFN
    for each selected expert e on NDP:
        output = quantized_ffn(activation, W_active[e], bitwidth=b_e)
        # 低 bitwidth: 更少的内存访问和计算，但精度更低
```

术语一般如何实现？如何使用？
- 开源实现: https://github.com/IST-DASLab/gptq
- 基于 PyTorch，支持 OPT, LLaMA, BLOOM 等模型
- 量化过程约 4 GPU-hours for 175B model
- 典型配置: group_size=128, 4-bit weight, 使用 calibration dataset (如 C4 或 WikiText-2)
- 优化: GPTQ 使用 GPU kernel 加速，量化后模型可通过定制 CUDA kernel 实现 3-4× 推理加速
- 局限性: 极限低位 (1-2 bit) 下精度下降显著，需配合补偿方法 (如 LoRC, QLoRA)；不均匀的 per-expert 敏感性导致统一精度不够高效

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
