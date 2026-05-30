## MR-GPTQ (Micro-Rotated GPTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MR-GPTQ（Micro-Rotated GPTQ）是专为 FP4 微缩放格式（MXFP4/NVFP4）设计的 GPTQ 变体，包含三个技术组件：(1) MSE-Optimized Grids——交替优化 per-tensor 和 per-group scales 最小化量化 MSE；(2) Static Activation Reordering——先确定 grid/scales 再按 Hessian 重排列列、量化后恢复原序，避免 dynamic act-order 的 10-20% 推理开销；(3) Block-wise Hadamard Rotations——以 group size 为单位进行旋转（"micro-rotation"），对大 G 格式（MXFP4 G=32）降低 per-element MSE，旋转融合入权重零推理开销。提供三种策略变体：MR-GPTQ-MXFP4（GPTQ + rotated MXFP4）、MR-GPTQ-NVFP4（GPTQ + rotated NVFP4 + MSE-optimized grid）、GPTQ + standard NVFP4（无旋转）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MR-GPTQ-MXFP4 离线量化
for each Linear layer:
    W_fp16 = layer.weight
    H_k = block_diag_hadamard(k=32)           # k = group_size
    
    # 1. 权重旋转（离线预融合）
    W_rot = W_fp16 @ H_k                       # Laplace → Normal
    
    # 2. MSE grid（MXFP4: 统一静态 s_T）
    s_T, s_G = MSE_optimize_scales(W_rot)
    
    # 3. Static act-order
    col_order = argsort(H_diag, descending=True)
    W_rot_ordered = W_rot[:, col_order]
    
    # 4. GPTQ per-column (OBS framework)
    for col in range(d_in):
        w_q = MXFP4_quantize(W_rot_ordered[:, col], s_T, s_G)
        error = (W_rot_ordered[:, col] - w_q) / H_inv[col, col]
        W_rot_ordered[:, col+1:] += error * H_inv[col, col+1:]
    
    # 5. 恢复原始列序
    W_q = inv_permute_cols(W_rot_ordered, col_order)

# 推理: 权重旋转已融合 → activation fused rotation (QuTLASS) → FP4 matmul
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：FP-Quant（https://github.com/IST-DASLab/FP-Quant）。量化模型托管于 HuggingFace（ISTA-DASLab/mr-gptq collection）。校准：FineWeb 1024 sequences + Hessian dampening λ=10^-2。关键结果：Llama-3.1-8B MXFP4 MR-GPTQ Avg Recovery 93.31%（RTN=87.83%，GPTQ=89.47%），将 MXFP4 与 NVFP4 精度差缩至 1-2%。大模型（70B+）两种格式均可恢复 98-99% FP16 精度。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization
