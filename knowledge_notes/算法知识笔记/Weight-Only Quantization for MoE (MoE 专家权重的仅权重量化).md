## Weight-Only Quantization for MoE (MoE 专家权重的仅权重量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Weight-Only Quantization 是一种只对模型权重进行量化、保持激活值为浮点精度的推理优化技术。在 MoE 场景下，选择仅量化 expert 权重而非所有参数（也不量化 activations），因为：(1) MoE 的 expert 权重占模型总参数 90% 以上，是内存的主要消耗者；(2) 只量化权重避免了 activation quantization 所需的 calibration，使得量化策略无需 post-training calibration 即可直接应用到不同语言家族；(3) 所有 activations 和 biases 保持 FP16，dequantized weights 也恢复到 FP16，因此矩阵乘法仍使用浮点运算，无需整数 Tensor Core。

论文 "Who Says Elephants Can't Run" 使用 symmetric range-based per-channel quantization：对每个 expert 权重矩阵 $W \in \mathbb{R}^{E \times M \times N}$（E 个专家，M×N 权重），沿输出 channel（N 维度）计算 per-channel scales $S \in \mathbb{R}^{E \times 1 \times N}$。INT8 使用 scale factor = max(|W[:,:,n]|) / 127，INT4 使用 max(|W[:,:,n]|) / 7。量化后权重加常量偏移（INT8: +128, INT4: +8）转为无符号数，简化后续 dequantize 的位操作。

从算法pipeline角度拆解术语：

MoE 模型的 INT4/INT8 推理 pipeline：
```
离线量化阶段（训练完成后一次性执行）：
for expert e in 0..E-1:
    for output_channel n in 0..N-1:
        max_abs = max(|W_fp16[e, :, n]|)
        S[e, 0, n] = max_abs / max_val_int     # 127 (INT8) or 7 (INT4)
        W_quant[e, :, n] = round(W_fp16[e, :, n] / S[e, 0, n])
        W_plus[e, :, n] = W_quant[e, :, n] + offset  # 128 or 8

在线推理阶段（每个 MoE layer 执行时）：
for each MoE layer:
    gate_logits = x @ W_router                  # FP16 matmul (Router)
    expert_idx, expert_scale = top_k(gate_logits, k=1)
    # Token routing: CUB radix sort + permute (FP16 activations)
    tokens_perm = radix_sort_and_permute(x, expert_idx)
    # Fused GEMM + Dequantize per expert
    for expert e with active tokens:
        W_deq = int_to_fp16_fast(W_plus[e] - offset)
        W_deq = W_deq * S[e]                    # FP16 乘 scale
        out_e = tokens_e @ W_deq                # FP16 GEMM
    output = unpermute_and_scale(out, expert_scale)
```
核心 insight：不量化 activation 避免了 calibration，所有中间结果保持 FP16，只有 weight load 和 dequantize 在 GEMM 内部 fused 处理。

术语一般如何实现？如何使用？

通常配合 CUTLASS 或 cuBLAS 实现 fused kernel。LLM.int8() 使用混合精度分解，GPTQ 使用 optimal brain quantization。MoE 场景中 weight-only 量化尤其有效——expert 权重冗余度高（大量独立 expert FFN 参数），量化 bit 损失被稀释。当前论文显示 INT4 实现 8× 模型压缩（5B→~625MB expert weights），INT4 GEMM 最高 1.85× 加速，BLEU 质量几乎无损（10 语言对平均 ΔBLEU = -0.167）。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
