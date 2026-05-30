## Sub-channel Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-channel quantization（子通道量化）也称为 block-wise quantization 或 group-wise quantization，是 LLM weight-only 量化的标准方法。它将权重矩阵按固定大小（block size）分组，每组独立计算一个量化缩放因子（通常是该组的 absmax），组内每个权重按该因子归一化后量化。与 per-tensor quantization（整个张量共享一个 scale）和 per-channel quantization（每行/每列一个 scale）相比，sub-channel quantization 在精度和存储开销之间取得平衡：更小的 block size → 更精准的局部归一化 → 量化误差更低，但需要存储更多 scale 因子。本论文使用 block size=128 作为默认值，因为它在精度和硬件对齐（大多数 MAC 单元可无需拆分累加即可处理 128 元素的 dot-product）之间取得折中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sub-channel weight-only 量化流程：

```
# W ∈ R^{d_out × d_in}
# Block size B=128 (沿 d_in 维度分组)
# 对于 INT4 量化：
W_flat = W.reshape(d_out, d_in // B, B)  # (d_out, n_blocks, B)
scales = zeros(d_out, n_blocks)            # per-block FP16 scales
W_q = zeros(d_out, n_blocks, B, dtype=int4)

for out_ch in 1..d_out:
    for block in 1..n_blocks:
        w_block = W[out_ch, block*B : (block+1)*B]
        s = max(|w_block|)                         # absmax scale
        scales[out_ch, block] = s
        W_q[out_ch, block] = round(w_block / s * Q_max)  # Q_max=7 for INT4

# 推理时解码:
Ŵ[out_ch, i] = scales[out_ch, i//B] * W_q[out_ch, i//B, i%B] / Q_max
```

本论文在 block size sweep 实验（Table 5）中发现：即使 block size 小到 16（超出当前 DNN 加速器的有效支持），各数据类型之间的 format 差异依然存在。例如 channel-wise（block size=d_in）下 E2M1-SP 比 INT4 平均准确率高 4.14%，block size=16 时差距仍为 1.59%。这表明数据类型的选择与 sub-channel 粒度相对独立。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sub-channel quantization 通过修改版 Intel Neural Compressor 库（论文）或 bitsandbytes/QLoRA 库实现。存储格式通常为：4-bit（或 N-bit）packed indices 数组 + per-block FP16/BF16 scale 数组。推理时使用 fused CUDA kernel：在寄存器中查表解码 → FP16 GEMM（或直接 INT4 GEMM for INT4）。论文指出 block size=128 是"足够大以对齐大多数 MAC 单元无需拆分累加"的平衡点。更小的 block size 虽然提升精度，但增加了 scale 存储开销（B=32: scale 占额外 1/32×16bit ≈ 0.5 bit/elem）和反量化调度的复杂度（需要更多 scale 加载和更细粒度的去量化操作）。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
