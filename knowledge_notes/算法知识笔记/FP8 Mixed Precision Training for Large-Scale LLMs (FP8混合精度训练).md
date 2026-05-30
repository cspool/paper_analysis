## FP8 Mixed Precision Training for Large-Scale LLMs (FP8混合精度训练)

术语解释
FP8 Mixed Precision Training 是 DeepSeek-V3 提出的低精度训练框架，首次在超大规模模型（671B）上验证 FP8 训练的有效性。核心方案：将大部分 GEMM 操作（Fprop/Dgrad/Wgrad）在 FP8 E4M3 格式下执行，通过 fine-grained quantization（activation: 1×128 tile-wise, weight: 128×128 block-wise）和 CUDA Core FP32 promotion 解决 H800 Tensor Core 仅 14-bit 累积精度的硬件限制。BF16 → FP8 训练的 relative loss error <0.25%。

术语是什么？
DeepSeek-V3 FP8 框架的关键技术：(1) **Fine-grained quantization**：activation 按 1×128 tile 分组缩放（per token per 128 channels），weight 按 128×128 block 分组缩放，优于传统 tensor-wise scaling；(2) **Increased accumulation precision**：每 N_c=128 个 Tensor Core WGMMA 结果拷贝到 CUDA Core 做 FP32 完整精度累积+dequantization（scaling factor 乘法融合），两个 warpgroup 交替执行；(3) **E4M3 for all tensors**：不使用 E5M2，通过 fine-grained scaling 弥补动态范围不足；(4) **Online quantization**：每 tile/block 实时计算 max absolute value 确定 scaling factor，不使用历史值；(5) **Low-precision storage**：BF16 optimizer states（first/second moments），FP8 cached activations（E5M6 for attention inputs），FP8 dispatch activations。

从算法pipeline角度拆解术语：
```
=== FP8 GEMM Forward with Fine-Grained Quantization ===

Input: X [M, K] in BF16, W [K, N] in BF16

// Quantization
for tile i in 1..(K/128):            // 1×128 tile-wise for activation
    scale_X[i] = max(|X[:, i*128:(i+1)*128]|) / 448.0  // FP8 E4M3 max
    X_FP8[:, i*128:(i+1)*128] = X[:, i*128:(i+1)*128] / scale_X[i]

for block (i,j) in (K/128)×(N/128):  // 128×128 block-wise for weight
    scale_W[i,j] = max(|W[i*128:(i+1)*128, j*128:(j+1)*128]|) / 448.0
    W_FP8[block] = W[block] / scale_W[i,j]

// MMA with CUDA Core Promotion (alternating warpgroup pairs)
for k_step in 0..(K/128-1):
    partial_k = WGMMA(X_FP8_k, W_FP8_k)   // Tensor Core, ~14-bit accumulation
    if (k_step+1) % 4 == 0:  // every N_c=128 elements (4 WGMMAs)
        C += FP32_promote(partial_sum) * scale_X * scale_W  // CUDA Cores
    else:
        accumulate_in_tensor_core(partial_sum)  // limited precision

// High-precision retained for: embedding, output head, MoE gating,
// normalization, attention operators
```

术语一般如何实现？如何使用？
训练速度理论 2× over BF16。DeepSeek-V3 每 trillion tokens 仅需 180K H800 GPU hours。FP8 通信优化：MoE up-projection 前 activation 量化为 FP8 → dispatch → FP8 Fprop。组合通信保留 BF16。H800 FP8 Tensor Core GEMM 默认累积精度仅 ~14 bits（K=4096 时最大相对误差近 2%），通过 CUDA Core promotion 解决。与 NVIDIA TransformerEngine 不同，不使用 delayed scaling。未来硬件建议：Tensor Core 原生支持 tile/block-wise group scaling + FP32 累积精度。与 microscaling 格式（MXFP）理念一致。

涉及论文标题：
- DeepSeek-V3 Technical Report
