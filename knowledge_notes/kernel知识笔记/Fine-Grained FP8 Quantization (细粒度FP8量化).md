## Fine-Grained FP8 Quantization (细粒度FP8量化)

术语解释
Fine-Grained FP8 Quantization 是 DeepSeek-V3 提出的低精度训练量化策略，通过在小粒度元素组（1×128 tile for activation, 128×128 block for weight）级别进行缩放，解决 FP8 格式动态范围有限导致的量化误差问题。与传统的 per-tensor scaling 不同，fine-grained scaling 使每个小 group 有独立的 scaling factor，从而更好地容纳 outlier 值（如激活中的 massive activations）。

术语是什么？
量化粒度：(1) Activation: 1×128 tile（per token per 128 channels），即每个 token 沿 hidden dimension 每隔 128 channels 用一个 scale；(2) Weight: 128×128 block（per 128 input channels per 128 output channels）。Scale factor 沿 GEMM inner dimension K 方向，非标准 FP8 GEMM 直接支持，需结合 CUDA Core promotion 实现：每 N_c=128 个 WGMMA 结果取出，在 CUDA Core 上乘 scale factor 并做 FP32 累积。

从kernel调度角度拆解术语：
```
=== Fine-Grained FP8 Quantization + GEMM ===

// Activation quantization (1×128 tile-wise, online)
// X_BF16: [M, K], batch M tokens × inner dim K
for i in 0..M-1:                        // per token
    for j in 0..(K/128)-1:              // per 128-channel tile
        tile = X_BF16[i, j*128:(j+1)*128]
        scale_X[i,j] = max(abs(tile)) / 448.0   // E4M3 max = 448
        X_FP8[i, j*128:(j+1)*128] = tile / scale_X[i,j]

// Weight quantization (128×128 block-wise, online)
// W_BF16: [K, N]
for i in 0..(K/128)-1:
    for j in 0..(N/128)-1:
        block = W_BF16[i*128:(i+1)*128, j*128:(j+1)*128]
        scale_W[i,j] = max(abs(block)) / 448.0
        W_FP8[i*128:(i+1)*128, j*128:(j+1)*128] = block / scale_W[i,j]

// FP8 GEMM with Scaled Accumulation
// C = X_FP8 × W_FP8, with per-group scaling
C = zeros([M, N], FP32)
for k_step in 0..(K/128)-1:
    // WGMMA: [M, 128] × [128, N] → [M, N] partial (Tensor Cores, ~14-bit)
    partial = WGMMA(X_FP8[:, k_step*128:(k_step+1)*128],
                     W_FP8[k_step*128:(k_step+1)*128, :])
    if (k_step+1) % 4 == 0:              // N_c=128 elements interval
        // CUDA Core FP32 promotion + dequantization
        scale = scale_X[:, k_step//4] * scale_W[k_step//4, :]  // broadcast
        C += FP32_promote(accumulator) * scale
    else:
        accumulate(TensorCore, partial)  // limited precision
```

术语一般如何实现？如何使用？
与 microscaling (MX) 格式理念一致（NVIDIA Blackwell 已宣布支持）。H800 上实现限制：Tensor Core 不支持 per-group scaling，需额外 CUDA Core 步骤；频繁的 Tensor Core ↔ CUDA Core 数据移动限制效率。未来硬件建议：Tensor Core 直接接收 scaling factors，在 MMA 内部完成 group-scaled 累积+dequantization。在线量化效率问题：需从 HBM 读取 BF16 值进行量化，再写 FP8 回 HBM，建议融合 FP8 cast + TMA access 为单 fused 操作。与 TransformerEngine 的 delayed scaling 不同：DeepSeek-V3 使用 online max 计算替代历史值推断，更精确。

涉及论文标题：
- DeepSeek-V3 Technical Report
