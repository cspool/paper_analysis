## Bit-shift for Power-of-Two Scaling（2 的幂次缩放中的位偏移操作）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-shift for Power-of-Two Scaling 是 DMQ 在 CUDA kernel 中实现的一种将 power-of-two 缩放等价转换为整数左移操作的低开销硬件优化技术。核心原理：在 W4A8 量化推理中，PTS（Power-of-Two Scaling）对激活施加通道级 2^δ 缩放，矩阵乘法中 2^δ 因子等价于对量化权重 W̃ 执行左移操作：W̃^{shifted}_{kj} = W̃_{kj} ≪ δ_k = W̃_{kj} × 2^{δ_k}。由于现代 GPU 没有原生的 "multiply-bitshift-add" 融合指令，直接在 multiply-accumulate 路径中插入 shift 会低效。DMQ 将 shift 操作放在权重加载阶段——权重从 packed INT4 解包后、进入 GEMM 累加前，在寄存器中完成位偏移。这样 shift 操作不进入 MAC 流水线，每 bit 的 shift 仅需约 1 个 cycle。DMQ 验证了该策略的实际效率：自定义 CUDA kernel 在 M=3072 时相比 PyTorch FP32 GEMM 达到 5.17× 加速，bit-shift 开销极小且被 GEMM 的 memory-bound 特性掩盖。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 DMQ 的 W4A8 GEMM kernel with bit-shift 为例：

```cuda
// 自定义 CUDA kernel: fused Quant + Bit-shift + GEMM + Dequant (W4A8)
// 输入: packed_W [C_out][C_in/2] INT4, X [B][C_in] INT8
//       s_X scalar, s_W [C_out], delta [C_in] (PTS exponents)
// 输出: Y [B][C_out] FP32

__global__ void gemm_w4a8_bitshift(
    const uint8_t* packed_w,   // INT4 packed weights
    const int8_t* input,        // INT8 quantized activations
    const int* delta,           // PTS shift amounts per channel [C_in]
    float* output,              // FP32 output
    int B, int C_in, int C_out,
    float s_x, const float* s_w
) {
    int row = blockIdx.x;  // output row (batch element)
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output column
    
    if (col >= C_out) return;
    int accum = 0;  // INT32 accumulator
    
    for (int k = 0; k < C_in; k += 2) {
        // Step 1: Load packed INT4 weight byte
        uint8_t byte = packed_w[col * (C_in/2) + k/2];
        
        // Step 2: Unpack two 4-bit weights (sign-extend to INT8)
        int8_t w0 = (int8_t)((byte & 0x0F) << 4) >> 4;  // sign extend
        int8_t w1 = (int8_t)((byte >> 4) << 4) >> 4;
        
        // Step 3: BIT-SHIFT — apply PTS factor 2^{delta[k]}
        // Done in register, NOT in MAC path
        w0 = w0 << delta[k];    // = w0 * 2^{delta[k]}
        w1 = w1 << delta[k+1];
        
        // Step 4: INT8 MAC (fused into standard MAD instructions)
        accum += (int)w0 * (int)input[row * C_in + k];
        accum += (int)w1 * (int)input[row * C_in + k+1];
    }
    
    // Step 5: Dequantization
    output[row * C_out + col] = s_x * s_w[col] * (float)accum;
}
```

关键设计点：
- **Shift 位置**：在权重解包后、MAC 前（不在 MAC 内部），避免打破 GPU 的 MAD 指令融合
- **寄存器中完成**：shift 结果仅存在寄存器中，不写回 shared memory
- **选择性应用**：δ_k = 0 时 shift amount = 0，即 w << 0 = w（无操作）
- **每个 shift 仅 1 cycle**，对 GEMM 的整体 memory-bound 特性影响微乎其微

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-shift 在 GPU 上通过 PTX 指令 `shl.b32` 或 CUDA C 直接使用 `<<` 运算符实现。DMQ kernel 中 PTS 仅应用于 skip connection 层（网络总层数的 ~10-15%），整体延迟增加可忽略。Section E 的延迟测量：M=3072 时 kernel 延迟远低于 FP32 GEMM baseline，bit-shift 开销被 GEMM 主循环的 memory 开销完全覆盖。该技术也适用于其他 power-of-two 量化的硬件部署场景（如 ARM NEON 的 `vshlq_s32` 或 x86 AVX-512 的 `vpslld`），尤其适合边缘设备中乘法器受限的场景。

涉及论文标题：
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

---
