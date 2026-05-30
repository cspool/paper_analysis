## QuTLASS

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuTLASS 是 MR-GPTQ 论文配套发布的高性能低精度量化 kernel 库，基于 NVIDIA CUTLASS 构建，专为 Blackwell GPU（SM100/SM120）优化。提供两类 kernel：(1) Quantization-related kernels——轻量级 fused kernel 实现在线 block-wise 旋转 + 量化 + scale 计算的融合，支持 k∈{16,32,64,128} 的 block diagonal 矩阵旋转，通过自定义 epilogue function 将量化/scale 直接集成进变换 kernel；(2) Matmul-related narrow precision kernels——处理 FP4 量化与 tcgen05.mma 矩阵乘间的 scale 重排（硬件强制的 block scaling factors layout），通过 Triton kernel 实现，支持 CUTLASS 和 FlashInfer 多后端灵活切换。

QuTLASS 的关键设计：对 k<256 的 block 旋转，dense 变换为 memory-bound，因此旋转矩阵可以运行时从内存加载（任意矩阵，不限于 Hadamard），所有旋转矩阵几乎同成本。量化方法通过模板设计支持 MSE 和 Abs-Max，便于扩展。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuTLASS 中 FP4 推理的完整 kernel 执行流程：
```
// Kernel 1: Fused Online Rotation + Quantization (CUDA)
// 输入: FP16 activation X [M, K], block-diag matrix H_k in memory
// 输出: MXFP4 quantized activation X_q + per-group scales
__global__ void fused_rotate_quantize_mxfp4(
    half* X, half* H_k,        // H_k: k×k per block, row-major
    uint8_t* X_q_packed,        // 输出: E2M1 packed (2×4-bit/byte)
    uint8_t* scales_E8M0,       // 输出: E8M0 power-of-two scales
    int M, int K, int k         // k = group_size = 32
) {
    // Step 1: Block-wise Hadamard rotation (memory-bound for k<256)
    // X_rot[b, :k] = X[b, :k] @ H_k  per block
    // k 小时每个 thread 处理的 FLOPs 少于 bytes loaded
    
    // Step 2: Per-group absmax scale (fused epilogue)
    // s_G = max(|X_rot[b, g*k : (g+1)*k]|) for each group g
    // s_G_q = round_to_power_of_two(s_G)  // E8M0 quantization
    
    // Step 3: E2M1 quantization (fused epilogue)
    // x_norm = X_rot / s_G_q
    // x_fp4 = RTN_E2M1(x_norm)   // 4-bit E2M1 format
    // Pack 2×4-bit into 1 byte
}

// Kernel 2: Scale Rearrangement (Triton)
// 输入: per-group scales in natural order
// 输出: scales rearranged for tcgen05.mma layout
// 原因: NVIDIA Blackwell tcgen05.mma 要求特定的 block scaling factors layout
// 参照 cuBLAS doc: block-scaling-factors-layout

// Kernel 3: FP4 Matrix Multiplication (CUTLASS/FlashInfer backend)
// 输入: W_q (MXFP4 packed), X_q (MXFP4 packed), rearranged scales
// 调用: tcgen05.mma (Blackwell hardware instruction)
// 输出: FP16/BF16 output activation
```

B200 单层 throughput 结果（Llama-3.3-70B 典型层形状）：
- "Ideal": 仅 FP4 matmul（不含旋转/量化/scale 开销）= ~4× vs FP16
- "Actual" (QuTLASS): 含全部开销 = ~3.6× vs FP16（MXFP4）
- MXFP4 比 NVFP4 高 ~15% throughput（power-of-two scales 降低硬件开销）
- RTX 5090: 6× layer-wise（ideal 8×），4× end-to-end

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/IST-DASLab/qutlass。基于 CUTLASS（https://github.com/NVIDIA/cutlass）构建，利用 CUTLASS 的 epilogue fusion 机制将量化和 scale 操作融合进 rotation kernel。FlashInfer 后端支持（https://github.com/flashinfer-ai/flashinfer）。Triton kernel 用于 scale rearrangement（硬件强制的 layout 转换）。支持 MXFP4 和 NVFP4 两种格式。集成进 vLLM 进行端到端推理评估：Llama-3.3-70B B200 端到端 2.2× speedup vs BF16，RTX 5090 端到端 4× speedup。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization

---
