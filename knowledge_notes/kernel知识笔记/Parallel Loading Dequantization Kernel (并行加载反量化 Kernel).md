## Parallel Loading Dequantization Kernel (并行加载反量化 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parallel Loading Dequantization Kernel 是 D2MoE 为 MWQ 专门设计的 CUDA 反量化 kernel，解决动态 bit-width 权重推理时 GPU 存储层级间并行效率不足的问题。传统方法先完整反量化 INT→FP16 写入 global memory，再启动 GEMM 读取——中间 FP16 结果往返 HBM 浪费带宽。D2MoE 的 kernel 同时优化三个维度：

1. **加载并行 (Loading Parallelism)**：从 NVMe SSD 到 GPU global memory 的量化权重传输 (cudaMemcpyAsync) 与激活值从 global memory 到 L2 cache 的移动并发
2. **计算并行 (Computation Parallelism)**：反量化操作（CUDA cores 上拆解 packed INT + 位操作）与矩阵乘法（Tensor cores 上 GEMM）通过独立 CUDA stream 重叠
3. **去量化优化**：使用 Any-Precision LLM 的二进制位操作代替传统 bit-transpose，直接解包 packed 整数到 FP16

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 D2MoE-V1 (b₁=2, b_K=4, group_size=128) 的 MWQ 反量化为例：

```
=== Parallel Loading Dequantization Kernel (CUDA Pseudocode) ===

// 输入: MWQ 量化权重（已在 GPU Global Memory）
//   Q_W_b1:  packed INT2 [s, h/8] (每 byte 存 4 个 INT2)
//   z_b1:    INT8 [s, h/128]  (per-group zero points)
//   s_b1:    FP16 [s, h/128]  (per-group scales)
//   Q_W_bk:  packed INT1 [s, h/8] (binary residual, k=2..K, 每 byte 8 bit)
//   s_bk:    FP16 [s, h/128]  (per-group scales for residual)
// 输出: Ŵ_bK ∈ FP16 [s, h] (dequantized expert weight, 立即送入 GEMM)

__global__ void MWQ_Dequant_Kernel(
    uint8_t* Q_W_b1,        // packed INT2 base
    int8_t*   z_b1,          // per-group zero points
    half*     s_b1,          // per-group scales
    uint8_t* Q_W_bk_packed, // packed binary residuals (k=2..K)
    half*     s_bk,          // residual scales (k=2..K)
    half*     W_deq,         // output: dequantized FP16 weight
    int s, int h, int group_size = 128, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= s || col >= h) return;
    
    int group_id = col / group_size;
    
    // Step 1: 解包 INT2 base
    int byte_idx = (row * h + col) / 4;  // 4 elements per byte for INT2
    int bit_offset = (col % 4) * 2;
    uint8_t byte_val = Q_W_b1[byte_idx];
    int q_val_b1 = (byte_val >> bit_offset) & 0x03;  // extract 2 bits
    
    // Step 2: Asymmetric dequantization (INT2 → FP16)
    float w_acc = (float)(q_val_b1 - z_b1[group_id]) * __half2float(s_b1[group_id]);
    
    // Step 3: 叠加 binary residuals (k=2..K)
    for (int k = 2; k <= K; k++) {
        int residual_byte = (row * h + col) / 8;  // 8 elements per byte for 1-bit
        int bit_shift = col % 8;
        uint8_t residual_byte = Q_W_bk_packed[(k-2) * s * h/8 + residual_byte];
        // 提取 1 bit → 映射到 {+1, -1}
        int q_val_bk = ((residual_byte >> bit_shift) & 0x01) ? 1 : -1;
        w_acc += (float)q_val_bk * __half2float(s_bk[(k-2) * s * h/128 + group_id]);
    }
    
    W_deq[row * h + col] = __float2half(w_acc);  // → L2 cache → Tensor core GEMM
}

=== CUDA Stream Orchestration (Triton-level) ===

// Stream 1 (I/O): cudaMemcpyAsync, disk→global memory
// Stream 2 (Compute): dequantization + GEMM

// Triton 协调:
// 当 Stream 1 加载 batch N 的 Q_W 时
// Stream 2 同时执行 batch N-1 的 dequant + GEMM
// 这种双缓冲策略最大化 GPU 利用率

for batch_id in range(num_batches):
    io_event = cudaMemcpyAsync(Q_W_tensors[batch_id], disk_ptr, size, H2D, io_stream)
    if batch_id > 0:
        wait(comp_event[batch_id-1])  # 等前一批 GEMM 完成
        cudaEventRecord(comp_event[batch_id], comp_stream)
        MWQ_Dequant_Kernel<<<grid, block, 0, comp_stream>>>(...)
        cuBLAS_GEMM<<<..., comp_stream>>>(W_deq, activation, output)
    sync()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 基于 NVIDIA Ampere/Ada Lovelace 架构实现。Kernel 使用 CUDA C++ 编写，与 PyTorch 通过 `torch.utils.cpp_extension` 集成。位操作优化参考自 Any-Precision LLM (Park et al., ICML 2024)，用直接位掩码和移位操作代替传统 int→float bit-transpose 方法，处理速度显著提升。

dequantization kernel 开销分析（Figure 12）：
- 4 requests 时：计算开销 ~20.77%，延迟开销 ~18.56%
- 32 requests 时：计算开销 ~16.77%，延迟开销 ~5.3%（因 MWQ 嵌套结构使 base 反量化结果被更多 request 复用）
- 临时 FP16 中间内存立即释放，对峰值内存影响极小

适用于：端侧设备（RTX 3060/AGX Orin）上需要动态 bit-width 的 MoE 模型推理，KV Cache 量化的反量化 kernel（类似 KIVI 的 Q_MatMul，但 D2MoE 针对的是 expert 权重而非 KV Cache）。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
