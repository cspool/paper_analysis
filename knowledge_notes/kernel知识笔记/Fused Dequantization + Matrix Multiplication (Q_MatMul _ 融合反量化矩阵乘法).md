## Fused Dequantization + Matrix Multiplication (Q_MatMul / 融合反量化矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Dequantization + Matrix Multiplication（KIVI 中称为 Q_MatMul）是一种 CUDA kernel 优化技术，将低比特张量的反量化（dequantization）和矩阵乘法（matmul）在 GPU kernel 的 tiling 级别融合执行，避免将反量化后的大尺寸 FP16 中间结果写回 GPU 全局内存（HBM）。标准做法是先完整反量化整个量化张量到 FP16 存入 HBM，再执行 matmul 从 HBM 读取。这种做法对于 KV Cache Quantization 而言是巨大的浪费——2bit 量化的 KV Cache 反量化后膨胀 8×（2bit→16bit），相当于量化节省的内存带宽全被反量化浪费。

KIVI 中 Q_MatMul 在 GPU shared memory 内完成：将 query tile 和对应的 quantized KV cache tile 加载到 SRAM → 即时反量化 tile → 在 SRAM 中直接计算 tile 矩阵乘法 → 只将最终结果写回 HBM。避免了 FP16 中间 KV cache 的 HBM 写入和再次读取，有效利用量化减少的内存带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Q_MatMul kernel 伪代码（CUDA 实现）：

```
// 输入: t_Q ∈ R^{M × K} (FP16), Q_X ∈ int2 (packed), scales ∈ FP16, zeros ∈ FP16
// 输出: Out ∈ R^{M × N} (FP16)

__global__ void Q_MatMul(
    half* t_Q,           // [M, K] query tile in shared memory
    uint8_t* Q_X_packed, // [K/4, N] packed 2bit KV cache
    half* scales,        // [num_groups, N] per-group scale
    half* zeros,         // [num_groups, N] per-group zero-point
    half* Out,           // [M, N] output
    int G                // group size = 32
) {
    // 1. 加载 t_Q tile 到 SRAM (寄存器 + shared memory)
    __shared__ half Q_tile[TM][TK];
    // ...load t_Q into Q_tile...

    // 2. 加载 packed 2bit K tile 和 scale/zero 到 SRAM
    __shared__ half K_deq[TK][TN];   // dequantized K tile

    for each element in tile:
        byte = Q_X_packed[packed_idx];
        // 解包: 每 byte 存 4 个 2bit 值
        for bit_idx in [0, 1, 2, 3]:
            val_2bit = (byte >> (bit_idx * 2)) & 0x03;
            group_id = col / G;
            // 即时反量化: 2bit → FP16
            K_deq[row][col] = val_2bit * scales[group_id] + zeros[group_id];

    // 3. Tile matmul in SRAM (不写回HBM)
    for i in range(K):
        for m in range(TM):
            for n in range(TN):
                Out_tile[m][n] += Q_tile[m][i] * K_deq[i][n];

    // 4. 只将最终结果写回HBM
    // ...store Out_tile to Out...
}
```

KIVI 的完整 mixed-precision attention 流程使用两次 Q_MatMul：
1. `A_g = Q_MatMul(t_Q, Q(X_K_g))` — fused dequant+matmul for grouped key
2. `t_O_g = Q_MatMul(A_g_sm, Q(X_V_g))` — fused dequant+matmul for grouped value

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 使用 CUDA 实现 Q_MatMul。实现参考：(1) 映射 tile 到 CUDA thread block，每 block 处理一个输出 tile；(2) 使用 shared memory 缓冲 input tile 和 dequantized tile；(3) 反量化逻辑内联在 matmul 循环中，避免额外的 shared memory buffer。类似技术被广泛使用：FlashAttention 的 online softmax 融合、vLLM 的 fused kernel、CUTLASS 的 mixed-input GEMM。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
