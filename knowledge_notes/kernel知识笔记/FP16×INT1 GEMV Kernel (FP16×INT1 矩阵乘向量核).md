## FP16×INT1 GEMV Kernel (FP16×INT1 矩阵乘向量核)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FP16×INT1 GEMV 是 TailorKV 中为 1-bit 量化 KV cache 设计的 CUDA kernel，执行 FP16 query 向量（q ∈ R^{d_h}）与 INT1 key/value 矩阵（K/V ∈ R^{n × d_h}，每个元素 1-bit）的矩阵乘向量运算。该 kernel 将 dequantization 和 attention score 计算融合在一个 kernel 内，避免先解量化到 FP16 再计算的两步开销。1-bit 量化将每个 FP16 值（16 bit）映射为单个 bit（0 或 1），提供理论 16× 压缩比。配合 group-wise 量化参数（group_size=64, zero-point z 和 scaler s 各占 FP16），有效 bit-width 约为 1 + 32/64 = 1.5 bit per element。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// FP16×INT1 GEMV: q ∈ R^{d_h}, K_1bit ∈ {0,1}^{d_h × n}
// 量化参数: scale ∈ R^{d_h}, zero ∈ R^{d_h} (per-channel key 量化)
// 或: scale ∈ R^n, zero ∈ R^n (per-token value 量化)

// === Kernel 伪代码 (per threadblock) ===
__global__ void fp16_int1_gemv_kernel(
    half* q,           // [d_h] FP16 query
    uint32_t* K_packed,// [ceil(d_h/32) * n] INT1 packed (32 elements per uint32)
    half* scale,       // [n] per-token 或 [d_h] per-channel scale
    half* zero,        // [n] per-token 或 [d_h] per-channel zero
    half* output,      // [n] attention scores
    int d_h, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float acc = 0.0f;
    // 每个 warp 处理一个 token 的所有 channel
    for (int c = 0; c < d_h; c += 32) {
        // 1. 从 packed INT1 解包 32 个 channel
        uint32_t packed = K_packed[tid * ceil(d_h/32) + c/32];
        
        // 2. 逐 bit 解包 → 乘 query 值 → 累加（fused dequant+GEMM）
        #pragma unroll
        for (int b = 0; b < 32 && (c + b) < d_h; b++) {
            float k_bit = (float)((packed >> b) & 0x1);   // 0 或 1
            float k_val = k_bit * scale[tid] + zero[tid];  // dequantize
            acc += __half2float(q[c + b]) * k_val;         // FMA
        }
    }
    output[tid] = __float2half(acc);
}
```

与标准 FP16×FP16 GEMV 相比，FP16×INT1 的 FLOP 量减少（输入 operand 为 1-bit），主要收益在内存带宽：从 GPU DRAM 读取的 K 矩阵数据量仅为 FP16 的 1/16，使 memory-bound 的 decoding 阶段显著加速。

术语一般如何实现？如何使用？

实现建议：(1) 使用 CUDA C++ 编写，利用 `uint32_t` bit packing 存储 32 个 INT1 值；(2) Per-channel 量化用于 key cache（outlier 沿 channel 集中，per-channel 隔离 outlier channel），per-token 量化用于 value cache（无显著 outlier pattern）；(3) 与 FlashAttention 集成——在 quantization-friendly 层，先通过 FP16×INT1 GEMV 计算 attention scores，再将 dequantized V 与 scores 做矩阵乘（也可用 FP16×INT1 GEMV 模式）。

适用场景：极低精度（1-bit/2-bit）KV cache 量化的 decoding 阶段，特别是需要将压缩比推到极限（16× 以上）的长上下文场景。TailorKV 仅在 1-2 个 quantization-friendly 浅层使用此 kernel，其余层使用动态 token 检索策略。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

---
