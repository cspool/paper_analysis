## LUT-Based Dequantization Kernel (Non-Uniform Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-Based Dequantization Kernel 是用于非均匀量化权重推理的 CUDA kernel。权重以 b-bit indices（3/4-bit）存储，每个 output channel 对应一个 LUT（Look-Up Table，包含 2^b 个 FP16 centroid 值）。Kernel 在 GPU 上加载 packed bit indices → 逐 piece 查表还原为 FP16 权重 → 与 FP16 activation vector 进行矩阵-向量乘法（GEMV）。关键设计：(1) piece-by-piece dequantization 以减少寄存器压力和最大化内存带宽利用；(2) 所有算术在 FP16 执行；(3) LUT 存储在 GPU shared memory 或寄存器中以减少查表延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// CUDA kernel: LUT-based dequant + GEMV
// grid: num_output_channels / BLOCK_SIZE
// each thread handles one output row

__global__ void lut_dequant_matvec_kernel(
    const uint32_t* packed_indices,  // [out_c × chunks_per_row]
    const half* LUTs,                 // [out_c × k], k=2^bit FP16 centroids
    const half* activation,           // [in_features] FP16
    half* output                      // [out_features]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    half* lut_row = LUTs + row * k;  // 当前 channel 的 LUT
    half acc = 0.0;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        uint32_t packed = packed_indices[row * num_chunks + chunk];
        // 逐元素提取 b-bit index, LUT查表, FP16乘累加
        for (int j = 0; j < indices_per_chunk; j++) {
            int idx = (packed >> (j * BIT_WIDTH)) & ((1 << BIT_WIDTH) - 1);
            half w_deq = lut_row[idx];               // LUT查表
            acc += w_deq * activation[global_col++]; // FP16 FMA
        }
    }
    output[row] = acc;
}
```

延迟分析（A6000, LLaMA-7B, 128 tokens, 3-bit）：
- FP16 baseline (no quant): 3.2s
- Uniform quant (GPTQ, no group): 1.4s
- LUT non-uniform (SqueezeLLM): 1.5s (+7% vs uniform, 2.1x vs FP16)
→ LUT overhead 极小，因为推理是 memory-bound（memory bandwidth 掩盖了 LUT 查表计算）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源实现：https://github.com/SqueezeAILab/SqueezeLLM (CUDA kernels)。实现要点：(1) BIT_WIDTH=3 时用 custom bit extraction（CUDA 无原生 3-bit 类型），将 3-bit indices 紧密 pack 进 uint32；(2) LUT 大小：对 4096 output channels × 8 centroids × 2 bytes = 64KB per layer——可放入 L1 cache/shared memory；(3) 相比 uniform dequant（只需 scale × int + zero），LUT 多了一次 memory read（LUT lookup），但这在 memory-bound 场景下几乎不增加 wall-clock time；(4) 与 uniform quant kernel 的关键区别：uniform kernel 按 group 读取 scale/zero point → linear dequant，非均匀 kernel 按 element 或 sub-chunk 读取 index → LUT-based dequant。非均匀量化 kernel 的通用性：可用于任何基于 codebook/centroid 的量化方案（如 GPTVQ 的 1D VQ、NF4 等）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
