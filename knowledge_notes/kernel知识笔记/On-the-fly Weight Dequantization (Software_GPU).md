## On-the-fly Weight Dequantization (Software/GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
On-the-fly Weight Dequantization（在线权重反量化）是 TinyChat/AWQ 推理系统中将 4-bit 量化权重在 GPU/CUDA kernel 内部实时恢复为 FP16 的技术。与传统的"先反量化到显存再计算"不同，on-the-fly 方式在 GEMM/GEMV 主循环内部完成反量化——权重从 DRAM 以 packed INT4 格式读取到寄存器，在寄存器中解包并乘以 group-wise Δ（量化 scale），得到 FP16 权重值后立即参与 FMA 运算，然后丢弃（不写回 DRAM）。这样避免了将 4× 数据量的 FP16 反量化权重写回 DRAM，将 decode 阶段的 arithmetic intensity 从 ≈1 提升至 ≈4 FLOPs/Byte（RTX 4090 上峰值性能上限从 ~1 TFLOPS 升至 ~4 TFLOPS）。TinyChat 同时为矩阵-矩阵乘（prefill）和矩阵-向量乘（decode）实现了融合 dequantization kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Llama-2-7B decode 阶段（batch_size=1）的 W4A16 on-the-fly dequantization GEMV kernel 为例：
```cuda
// TinyChat CUDA kernel: fused INT4-dequant + FP16-GEMV
// 输入: packed_weights (INT4, 每 2 weights/byte), scales (FP16, per-group)
// 输入: activation [1, C_in] FP16
// 输出: output [1, C_out] FP16

__global__ void gemv_w4a16_dequant_fused(
    const uint8_t* packed_w,   // [C_out, C_in/2]
    const half* scales,        // [C_out, C_in/group_size]
    const half* input,         // [C_in]
    half* output,              // [C_out]
    int C_in, int C_out, int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // output row
    if (row >= C_out) return;

    float acc = 0.0f;
    int group_id = 0;

    for (int j = 0; j < C_in; j += 2) {
        // Step 1: 加载 packed byte
        uint8_t byte = packed_w[row * (C_in/2) + j/2];
        
        // Step 2: 解包两个 4-bit 权重
        int8_t w0 = (int8_t)(byte & 0x0F);       // 低 4-bit
        int8_t w1 = (int8_t)((byte >> 4) & 0x0F); // 高 4-bit
        // INT4 有符号范围: [-8, 7], 解包时做 sign extension
        
        // Step 3: 加载 group scale 并反量化
        if (j % group_size == 0) {
            half scale = scales[row * (C_in/group_size) + group_id++];
        }
        half w0_fp = __half2float(w0) * scale;
        half w1_fp = __half2float(w1) * scale;
        
        // Step 4: FMA 累加 (读到即算，不存回 DRAM)
        acc += w0_fp * __half2float(input[j]);
        acc += w1_fp * __half2float(input[j+1]);
    }
    output[row] = __float2half(acc);
}
```

关键设计决策：
- 权重读入寄存器后立即解包 → 反量化 → FMA，中间结果仅存于寄存器，不写回 shared memory / DRAM
- Group scale 按需加载（每 group_size=128 个权重加载一次），不占用过多寄存器
- 对于 batch_size > 1 的 GEMM 场景（prefill），可复用解包后的权重到 shared memory 中供多个 activation row 使用

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TinyChat 在 PyTorch 中通过 CUDA extension 实现 fused dequantization kernel。实现方式：(1) 使用 PyTorch 的 `torch.utils.cpp_extension.load_inline` 或 setuptools 编译 CUDA kernel；(2) 在模型 forward pass 中用 autograd Function 包装自定义 kernel；(3) 支持 FP16 和 BF16 两种 activation 精度。关键工程实践：针对不同 GPU 架构使用不同的 warp tile 大小——RTX 4090 (SM89) 使用 128-thread per output row，Jetson Orin (SM87) 使用 64-thread。TinyChat 的 on-the-fly dequantization 在 4090 上实现 ~194 tokens/s (Llama-2-7B)，相比 HuggingFace FP16 的 52 tokens/s 加速 3.7×。代码：https://github.com/mit-han-lab/llm-awq/tree/main/tinychat。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

---
