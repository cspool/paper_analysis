## Group-wise Mixed-Precision Dequantization Kernel (AutoGPTQ Extension)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
该 kernel 是 SliM-LLM 基于 AutoGPTQ 扩展的 CUDA 推理 kernel，支持 group 级别混合精度（1/2/3-bit）权重的 GPU dequantization 和矩阵乘法。核心设计：(1) Weight Packing——量化后的权重按 group 分别 pack 为整数（每个 group 内 128 个元素以相同精度 pack，利用 group_size=128 是任意 2 的幂的倍数这一特性，无需 padding）；(2) 额外 bit-widths array——每个 group 用 2-bit 编码精度（00=未使用, 01=1-bit, 10=2-bit, 11=3-bit），聚合成 32-bit 整数数组；(3) 逐 group 解包计算——GPU 上每个 thread 处理一列连续 pack 数据的 dequantization，与 block 内共享的 input activation 做向量点积，结果累加到输出矩阵对应位置。因为精度在 group 边界对齐（而非 element-wise），warp 内 32 threads 的 code path 和数据访问逻辑保持一致。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 每个Logical Block处理一段连续channel区域
__global__ void mixed_precision_dequant_matmul(
    uint32_t* w_packed,    // 混合精度packed权重
    uint32_t* z_packed,    // packed zeros
    half* scales,          // FP16 scales
    uint32_t* bit_widths,  // 每group的2-bit精度编码
    half* input,           // input activation [t, m]
    half* output           // output [t, n]
) {
    // 加载共享的input activation片段
    __shared__ half input_shared[BLOCK_SIZE];
    // 共享activation加载到shared memory
    load_input_to_shared(input, input_shared);

    // 逐group处理
    int group_offset = 0;  // 追踪packed数组中的累积偏移
    for (int g = block_start_group; g < block_end_group; g++) {
        // 读取当前group的bit-width
        int bw_idx = g / 16;  // 每32-bit存16个group的精度(每组2-bit)
        int bw_shift = (g % 16) * 2;
        int bw = (bit_widths[bw_idx] >> bw_shift) & 0x3;  // 提取2-bit

        // 根据bw计算packed元素数
        int elems_per_int = 32 / bw;  // 1-bit:32, 2-bit:16, 3-bit:10(余2)
        int idx_in_int = thread_id % elems_per_int;

        // 解包该thread对应的权重值
        uint32_t packed_val = w_packed[group_offset + thread_id / elems_per_int];
        int w_int = (packed_val >> (idx_in_int * bw)) & ((1 << bw) - 1);

        // 反量化
        half w_deq = (half)(w_int - zero) * scale;

        // 向量点积累加
        for (int t = 0; t < num_tokens; t++) {
            output[t * n + out_col] += w_deq * input_shared[t];
        }

        group_offset += (128 + elems_per_int - 1) / elems_per_int;  // 累积偏移
    }
}
```
与统一精度 kernel 的关键差异：bit-width 需要逐 group 读取和解析（额外 2-bit/group 的 array lookup）；累积偏移计算确保跨 group 的正确 start index；1-bit group 需要 sign+α 反量化（而非标准 INT dequantization），增加了分支。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 kernel 基于 AutoGPTQ 的 CUDA warp 机制实现。每个 warp 内 32 threads 处理一个 group（128 元素，组大小正好是 warp size 的倍数），使 threads 在相同 code path 上执行。实现的主要挑战：(a) 1-bit 权重使用 sign + α 格式反量化（ŵ = α · sign(w_fp)），不走标准 INT 量化路径，需要额外分支处理；(b) 3-bit 权重 128 个元素 pack 为 10 个 32-bit 整数的子集（10×32=320 bits > 128×3=384 bits → 需要 12 个 32-bit int），存在跨整数边界对齐问题。部署验证：LLaMA-7B 2-bit SliM-LLM inference 61.2 token/s vs GPTQ 2-bit 83.9 token/s（~27% slowdown），换取 90% perplexity 提升（PPL 14.58 vs 152.31）。开源代码：https://github.com/Aaronhuang-778/SliM-LLM。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
