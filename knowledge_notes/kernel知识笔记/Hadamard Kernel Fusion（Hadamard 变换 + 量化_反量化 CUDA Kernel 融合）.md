## Hadamard Kernel Fusion（Hadamard 变换 + 量化/反量化 CUDA Kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hadamard Kernel Fusion 是 SDP4Bit 中的 CUDA kernel 级优化，将 Walsh-Hadamard Transform (WHT) 与对称线性 (de)quantization 操作融合为单个 GPU kernel，消除中间 global memory traffic。核心设计：(1) 每个 CUDA thread block 处理一个 quantization group（大小需被 H 矩阵大小整除）；(2) 从 global memory 加载数据到 shared memory（1 次读）；(3) 在 shared memory 中执行 32×32 Hadamard transform（仅加减运算，memory-bound 在此大小）；(4) 在 shared memory 中直接计算 per-group scale 并执行量化；(5) 将 packed INT4 输出写回 global memory（1 次写）。融合效果：Hadamard transform 的额外开销降至 < 0.3%（Table 5 显示 w/ and w/o Hadamard 的 (de)quantization throughput 差异仅 301.8 vs 305.6 GB/s at 8MB）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused Hadamard + Quantize CUDA kernel 伪代码：
```cuda
// 输入: grad[N] in global memory (FP32)
// 输出: packed_int4[N/2] + scales[N/G] in global memory
// H=32×32, G=512 (group_size), 每个 block 处理一个 group

__global__ void fused_hadamard_quantize_int4(
    const float* grad, uint8_t* packed_out, float* scales,
    int N, int group_size, int H_size)
{
    int gid = blockIdx.x;  // group id
    int tid = threadIdx.x;
    int base = gid * group_size;

    // 1. 加载到 shared memory (1 次 global read)
    __shared__ float smem[512];  // group_size=512
    for (int i = tid; i < group_size; i += blockDim.x)
        smem[i] = grad[base + i];
    __syncthreads();

    // 2. Hadamard transform in shared memory (32x32 block-wise)
    //    H = H_32 ⊗ I_{(G/32)}  (Kronecker product, 无需额外数据移动)
    #pragma unroll
    for (int b = 0; b < group_size; b += H_size) {
        // Fast Walsh-Hadamard Transform: O(H_size * log H_size) adds/subs
        for (int step = 1; step < H_size; step <<= 1) {
            for (int i = tid; i < H_size; i += blockDim.x) {
                int idx = b + i;
                int pair = idx ^ step;  // butterfly pattern
                if (idx < pair) {
                    float a = smem[idx];
                    float b = smem[pair];
                    smem[idx] = a + b;
                    smem[pair] = a - b;
                }
            }
            __syncthreads();
        }
    }

    // 3. 在 shared memory 中量化 (无额外 global memory traffic)
    // 先由 warp reduce 求 max(|group|)
    float local_max = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(smem[i]));
    // warp-level reduction for scale
    float s = warp_reduce_max(local_max);
    if (tid == 0) scales[gid] = s;

    // 4. 量化并 packed write (1 次 global write)
    float inv_s = 7.0f / s;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int8_t q = roundf(clamp(smem[i] * inv_s, -7.0f, 7.0f));
        // pack 2 INT4 into 1 uint8
        int idx = base / 2 + i / 2;
        if (i % 2 == 0) packed_out[idx] = (q & 0xF);
        else           packed_out[idx] |= (q << 4);
    }
}
```

融合的关键约束：`group_size % H_size == 0`，确保每个 quantization group 可被整数的 Hadamard blocks 覆盖。SDP4Bit 选择 H=32 因为：(1) 32×32 transform 在 GPU 上是 memory-bound（compute 占比极低），融合后开销可忽略；(2) 32×32 足以有效平滑梯度 outlier（Fig. 6 证实）；(3) 与典型 group_size（128, 512）兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SDP4Bit 的 fused kernel 在 `megatron/core/tensor_parallel/` 下实现（CUDA C++），通过 PyTorch C++ extension 注册为 Python 可调用函数。CUDA kernel 使用 `__shared__` memory 存储中间数据，利用 warp shuffle instructions 进行 reduction。Hadamard transform 使用 in-place 的 butterfly 模式（FWHT），无需额外 memory。融合 kernel 支持 INT4 和 INT8 两种输出精度（分别对应 gradient inter-node 和 intra-node 量化）。使用时在 Megatron-LM 训练循环中替换独立的 Hadamard + quantize + dequantize kernel 调用。论文 Table 4 确认融合效果：grad comm time 从 64.6ms（unfused）降至 45.8ms（fused, -29%），E2E TFLOPs 从 55.2 升至 58.5（+6%）。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
