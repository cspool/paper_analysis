## Warp-per-Row GPU Dictionary Decoding Kernel (Sub1MatVec / Warp级按行字典解码GPU内核)

术语是什么？
Warp-per-Row Dictionary Decoding Kernel 是 QMoE 设计的自定义 CUDA kernel（命名 Sub1MatVec），用于执行压缩权重矩阵与输入向量的融合解压缩-矩阵乘运算。核心并行策略：每个 GPU warp (32 threads) 处理权重矩阵的独立一行，使用 28/32 threads 同时从字典中执行解码，4 threads 不参与（处理数据格式非均匀性）。该设计的关键约束和决策：(a) UINT16 codewords 映射到 2×UINT32 数据（64-bit），每半 UINT32 由 14 threads 独立解码；(b) 字典 512KB 驻留 GPU L2 cache，按概率降序排列实现 L1 prefetch；(c) ternary dequant 通过 shared memory lookup table 实现，复制 32× 避免 bank conflict。

从kernel调度角度拆解术语：
**Sub1MatVec Kernel 伪代码（简化）**：
```
template<int num_warps, int w_width>
__global__ void Sub1MatVec(
    int* dec,           // 字典 [2^16 * 2] UINT32
    ushort* w_comp,     // 压缩权重 [total_codewords] UINT16
    int* row_off,       // 行偏移 [num_rows+1]
    __nv_bfloat162* ter_minmax,  // 每行 {w_min, w_max}
    __nv_bfloat16* x,   // 输入向量
    __nv_bfloat16* y)   // 输出向量
{
    // === 1. Shared Memory 初始化（全 threadblock 协作）===
    __shared__ float x_shared[w_width];
    for (int i = thread; i < w_width; i += 32*num_warps)
        x_shared[i] = bfloat162float(x[i]);  // 加载输入向量
    
    // === 2. 构建 Ternary Dequant 表（每 warp 独立）===
    __shared__ float deq[3][32 * num_warps];  // 复制 32× 避免 bank conflict
    deq[0][thread] = 0;
    deq[1][thread] = __bfloat162float(ter_minmax[row].x);  // w_min
    deq[2][thread] = __bfloat162float(ter_minmax[row].y);  // w_max
    __syncthreads();
    
    // === 3. 每行独立解码（warp-per-row）===
    __shared__ ushort w_comp_block[32][num_warps];
    int idx = 0;  // 当前输入向量偏移
    float res = 0;  // per-thread 累加器
    
    for (int i = 0; i < row_off[row+1] - row_off[row]; i += 32) {
        // 3a. Coalesced load: 32 UINT16 codewords → shared memory
        w_comp_block[warp][lane] = w_comp[i + lane];
        
        // 3b. 28 threads (lane 0-27) 解码
        if (lane < 28) {
            for (int j = 0; j < 32; j++) {
                int enc = w_comp_block[warp][j];  // UINT16 codeword
                // 线程 0-13 取 UINT32[0]，线程 14-27 取 UINT32[1]
                int wx14 = dec[2 * enc + (lane / 14)];
                // 提取 2-bit ternary 值（shift + mask，无慢速 modulo）
                int ter = (wx14 >> (4 + 2 * (lane % 14))) & 0x3;
                // Dequant via shared memory lookup（无 bank conflict）
                float w = deq[ter][thread];
                // FMA: 连续 shared memory 读（无 bank conflict）
                res += w * x_shared[idx + lane];
            }
            // 偏移推进 = 解码的权重总数（pair_count × 2）
            idx += 2 * (wx14 & 0xf);  // pair_count 存于低 4 bits
        }
    }
    
    // === 4. Warp Reduction ===
    res = warp_reduce_sum(res);  // warp shuffle sum
    if (lane == 0) y[row] = float2bfloat16(res);
}
```
关键设计：每 threadblock 占 1 SM（避免 wave quantization）；超过 32 行时 warp 串行处理多行但每行仍独立；decoding bit ops (~1 cycle) vs global memory read (~200 cycles) → 解码被访存完全隐藏；coalesced memory access (line 3a) + contiguous shared memory reads (line 3b FMA) = 接近内存带宽利用率上限。

术语一般如何实现？如何使用？
- 实现：QMoE 源码 https://github.com/ISTDASLab/qmoe（CUDA kernel 完整版含所有边界条件处理）
- 性能：所有 MoE 矩阵形状下比 cuBLAS bfloat16 GEMV 更快（最高 35% speedup），因压缩后 global memory 读取量仅 ~1/20
- 适用：压缩 MoE 模型的 decode 阶段（memory-bound GEMV）。prefill 阶段（batch >1）建议先完全解压再使用 cuBLAS GEMM
- 限制：当前实现 naively 对同一 expert 的多个 token 分别执行独立 matvec（vs baseline 的 batched matmul 更高效）；可扩展为 kernel 内 token batching

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
