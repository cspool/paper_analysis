## On-the-fly GEMV Decoding CUDA Kernel（即时 GEMV 解码 CUDA 内核）

术语是什么？
On-the-fly GEMV Decoding CUDA Kernel 是 PuzzleMoE 设计的一个自定义 CUDA kernel，在 GPU 上执行矩阵-向量乘法（GEMV）的同时即时从 bit-packed Bfloat16 格式中解码出每个 weight 的实际值。核心思想：解码逻辑（bit shift + mask + exponent 恢复）是计算量极小的 in-place 操作，可以 piggyback 在 kernel 的 data-loading path 上，利用 warp-level scheduling 和 coalesced memory access 使解码延迟被 global memory 读取延迟（~200 cycles）完全隐藏。该 kernel 消除了在 GPU global memory 中创建独立 decoded weight matrix 的需求，避免额外的 memory allocation 和访存开销。

从kernel调度角度拆解术语：
**CUDA Thread Block 执行流程：**
```
__global__ void puzzle_gemv_kernel(
    half* X,          // input activation [B, d]
    uint16_t* W_packed, // packed weights [d, h]
    half* Y,          // output [B, h]
    int expert_pos)   // 0 or 1
{
    int row = blockIdx.x;  // which output dimension
    int tid = threadIdx.x;
    float acc = 0.0f;

    // Coalesced load of input
    for (int k = tid; k < d; k += blockDim.x) {
        half x_val = X[row * d + k];        // load from global memory (~200 cycles)
        uint16_t w_packed = W_packed[k * h + row]; // coalesced load

        // On-the-fly decode (<< 10 cycles, hidden by load latency)
        int mask_bit = (w_packed >> (13 - expert_pos)) & 1;
        if (mask_bit == 0) continue; // weight pruned for this expert

        int sign_bit = (w_packed >> (15 - expert_pos)) & 1;
        int exp = (w_packed & 0x0F80) + (112 << 7);
        uint16_t w_decoded = (sign_bit << 15) | exp | (w_packed & 0x007F);

        half w_val = __uint2half_rn(w_decoded);
        acc += __half2float(x_val) * __half2float(w_val);
    }

    // Warp-level reduction
    acc = warpReduceSum(acc);
    if (tid == 0) Y[row] = __float2half(acc);
}
```

**关键设计决策：**
1. **Decoding on data-load path**：w_packed 从 global memory 加载到寄存器后立即解码——decode 指令（3-4 条 bit ops）与 memory load 的延迟比约为 1:200，因此解码开销被访存完全隐藏。
2. **No materialized decoded matrix**：与传统方法（先解码整个矩阵到 memory 再执行 GEMM）相比，避免了 O(d×h) 的额外 memory 分配和访存。
3. **Zero-value skipping**：mask_bit=0 时直接跳过 FMA——虽然仍在 warp 内（warp divergence），但因 data-load 已发生（mask 在 register 中而非 global memory），整体收益仍为正。
4. **expert_pos parameter**：同一 merged weight 通过 expert_pos=0 和 expert_pos=1 分两次调用 kernel 即可获得两个 expert 的输出——mask/sign 的 bit position 由 expert_pos 动态决定。

术语一般如何实现？如何使用？
- 集成到 PyTorch 推理框架中作为 torch.autograd.Function 的 forward 实现，替换标准 torch.nn.functional.linear。
- Gate network 和 attention 部分使用标准 PyTorch 算子——仅 expert FFN 的 GEMV 使用自定义 kernel。
- 适用于 decode phase（单 token 生成，GEMV 而非 GEMM）——batch_size=1 的自回归解码。
- 也可为 prefill phase（多 token batch, GEMM）扩展为 tiled GEMM kernel，但论文主要 benchmark decode phase 加速。

涉及论文标题：
- PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference
