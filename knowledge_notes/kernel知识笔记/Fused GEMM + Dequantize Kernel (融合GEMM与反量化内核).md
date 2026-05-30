## Fused GEMM + Dequantize Kernel (融合GEMM与反量化内核)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused GEMM + Dequantize 是将量化权重的反量化（dequantization）操作融合进 GEMM 矩阵乘法 kernel 内部的技术。传统做法是先启动一个单独的 dequantize kernel 将 INT4/INT8 权重转为 FP16 写入 GPU 全局内存，然后再启动 GEMM kernel 读取 FP16 权重进行计算。这引入了额外的内存读写：write dequantized weights → read for GEMM。Fused 版本在 GEMM 的 weight tile load 阶段直接读取量化权重并在线程寄存器中完成 dequantize，避免中间的全局内存读写。对于 memory-bound 的 MoE GEMM 操作（加载多个 expert weights 受限于内存带宽）尤其重要。

论文 "Who Says Elephants Can't Run" 的核心洞察：profiling 发现 native CUDA IntToFloat (I2F) 是 fused kernel 的性能瓶颈，因此用 FP16 bit-trick 序列取代 I2F。

从kernel调度角度拆解术语：

INT8 Fused GEMM + Dequantize per-tile 计算过程（V100, SM70）：
```
for each K tile:                               // K dimension iteration
    // Load FP16 activations
    A_tile = load_to_smem(A, m_tile, k_tile)
    
    // Load INT8 weights + Fused Dequantize in registers
    w_packed = *reinterpret_cast<uint32_t*>(&W_plus[idx])
    // Extract bytes: e0,e1,e2,e3
    
    // Optimized INT8→FP16 (2 elements at a time):
    // 利用 FP16 性质: 0x6400 | X  = FP16(X+1024)
    R1 = construct_fp16_pair(e0+1024, e1+1024)  // 0x6400 | e1 << 16 | 0x6400 | e0
    R1_fp16 = R1_fp16 - [1152.0, 1152.0]         // = [e0-128, e1-128] as FP16
    // 1152 = 1024 (bias) + 128 (unsigned offset)
    W_deq = R1_fp16 * S[tile_n]                  // per-channel scale
    
    // FP16 Tensor Core GEMM accumulate
    acc += A_tile @ W_deq
```

INT4 优化变体：权重 layout 重排 `[e0..e7] → [e0,e2,e4,e6,e1,e3,e5,e7]`，减少 bit extraction 操作。减去常量从 1152 变为 1032 (=1024+8, offset=8)。

性能（V100, 32 active experts, 40 tokens）：INT8 optimized 1.59× FP16 baseline, INT4 optimized 1.85× FP16 baseline。对比 native I2F（INT8: 1.46×），optimized I2F 序列约 9% 额外加速。

术语一般如何实现？如何使用？

基于 CUTLASS 自定义 kernel，修改 GEMM 的 weight load "prologue" 阶段。需要在 CUTLASS collective builder / CuTe DSL 中定义自定义 weight tile load。关键考虑：(1) dequantize 计算不能成为新瓶颈（FP16 bit-trick 解决）；(2) INT4 需要特殊 weight layout 对齐 32-bit 加载；(3) 与 Grouped GEMM 配合时各 expert token 数不同（varlen-M）。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
