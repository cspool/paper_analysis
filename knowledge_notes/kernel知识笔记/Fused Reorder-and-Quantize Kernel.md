## Fused Reorder-and-Quantize Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Reorder-and-Quantize Kernel（融合重排量化kernel）是 MicroMix 的 GPU kernel 优化技术，将激活张量的通道重排（channel reordering）和 block-wise MX 量化融合为单个 CUDA kernel。混合精度 MX 量化中，相邻通道可能分配不同精度（MXFP4/6/8），需将同精度通道重排到连续块以实现规则内存访问。融合消除中间结果的 global memory 往返，将重排+量化开销控制在总 kernel 时间 20% 以内。

从kernel调度角度拆解术语，给出伪代码或具体计算过程。
```
__global__ void fused_reorder_quantize(
    half* X, int* sigma, int p4_K, int p6_K,
    MXFP4* out_G4, MXFP6* out_G6, MXFP8* out_G8
) {
    __shared__ half tile[BLOCK_M][BLOCK_K];
    load_tile(X, tile);  // coalesced read from global memory
    
    // 在 shared memory 中按 sigma 重排列
    __shared__ half reordered[BLOCK_M][BLOCK_K];
    for (int j = 0; j < BLOCK_K; j++)
        reordered[threadIdx.y][j] = tile[threadIdx.y][sigma[block_start + j]];
    
    // 分组量化：G4 (MXFP4) / G6 (MXFP6) / G8 (MXFP8)
    quantize_mxfp4_block(reordered[:, 0:p4_K], out_G4);
    quantize_mxfp6_block(reordered[:, p4_K:p4_K+p6_K], out_G6);
    quantize_mxfp8_block(reordered[:, p4_K+p6_K:K], out_G8);
}
// 每个 quantize_mxfpX_block 内部对 32 元素 block:
//   scale = 2^{floor(log2(max_abs)) - b}; Q(x) = round(clip(x/scale, -q_max, q_max))
```
注解：输入 X 为 FP16 [M, K]；sigma 为离线预计算的通道排列索引；p4_K/p6_K 为整数分界；输出三组 MX 张量直接供后续 MXFP GEMM 使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MicroMix 基于 CUDA 实现，shared memory 作重排缓冲区。要点：(1) coalesced 加载 FP16 激活；(2) shared memory 列索引重映射避免 global memory irregular access；(3) 量化紧接重排，直接输出 MX 格式；(4) 输出直连后续三路 MXFP GEMM kernel。RTX 5090 上 fused reorder-and-quantize 仅占总 kernel 时间 7.9%-17.0%（seqlen 128→4096），GEMM 占 83.0%-92.1%。适用于任何需通道重排+量化连续执行的混合精度量化场景。

涉及论文标题：
- MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

---
