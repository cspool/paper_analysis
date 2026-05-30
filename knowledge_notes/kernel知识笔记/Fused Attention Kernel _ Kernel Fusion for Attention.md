## Fused Attention Kernel / Kernel Fusion for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused Attention Kernel是指将attention计算的所有子操作（QK^T矩阵乘 → softmax含masking和dropout → PV矩阵乘）融合为单个GPU kernel的技术。标准attention实现将这些操作拆分为多个独立的CUDA kernel（例如cuBLAS GEMM for QK^T → element-wise kernel for softmax → cuBLAS GEMM for PV），每个kernel的中间结果必须写入HBM再由下一个kernel读取，产生大量HBM traffic。Fused attention kernel将所有操作合并到一个kernel中，中间tensor仅在on-chip SRAM中驻留和传递，消除了kernel间的HBM round-trip。FlashAttention首次实现了exact attention的完全融合kernel：tiling + online softmax使得attention可以在block粒度上streaming计算，每block pair的中间S_ij和P_ij在SRAM中产生、消费并丢弃。Kernel fusion直接提升了arithmetic intensity——原本每个element需要多次HBM访问（读S、写S、读S、写P、读P），融合后仅需一次加载和一次写出。FlashAttention的fused kernel实测将HBM R/W从35.3GB降至4.4GB（8× reduction），即使总FLOPs从66.6增至75.2 GFLOPs（因backward recomputation），wall-clock time仍从35.1ms降至11.7ms（3× faster）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention fused kernel的CUDA kernel内部调度（以A100, N=1024, d=64, B_c=384, B_r=64为例）：
```
// 单次kernel launch替代3+个独立kernel
__global__ void flash_attn_fwd_kernel(Q, K, V, O, m, l, N, d, B_r, B_c) {
    // Grid: (T_r, batch*heads)  每个CUDA block处理一个Q block
    // Shared memory（per CUDA block）:
    __shared__ half Q_s[B_r][d];      // 64x64x2B = 8KB
    __shared__ half K_s[B_c][d];      // 384x64x2B = 49KB
    __shared__ half V_s[B_c][d];      // 384x64x2B = 49KB
    __shared__ half S_s[B_r][B_c];    // 64x384x2B = 49KB（可与P_s复用）
    // Total SRAM: 8+49+49+49 = 155KB, fits in 192KB
    
    int i = blockIdx.x;  // Q block index
    load Q_s from HBM (Q_i);          // HBM → SRAM (8KB)
    
    half O_acc[B_r][d] = {0};         // 累加器在寄存器中
    float m_acc[B_r] = {-inf};        // running max per query row
    float l_acc[B_r] = {0};           // running exp-sum per query row
    
    for (int j = 0; j < T_c; j++) {   // 外循环: KV blocks (串行, T_c=3)
        load K_s, V_s from HBM;       // HBM → SRAM (49KB+49KB)
        __syncthreads();
        
        // BMM1: S = Q_s @ K_s^T (Tensor core WMMA/MMA)
        compute S_s = Q_s @ K_s.T;    // S_s in SRAM, 64x384
        
        // Online softmax (CUDA core: MUFU.EX2, FADD, FMUL)
        for (int r = 0; r < B_r; r++) {
            float m_ij = rowmax(S_s[r]);  // block local max
            float m_new = fmaxf(m_acc[r], m_ij);
            float rescale = exp2f((m_acc[r]-m_new) * LOG2E);
            // Rescale old accumulations
            l_acc[r] = l_acc[r] * rescale;
            for (int d_=0; d_<d; d_++) O_acc[r][d_] *= rescale;
            // Compute new block contributions
            float l_ij = 0;
            for (int c=0; c<B_c; c++) {
                S_s[r][c] = exp2f((S_s[r][c]-m_new)*LOG2E);  // P_ij (in-place)
                l_ij += S_s[r][c];
            }
            l_acc[r] += l_ij;
            // BMM2: O_acc += P_ij @ V_j (Tensor core MMA)
            accumulate O_acc[r] += S_s[r] @ V_s;  // SRAM GEMM
            m_acc[r] = m_new;
        }
        __syncthreads();
    }
    // Final normalization + write to HBM
    for (int r=0; r<B_r; r++) O_acc[r] /= l_acc[r];
    store O_acc → HBM (O_i: 64x64x2B=8KB);
    store m_acc, l_acc → HBM (64*4B*2 = 512B per Q block);
}
```
关键调度设计：外循环（KV blocks, T_c=3）串行执行因softmax跨KV block耦合；内循环（Q blocks, T_r=16）跨16个CUDA blocks并行，每个block独立处理一个Q_i。BMM1/BMM2使用Tensor core，softmax使用CUDA core，在warp级交错以实现计算与数据加载overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashAttention CUDA kernel已开源：https://github.com/HazyResearch/flash-attention（BSD许可）。PyTorch接口：`from flash_attn import flash_attn_func; output = flash_attn_func(q, k, v, causal=False, dropout_p=0.0)`。后续演进：FlashAttention-2（改善parallelism，减少non-matmul FLOPs，优化work partitioning across thread blocks），FlashAttention-3（Hopper架构利用TMA异步拷贝 + warp-group specialization producer-consumer pattern），FlashAttention-4（Blackwell架构pipeline优化）。kernel fusion理念已从attention扩展到FFN（SRAMFFN）、通信重叠（FlashOverlap）等场景。实现核心挑战：(1) 正确性——online softmax跨block的数值精度需与标准实现一致；(2) block size调优——需balance SRAM capacity、SM occupancy和HBM pass数。FlashInfer、xFormers等库也提供类似的fused attention kernel。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
