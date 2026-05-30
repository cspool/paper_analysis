## Cooperative Softmax with Cross-Warp Shared Memory Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cooperative Softmax 是 BitDecoding 提出的多 warp 协作 softmax 算法，解决多 warp 沿 N 维并行时 register-level softmax 不可行的问题。原始 FlashAttention 中单 warp 持有完整 attention row P 在 register，可直接做 row-wise softmax。当 W_n > 1 时，每个 warp 仅持有 P 的部分 tile，需跨 warp reduction 计算 row-wise max 和 sum。Cooperative Softmax 利用 shared memory 做中间桥梁：sTMP buffer 做跨 warp max reduction，sAcc buffer 暂存 P 并通过 ldmatrix 重载确保后续 MMA 的 layout 对齐。

从kernel调度角度拆解术语。

```
// Algorithm: Multi-warp Cooperative Softmax
// sTMP ∈ R^{W_n}, sAcc ∈ R^{T_m × T_n} in shared memory

for each K/V tile j in 0..ceil(L/T_n):
    // Step 1: MMA compute S_i = Q_i K_j^T
    S_i = mma(Q_i_reg, K_j_dequant_reg)    // [T_m, T_n], in registers

    // Step 2: Row-wise max (cross-warp reduction)
    local_max = row_max(S_i)                // intra-warp, in register
    sTMP[warp_id] = local_max               // store to shared memory
    __syncthreads()
    global_max = max(sTMP[0:W_n])           // inter-warp, via shared mem
    __syncthreads()

    // Step 3: Online softmax update
    m_new = max(m_old, global_max)
    P_i = exp(S_i - m_new)                  // [T_m, T_n], in registers
    sAcc[tile_of_warp] = P_i                // store to shared memory
    __syncthreads()

    // Step 4: Reload P via ldmatrix for MMA alignment
    P_aligned = ldmatrix(sAcc)              // ensures interleaved TC layout
    O_new = mma(P_aligned, V_j_dequant_reg) + exp(m_old - m_new) @ O_old
    m_old, O_old = m_new, O_new

// Hopper optimization: sAcc directly consumed by wgmma_SS (no s2r step)
```

术语一般如何实现？如何使用？

实现在 BitDecoding Packing Kernel 中（~200 行 CUDA PTX）。W_n 典型值 4 或 8。Trade-off：增加 W_n 提升 parallelism 但增加 O(log W_n) shared memory access 的 cross-warp reduction overhead。Paper 表 III 表明 W_n=4 在 A100 上接近最优：overhead 仅 0.5%（3.746ms→0.613ms），TC utilization 从 10.91% 提升到 19.66%。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache
- Hardware-Efficient_Attention_for_Fast_Decoding

**补充（来自 Hardware-Efficient Attention for Fast Decoding）**：GLA kernel 同样采用了多 warp 协作 softmax，在 GLA GEMV 解码场景中（W_m=1，W_n>1），sTMP buffer 做跨 warp row-max reduction，sAcc buffer 暂存 attention scores 并通过 ldmatrix 重载保证 Tensor Core MMA 的 interleaved layout 对齐。与 BitDecoding 的带 dequantization 变体相比，GLA kernel 的 softmax 路径更简单（无低比特解量化），但由于 GLA 使用 latent attention（K/V 从 latent 直接参与 attention 而非常规 K/V），每 head 的 attention 维度为 2d_h（而非 d_h），算术强度更高，多 warp 协作的收益更显著。

---
