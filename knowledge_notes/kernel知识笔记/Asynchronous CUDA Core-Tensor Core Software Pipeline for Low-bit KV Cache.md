## Asynchronous CUDA Core-Tensor Core Software Pipeline for Low-bit KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

这是 BitDecoding Packing Kernel 的 register-level 异步流水线设计，使 CUDA Cores 的 dequantization 与 Tensor Cores 的 mma 重叠执行。Global→Shared Memory: cp.async 异步加载 Q tile、packed K/V tile、量化参数 tile，不同 caching strategy（cg for no-reuse、ca for byte-aligned fine-grained access）。Shared→Register: ldmatrix 加载 packed data 到 TC register layout + lop3 75316420 pattern remapping 做高效 dequantization。核心异步机制：第 i 个 tile 在 TC 上做 mma 的同时，第 i+1 个 tile 的 ldmatrix + dequant 在 CUDA Cores 上执行。Hopper 上利用 warp-specialized pipeline（部分 warp 负责 STSM + wgmma，部分负责 ldmatrix + dequant）。

从kernel调度角度拆解术语。

```
// 两级异步流水线 (Inter-tile + Intra-tile)
// Shared Memory Double Buffering: SMEM[0], SMEM[1]

// === Prologue: Prefetch tile 0 ===
cp.async.cg: Q_tile, K_pack[0], V_pack[0] → SMEM[0]
cp.async.ca: K_p[0], V_p[0] → SMEM[0]
cp.async.wait_group(0)  // 等待 tile 0 完成
__syncthreads()

// === Steady State: Pipeline iteration ===
for i in 0..C_n-1:  // C_n = num KV tiles
    // Stage 1: Prefetch tile i+1 (async, non-blocking)
    cp.async.cg: Q_tile, K_pack[i+1], V_pack[i+1] → SMEM[(i+1)%2]
    cp.async.ca: K_p[i+1], V_p[i+1] → SMEM[(i+1)%2]

    // Stage 2: Load + Dequant tile i (CUDA Cores)
    K_reg = ldmatrix(K_pack_smem[i%2])
    K_param = ldmatrix(K_p_smem[i%2])
    K_fp16 = lop3_75316420_remap(K_reg)  // INT4/INT2→FP16
    K_fp16 = K_fp16 * K_param.scale + K_param.zp

    // Stage 3: MMA tile i (Tensor Cores)
    // 与下一个 tile 的 Stage 1 cp.async 和 Stage 2 dequant 在 SM 内重叠
    S = mma(Q_reg, K_fp16)
    // ... cooperative softmax ...
    O = mma(P_aligned, V_fp16)

    cp.async.wait_group(0)  // 等待 tile i+1 load 完成
    __syncthreads()

// === Epilogue: 最后一个 tile ===
```

术语一般如何实现？如何使用？

实现在 BitDecoding Packing Kernel（~500 行 CUDA PTX）。Memory transaction 优化：Q/K_pack/V_pack 用 `cp.async.cg`（cache global only，无 L1 pollution）；K_p/V_p 用 `cp.async.ca`（支持 byte-aligned 小粒度）。Hopper 版用 TMA + warp specialization。Dequantization overhead 从 baseline ~50% 降至 <15%(4-bit) 和 <35%(2-bit)。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
