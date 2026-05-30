## Ping-Pong Scheduling (Warpgroup MMA/IO Overlap)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ping-Pong Scheduling（乒乓调度）是 NVIDIA Hopper GPU 上 warp-specialized GEMM kernel 的线程调度策略：将 2 个 consumer warpgroups 交替分配 MMA 和 epilogue/IO 工作。当 consumer WG 0 执行 WGMMA 进行当前 tile 的矩阵乘法时，consumer WG 1 同时执行上一 tile 的 epilogue（activation、store to HBM）。每 tile 完成后角色互换。此概念最早见于 CUTLASS Hopper warp-specialized kernels，FlashAttention-3 (Shah et al. 2024) 将其用于 attention。SonicMoE 首次将 Ping-Pong 应用于 MoE kernel，特别是针对细粒度 MoE 的 heavy epilogue 场景（如 backward dH kernel 需在 epilogue 中同时执行 dSwiGLU、dS reduction 和 TMA store A'）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SonicMoE forward down-proj Y kernel Ping-Pong 调度（H100）：

```
// 2 consumer warpgroups 交替执行 MMA 和 epilogue
for tile_i in work_tiles:
    if tile_i % 2 == 0:  // Ping phase
        consumer[0].wgmma(tile_i)    // WGMMA compute
        consumer[1].epilogue(tile_i-1) // TMA store previous tile
    else:                 // Pong phase
        consumer[1].wgmma(tile_i)
        consumer[0].epilogue(tile_i-1)
    sync_warpgroups()  // role switch barrier
```

传统无 Ping-Pong：MMA → barrier → epilogue → barrier → next MMA。Ping-Pong 将 epilogue latency 完全隐藏在 MMA 之下。特别对 fine-grained MoE（dH kernel epilogue 需 load H + compute dSwiGLU + dS reduction + store dH/dS/A'），Ping-Pong 维持了高 Tensor Core utilization。

Blackwell 上的等效：利用 TMEM 2-stage（每 stage 256×128 columns of 32-bit cells）和 UMMA 单线程异步指令——MMA warp 写入一个 TMEM stage 时，epilogue warps 并发读取另一个 stage 的结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CuTe-DSL 实现：创建 2 个 consumer warpgroup pipeline stages，使用 `cute::conditional_return` 和 warpgroup barrier 同步。适用条件：epilogue IO 相对 MMA tile 不可忽略（n < 1024 的 fine-grained MoE），且 SMEM 充足支持双缓冲。SonicMoE 动态选择 Ping-Pong vs 普通 scheduling 取决于 intermediate size。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
