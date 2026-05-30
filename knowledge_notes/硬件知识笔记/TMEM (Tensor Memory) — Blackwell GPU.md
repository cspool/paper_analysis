## TMEM (Tensor Memory) — Blackwell GPU

术语解释
Tensor Memory (TMEM) 是 NVIDIA Blackwell GPU 架构引入的专用 on-chip memory，每 SM 256KB，组织为 128 rows × 512 columns of 32-bit cells，用于存储 UMMA (Unified Matrix Multiply-Accumulate) 指令的 accumulator 结果，替代 Hopper 架构中用 register file 存储 accumulator 的方式。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TMEM 是 Blackwell SM 内新增的存储器层次，位于 register file (RF) 和 shared memory (SMEM) 之外。核心特性：(1) 512-column 结构天然支持 2-stage accumulator pipeline——每 stage 使用 256 columns：当 MMA warp 通过 UMMA 指令向 Stage A 写入 accumulation 结果时，epilogue warps 可从 Stage B 读取上一 tile 的结果执行 epilogue，无需等待 MMA 完成；(2) 每 column 可容纳 128 rows × 32-bit = 512B，总计 128×512×4 = 256KB；(3) UMMA 是单线程异步指令（vs Hopper WGMMA 的 warpgroup 同步），消除了 accumulation 对 register file 的消耗。SonicMoE 利用 TMEM 2-stage 结构实现比 Hopper Ping-Pong 更高效的 MMA/epilogue overlap，特别在 backward dH kernel 的 heavy epilogue 场景中受益显著。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Blackwell B300 SM 内 TMEM-based MMA 流水线（SonicMoE dH kernel）：

```
[Producer warp] → TMA load W_2, dO, H, S 到 SMEM

[MMA warp]
for tile in [0..num_tiles):
    if tile % 2 == 0:
        UMMA accumulate → TMEM Stage A (columns 0-255)
    else:
        UMMA accumulate → TMEM Stage B (columns 256-511)
    // UMMA 是异步的，MMA warp 立即处理下一 tile

[Epilogue warps] (并发运行)
for tile in [0..num_tiles):
    wait for TMEM stage ready (signal from MMA warp)
    if tile % 2 == 0:
        read from TMEM Stage A → compute dSwiGLU/dS/A' → TMA store
    else:
        read from TMEM Stage B → compute dSwiGLU/dS/A' → TMA store
```

对比 Hopper：WGMMA 结果在 RF 中（每线程 64+ registers），epilogue 需要先 move RF→SMEM 或直接从 RF 操作，且 WGMMA 是 warpgroup 级同步指令，无法实现 thread 级的 MMA/epilogue overlap。TMEM+UMMA 解耦了这两个操作，因此 Blackwell 上 SonicMoE 不需要显式的 warpgroup 角色切换（Ping-Pong）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 CUDA 中通过 `umma` PTX 指令编程，CUTLASS 3.x+ 在 SM100 kernel 中支持 TMEM。Blackwell GPU 独有（B100/B200/B300），Hopper 及以下架构无 TMEM。使用 TMEM 的关键约束：每 SM 256KB TMEM 由 SM 内所有 warps 共享，大 tile size 配置可能超出 TMEM capacity。

涉及论文标题：
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
