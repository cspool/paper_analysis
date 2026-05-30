## Fusion-based Computation-Communication Overlapping

术语是什么？

Fusion-based overlapping（基于融合的重叠）将通信原语实现直接融合到 GEMM kernel 内部，在单个 GPU kernel 中通过指令调度实现 tile 级的计算-通信交织。代表性工作：FLUX、Comet (MLSys'25)、TileLink、cuBLASMp (NVIDIA)、AMD fused embedding+GEMV+GEMM。

从系统架构角度拆解术语：

Fusion-based 方法的架构权衡：

```
单 kernel 内 tile 级交织:
for each tile t:
    compute_tile(t);           // GEMM main loop for this tile
    allreduce_tile_in_kernel(t);  // 手动实现的 AllReduce: local reduce + P2P send/recv

优势:  tile-wise overlapping ✓ → 最大重叠机会
       单 kernel launch → 低 launch overhead
  
限制:
  - 通信需手动实现，无法利用 NCCL 优化（RING algorithm, SHARP 等）
  - 每种通信原语（AR, RS, A2A）需独立融合实现
  - 融合时协调 pipeline 可能需修改 tiling 策略 → 性能退化风险
  - 不同 GPU 架构需不同优化 → 跨平台适配成本高
```

术语一般如何实现？如何使用？

FLUX 针对 TP 优化，融合通信到 GEMM kernel tile 级边界。Comet 用 thread block specialization——computation 和 communication 分配在不同 SM 并行执行。TileLink 用 compiler-based 方法自动生成 overlapping kernel。适合 GPU 架构固定且值得投入 manual optimization 的生产环境。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
