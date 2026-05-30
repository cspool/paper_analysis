## CUTLASS

术语是什么？
CUTLASS（CUDA Templates for Linear Algebra Subroutines and Solvers）是NVIDIA开发的开源CUDA C++模板库，用于编写高性能GEMM（通用矩阵乘法）及GEMM-like计算。它通过多层模板抽象将GPU的线程层次结构（thread→warp→threadblock/CTA→grid）映射到矩阵计算的分块层次结构。CUTLASS将GEMM表示为一个tiled triple-loop nest：最外层CTA级tile循环（计算output tile，沿k轴迭代）、中间warp级tile循环（warp内的矩阵乘）、最内层instruction级MMA操作（Tensor Core或CUDA Core）。CUTLASS提供data-movement和multiply-accumulation类，允许用户在GPU线程层次的所有级别组合自定义GEMM-like计算。

从编译框架角度拆解术语：
CUTLASS的工作流程（以GEMM kernel为例）：

```
开发者编写 CUTLASS kernel:
  1. 定义 ElementA/ElementB/ElementC 数据类型 (e.g., half, float)
  2. 选择 TileShape (e.g., 128×128×32 for FP16→32 GEMM)
  3. 配置 WarpShape 和 InstructionShape (e.g., 64×64×32 warp, 16×8×16 MMA)
  4. 实例化:
     - ThreadblockSwizzle (grid tile到CTA的映射策略)
     - Epilogue (output tile的element-wise操作, 如bias/activation)
     - Mainloop (MAC-loop迭代的执行策略, 含software pipeline)

  → CUTLASS模板在编译期展开:
     CTA级别: 生成grid launch配置, 每CTA处理一个output tile
     Warp级别: 生成warp-specialized代码 (producer/consumer warp groups)
     线程级别: 生成per-thread register blocking和fully unrolled MAC循环
     Shared memory: 生成software-pipelined shared memory staging
     
  → CUDA编译 (nvcc) → GPU可执行kernel

运行时:
  调用 compiled kernel(grid_dims, block_dims, shared_mem_bytes, stream)
  → 每个CTA独立执行分配的output tile
```

Stream-K论文在CUTLASS中实现了新的grid-level decomposition（Stream-K），而不修改CTA内部的MacLoop()子程序。这体现了CUTLASS的模块化设计：grid-level work decomposition可以独立于tile-level实现进行替换。

术语一般如何实现？如何使用？
CUTLASS已开源（https://github.com/NVIDIA/cutlass），目前版本（3.x+）支持：
- 多种数据类型的GEMM：FP64, FP32, FP16, BF16, INT8, INT4, FP8 (Hopper+)
- Warp specialization：Producer warp groups (TMA/DMA) 和 Consumer warp groups (MMA)
- 多种调度策略：Data-parallel, Fixed-split, Stream-K (自v2.11)
- Epilogue fusion：bias, activation (ReLU/GELU/SiLU), 量化/反量化
- Thread Block Clusters (Hopper SM90+) 和 TMA (Tensor Memory Accelerator)
- Grouped GEMM 和 Convolution

CUTLASS是cuBLAS、cuDNN、PyTorch等库的底层GEMM kernel来源之一。用户可以直接使用CUTLASS编写自定义kernel，也可以依赖CUTLASS的Python绑定（通过PyTorch的torch.compile集成）。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
