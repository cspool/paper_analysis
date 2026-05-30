## CUTLASS (CUDA Templates for Linear Algebra Subroutines)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUTLASS 是 NVIDIA 开源的 CUDA C++ 模板库（https://github.com/NVIDIA/cutlass），通过 C++ 模板将 GEMM 分解为可组合的抽象层次（tile → warp → thread），编译期生成针对特定数据类型、矩阵布局和 GPU 架构优化的 kernel。FlashMoE 使用 CUTLASS 作为 in-kernel BLAS，在持久 kernel 内直接调用 device-side GEMM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// CUTLASS 三级分块:
// 1. Thread Block Tile → shared memory (TILE_M×TILE_K + TILE_K×TILE_N)
// 2. Warp Tile → registers (WARP_M×WARP_N), iterate K
// 3. Thread Tile → Tensor Core mma.sync (M16N8K16)

// FlashMoE Processor 内调用:
fused_device_gemm(
    A = input_tile[128, 2048],
    B = expert_W1[2048, 2048],
    C = output_tile[128, 2048],
    epilogue = SiLU,  // fused in registers
    bias = expert_bias
);
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- cuBLAS = 预编译库（每次调用需 launch）；CUTLASS = 模板库（编译期嵌入）
- CUTLASS 3.x Cute 抽象提供更轻量 layout + tiling algebra
- FlashMoE 255 registers/thread 部分归因于 CUTLASS register-intensive GEMM
- Megatron-LM CUTLASS backend: 85 kernel launches; FlashMoE: 仅 1 次

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
