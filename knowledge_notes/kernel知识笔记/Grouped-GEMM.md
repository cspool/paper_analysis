## Grouped-GEMM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-GEMM 是将多个具有相同形状但不同数据的矩阵乘法（GEMM）批量执行的 kernel 技术。在 MoE 推理/训练中，每个 expert 的 FFN 计算是独立的 GEMM：B_i W_i（token batch × expert weight matrix）。当每 GPU 持有多个 experts 时，传统方案是 for-loop 逐个 expert 调用 cuBLAS GEMM（N 次 kernel launch，N 次 kernel launch overhead），而 Grouped-GEMM 将所有 expert 的 GEMM 合并为单次 kernel launch，消除 N-1 次 launch overhead。

LLEP 论文 Fig. 8 对此进行了基准测试：在相同总 FLOPs (B=65536 tokens 均匀分配到 N experts, D=H=8192) 下，cuBLAS 独立 GEMM（for-loop 多次 launch）vs Triton fused Grouped-GEMM（单次 launch + TMA）。结果显示 cuBLAS 的多次 launch 仍然快于 Triton 的单次 fused kernel，因为每个 cuBLAS GEMM 是硬件特定的高度优化实现（针对 NVIDIA GPU 架构级别的优化），而 Triton 版本是通用实现。这说明即使消除 launch overhead，对 hardware-optimized 的 GEMM 来说，数据布局和 tile 策略的重要性超过 kernel launch 数量。

从kernel调度角度拆解术语：

MoE 中 Grouped-GEMM 的两种实现方式对比：

```
方式 1: cuBLAS 独立 GEMM (for-loop, N 个 experts)
  for i in range(N):
      if B_i is not empty:
          output[i] = cublasGemmEx(B_i, W_i)  // 每次 launch + 硬件优化
  // N 次 kernel launch, 但每次硬件高度优化
  // 时间 = N × T_overhead + Σ(B_i × T_Bi,D,H)

方式 2: Triton fused Grouped-GEMM (单次 launch)
  @triton.jit
  def grouped_gemm_kernel(B_ptrs, W_ptrs, output_ptrs, B_sizes):
      // 单次 kernel launch, TMA 加速数据加载
      for i in range(N):
          // 所有 expert 在同一 kernel 内计算
  // 1 次 kernel launch, 但通用实现未针对硬件 tuning
```

LLEP 的发现：在 D, H 固定时，B_i 越大 GEMM 效率越高（T_B1,D,H < T_B2,D,H when B1 > B2）。因此给定固定 FLOPs，少量大 GEMM（少量 experts 大量 tokens）远快于大量小 GEMM（大量 experts 少量 tokens）。EP 和 LLEP 均利用此原理——将 experts 分布到多 GPU，每 GPU 仅计算少数 experts 的大 batch GEMM。

术语一般如何实现？如何使用？

主流 Grouped-GEMM 实现选项：
- **cuBLAS** (NVIDIA proprietary): 硬件优化的独立 GEMM，通过 `cublasGemmEx` 调用。对单个大 GEMM 效率最高。
- **CUTLASS GroupedGEMM**: NVIDIA 开源模板库，支持单次 kernel launch 执行不同形状的 GEMM（不同 B_i 和相同 W_i 形状）。
- **Triton Grouped-GEMM**: 通用实现，可搭配 TMA (Tensor Memory Accelerator) 加速 H100+ 上的数据加载。
- **MegaBlocks**: 专为 MoE 设计的 sparse GEMM kernel，支持 block-sparse 和 grouped GEMM 操作。

选择建议：大 B_i（> 1000 tokens per expert）用 cuBLAS 独立 GEMM；小 B_i 且多 experts 用 CUTLASS GroupedGEMM 减少 launch overhead；H100+ 平台上可考虑 Triton+TMA 的融合 kernel。

涉及论文标题：
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts
