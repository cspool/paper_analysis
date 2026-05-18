## Hierarchical Software Pipeline for GPU Kernels（GPU Kernel的两级软件流水线）

术语是什么？通过联网搜索让回答具体和精准。
Hierarchical software pipeline是GPU kernel中使用多级double buffering隐藏memory latency和compute latency的流水线设计。ZipGEMM首次提出两级pipeline用于fused decompression-GEMM：Coarse-level（tile级）用shared memory double buffering重叠global→shared传输与计算；Fine-level（slice级）用ALU/Tensor Core交错重叠decompression与MMA。Quantix进一步发展了该技术用于non-uniform dequantization-matmul：Inter-tile级用Smem0/Smem1双buffer重叠cp.async prefetch与dequant+MMA；Intra-tile级用Reg0/Reg1双buffer重叠CUDA core dequantization与Tensor Core MMA。两级barrier协调：cp.async.wait_group<0>() + __syncthreads()同步inter-tile buffer切换；intra-tile内warp的SIMT lockstep执行天然同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两级pipeline执行timeline（以ZipGEMM为例，单个thread block内）：
```
时间 →
Buffer A: [async load tile 0] [---idle---] [async load tile 2] ...
Buffer B: [---idle---] [async load tile 1] [---idle---] ...
Compute:  [---idle---] [decomp+mma tile 0 slices] [decomp+mma tile 1 slices] ...

单个tile内的fine-level slice interleaving：
  slice 0: [load+decomp w0→regs] [mma w0 × act]
  slice 1:                           [load+decomp w1→regs] [mma w1 × act]
  slice 2:                                                     [load+decomp w2→regs] ...

  ALU:  [decomp s0][decomp s1][decomp s2]...
  TC:            [mma s0] [mma s1] [mma s2]...
```

Quantix的两级pipeline执行timeline：
```
// Inter-tile (K-tile granularity, shared memory double buffering):
//   Smem 0: [cp.async W'/C/A tile 0] [dequant+mma tile 0 subtiles]
//   Smem 1:                          [cp.async W'/C/A tile 1]              [dequant+mma tile 1 subtiles]

// Intra-tile (subtile granularity, register double buffering):
//   Reg 0: [ld.shared + dequant s0] [mma s0] [ld.shared + dequant s2] [mma s2] ...
//   Reg 1:            [ld.shared + dequant s1] [mma s1]           [ld.shared + dequant s3] ...

// 三级overlap:
//   Global→Shared: [cp.async tile 0] [cp.async tile 1] ...
//   Shared→Reg + CUDA Cores (dequant): [dequant s0][dequant s1][dequant s2]...
//   Tensor Cores (mma):                     [mma s0] [mma s1] [mma s2]...
```

ZipGEMM设计使ALU利用率达66.0%（来自decompression的LOP3/IADD/POPC指令），Tensor Core利用率保持cuBLAS的71.6%。Quantix的ablation显示：禁用pipeline（全部序列执行）性能降至完整版本的约41%，证明两级pipeline对隐藏dequantization latency的关键作用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **Coarse/Inter-tile级**：两个shared memory buffer（weights + activations + centroids），通过cp.async将数据从global memory异步加载到"下一个"buffer，__syncthreads() barrier确保所有线程完成当前tile计算后再切换
2. **Fine/Intra-tile级**：ZipGEMM在每个tile内手动unroll K维度的slice循环，交错安排load指令和mma指令；Quantix用两个register buffer（Reg0/Reg1），当Reg0做dequantization时Reg1被Tensor Cores消费
3. Barrier策略：inter-tile用cp.async.wait_group<0>()等待所有async copy完成 + __syncthreads()所有线程同步；intra-tile内warp的SIMT lockstep执行天然同步
4. Quantix的pipeline需要3类数据同时流动：W1'/W2' (packed indices)、C (centroids)、A (activations)，比ZipGEMM的2类数据（compressed weights + activations）更复杂
5. 该技术可推广至其他需要重叠memory/preprocessing/computation的GPU kernel设计

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

