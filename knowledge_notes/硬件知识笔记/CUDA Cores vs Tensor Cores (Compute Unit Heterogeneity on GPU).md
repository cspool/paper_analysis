## CUDA Cores vs Tensor Cores (Compute Unit Heterogeneity on GPU)

术语是什么？
CUDA Cores和Tensor Cores是NVIDIA GPU上两种不同吞吐量和设计目标的执行单元。CUDA Cores是通用SIMT执行单元，处理整数/浮点标量运算、地址计算、逻辑和控制流；Tensor Cores是专用矩阵乘累加单元，在一个时钟周期内完成一个小矩阵乘法（如m16n8k16 FP16），提供远高于CUDA Cores的计算吞吐。在H100上，CUDA Cores峰值约60 TFLOPS (FP32)，而Tensor Cores INT8峰值约990 TFLOPS——约16.5×差距。这一巨大差距是W4A8 dequantization瓶颈的根本原因：dequantization在CUDA Cores上执行，若每元素需10+指令，CUDA Cores无法跟上Tensor Cores的消费速度。

从硬件架构角度拆解术语：
LiquidGEMM的H100/H800 GEMM pipeline中的异构执行：
```
异构硬件单元及其角色 (LiquidGEMM ImFP pipeline):
  TMA (DMA Engine):      GMEM → SMEM 异步weight搬运
  CUDA Cores (SIMT):     Unpack 4-bit + LQQ dequantization (IMAD+XOR)
  Tensor Cores (MMA):    WGMMA INT8 matrix multiply-accumulate

Pipeline overlap (时间线上并发):
  Time →  |--- TMA load tile_k ---||--- TMA load tile_{k+1} ---|
           |--- CUDA Dequant tile_{k-1} ---||--- CUDA Dequant tile_k ---|
           |--- TC MMA tile_{k-2} ---||--- TC MMA tile_{k-1} ---|

去量化的性能瓶颈来自Φ_CUDA << Φ_TC:
  - dequantization在CUDA Cores上序列化执行
  - MMA在Tensor Cores上异步后台执行
  - 若T_DQ > T_MMA, CUDA Cores成为瓶颈, Tensor Cores闲置
  - 解决: 降低α (LiquidQuant) + pipeline overlap (ImFP)
```

LiquidGEMM cost model (Section 3.2)量化了这一异构性：
- T_DQ = α · N · K / Φ_CUDA（dequant在CUDA Cores上的时间）
- T_MMA = min(Mt,M) · 2 · N · K / Φ_TC（MMA在Tensor Cores上的时间）
- 在compute-bound时（M大），T_MMA减小，T_DQ相对增大→dequant更成瓶颈
- 在memory-bound时（M小），T_LD主导，T_DQ仍不可忽略

术语一般如何实现？如何使用？
CUDA Cores和Tensor Cores的异构性是现代GPU编程的核心约束。Warp specialization（CUTLASS, FlashAttention-3）利用这种异构性将不同任务分配到不同warp：producer warp用CUDA Cores+TMA做data movement，consumer warp用Tensor Cores做MMA。LiquidGEMM的ImFP进一步利用warp group级别的异构：Load WG(TMA)、Compute WG(CUDA+Tensor)并发。关键设计原则：最小化CUDA Cores上的工作负载（LiquidQuant的α=0.875 vs QServe的α≥10），使CUDA Cores不成为pipeline瓶颈。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
