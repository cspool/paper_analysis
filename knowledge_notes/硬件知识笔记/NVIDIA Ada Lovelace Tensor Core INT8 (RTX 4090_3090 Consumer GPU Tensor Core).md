## NVIDIA Ada Lovelace Tensor Core INT8 (RTX 4090/3090 Consumer GPU Tensor Core)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA Ada Lovelace Tensor Core (RTX 4090) 是 consumer GPU 上的第四代 Tensor Core，提供 INT8 和 FP16 矩阵乘加硬件加速。RTX 4090 (AD102 die, 128 SMs, boost clock ~2520 MHz) 的 Tensor Core 理论吞吐：INT8 = 660 TOPS (non-sparse) / 1320 TOPS (sparse)；FP16 = 330 TFLOPS (non-sparse) / 660 TFLOPS (sparse)。关键特性：(1) INT8 MMA (u8.u8.s32) 指令使 consumer GPU 上的 INT8 推理达到数据中心 GPU (A100: 624 TOPS) 相当甚至更高的 INT8 吞吐；(2) FP16+FP16 accumulator MMA (f16.f16.f16) 每个 SM 每 cycle 512 FMA——是 FP16+FP32 accum (256 FMA) 的 2× 加速；(3) 注意 A100/H100 数据中心 GPU 无此 consumer GPU 独有特性——它们以 FP32 accum 为默认快速路径。SageAttention 充分利用这两个特性：QK^⊤ 用 INT8 MMA (2× faster than FP16)、PV 用 FP16+FP16 accum MMA (2× faster than FP32 accum)，合计约 2× overall speedup。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SageAttention 在 RTX 4090 上的执行流程按 Tensor Core 层次：
```
SM (Streaming Multiprocessor) x 128, each SM has:
  4 Tensor Cores (4th gen), 128KB SRAM, 65536 registers (256KB)

Single Tensor Core INT8 MMA instruction:
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  → 16×8 output tile × K=32 → 2×(16×8×32) = 8192 INT8 ops/instruction
  → Throughput: 1 inst/cycle/TensorCore → per SM: 4×8192 = 32768 ops/cycle

FP16+FP16 accum MMA:
  mma.sync.aligned.m16n8k16.row.col.f16.f16.f16
  → 16×8 output tile × K=16 → 2×(16×8×16) = 4096 FP16 ops/instruction
  → Throughput: 1 inst/cycle/TensorCore (FP16 accum mode)
  → FP32 accum mode: throughput halved → 1 inst/2cycles/TensorCore
```

Tiling 映射到 Tensor Core:
- INT8 QK^⊤: Q̂_i [128×64] INT8 @ K̂_j^T [64×64] INT8 → 128×64 output → 4×8 = 32 MMA instructions (M=16,N=8,K=32 tiles)
- FP16 PV: P̃_ij [128×64] FP16 @ V_j [64×128] FP16 → 128×128 output → 8×8 = 64 MMA instructions (M=16,N=16,K=16 tiles)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编程方式：Triton `tl.dot()` 自动映射到 Tensor Core 指令（无需手写 PTX）。CUDA: 使用 `nvcuda::wmma` 或直接 PTX inline assembly。SageAttention 实测 340 TOPS 在 RTX4090 上——达到 INT8 理论峰值 660 TOPS 的 52%（FlashAttention2 在 FP16 峰值 330 TFLOPS 下仅 165 TFLOPS = 50%）。利用率瓶颈：tiling 导致部分 MMA tile 的边缘浪费 + softmax non-matmul overhead + DRAM bandwidth（HBM: 1008 GB/s）。RTX 3090 (Ampere, GA102, 82 SMs): 实测 131.74 TOPS (UltraPixel headdim=64, 2.00× vs FlashAttn2 65.86 TOPS)。开源: SageAttention https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
