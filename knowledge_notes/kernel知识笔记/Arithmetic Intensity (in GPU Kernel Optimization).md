## Arithmetic Intensity (in GPU Kernel Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Arithmetic Intensity（算术强度）是 GPU kernel 优化的核心概念，定义为计算操作数（FLOPs）与内存访问量（bytes）的比值：AI = FLOPs / Bytes。在 Roofline 性能模型中，算术强度决定 kernel 是 compute-bound 还是 memory-bound。每个 GPU 有一个由硬件决定的 critical arithmetic intensity I* = Peak_FLOPS / Peak_Memory_Bandwidth。例如 NVIDIA A100（FP16 Tensor Core）：312 TFLOPS / 2.0 TB/s ≈ 156 FLOP/byte（实际 kernel 因 SRAM/cache hierarchy 约 12-15 FLOP/byte 即达 compute-bound 临界点，因为 kernel 的 effective bandwidth 远低于 peak HBM bandwidth）。

当 AI < I* 时 kernel 为 memory-bound（性能受限于 HBM 带宽→优化目标为减少内存访问）；当 AI > I* 时为 compute-bound（性能受限于 FLOPS→优化目标为减少计算量）。Full Attention 在 training/prefilling 阶段为 compute-bound（大 batch matmul），在 decoding 阶段为 memory-bound（每次只生成 1 个 token 却需加载全部 KV cache）。NSARR 通过减少 KV cache 加载量（memory-bound decoding 加速）和 group-centric 设计提升算术强度（compute-bound training 加速）实现双向优化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**Roofline 分析示例：NSA vs Full Attention 在 A100 上**

```
// === Full Attention Decoding (per token, T=64k, d=128) ===
// 计算量: Q(1,d) @ K(T,d)^T + softmax + attn @ V
//  FLOPs ≈ 2×T×d + ... ≈ 2×65536×128 ≈ 16.8M FLOPs
// 内存: 加载 Q(128) + K(65536×128) + V(65536×128) ≈ 16.8M elements ≈ 33.6MB (BF16)
// Arithmetic Intensity = 16.8M FLOPs / 33.6 MB ≈ 0.5 FLOP/byte
// → 远低于 A100 critical ≈ 12.5 FLOP/byte → MEMORY-BOUND
// → 优化方向: 减少 KV cache 加载量

// === NSA Decoding (per token, T=64k, d=128) ===
// 压缩 KV: 4096 tokens; 选择 KV: 1024 tokens; 窗口 KV: 512 tokens
// 总等效 KV 访问: 5632 tokens
// 内存: Q(128) + KV(5632×128) ≈ 1.44M elements ≈ 2.88MB (BF16)
// FLOPs ≈ 2×5632×128 ≈ 1.44M FLOPs
// AI = 1.44M / 2.88MB ≈ 0.5 FLOP/byte (仍 memory-bound)
// 但内存访问量降 11.6× → 实际加速 ≈ 11.6×（因为 memory-bound）
// → 加速比 ≈ 内存访问量之比 = 65536/5632 ≈ 11.6×

// === NSA Training/Prefilling (T=64k, B=8, d=128, H=16) ===
// Group-centric kernel: 每个 inner loop iteration
//   计算: H × B_k × (2d_k + 3d_v) ≈ 16×64×(2×128+3×128) ≈ 655K FLOPs
//   内存: B_k × (d_k + d_v) ≈ 64×(128+128) ≈ 16K elements ≈ 32KB (BF16)
//   AI ≈ 655K / 32KB ≈ 20 FLOP/byte
// → 超过 A100 critical ≈ 12.5 → COMPUTE-BOUND
// → 对比 FA2 (query block连续加载导致碎片化内存访问)
//   FA2 AI ≈ 1×64×(2×128+3×128) / 32KB ≈ 1.25 FLOP/byte → MEMORY-BOUND
// → NSA kernel 将 sparse attention training 从 memory-bound 转为 compute-bound
```

**Critical Arithmetic Intensity 计算**（A100 80GB PCIe）：
| 指标 | 值 |
|------|-----|
| Peak FP16 Tensor Core FLOPS | 312 TFLOPS |
| Peak HBM2e Bandwidth | 2,039 GB/s |
| Critical AI (理论) | 312,000 / 2,039 ≈ 153 FLOP/byte |
| Critical AI (实测, kernel 级) | ~12-15 FLOP/byte (effective bandwidth < peak) |

术语一般如何实现？如何使用？

在实际 GPU kernel 优化中使用算术强度分析的典型流程：(1) 用 NVIDIA Nsight Compute 或手动计算 kernel 的 FLOPs 和 HBM traffic；(2) 在 Roofline 图上定位当前 kernel；(3) 若 memory-bound：减少 HBM 访问（shared memory tiling、kernel fusion、数据压缩）、提升访问模式效率（coalescing、bank conflict avoidance）；若 compute-bound：减少计算量（稀疏化、量化）、提升计算效率（Tensor Core 利用率、指令级并行）。

在 LLM 注意力优化中：Decoding 阶段优化 → memory-bound → 目标减少 KV cache 加载（GQA/MQA/KV cache compression/NSA-like sparse selection）；Training/Prefilling 阶段 → compute-bound → 目标减少总 FLOPs（FlashAttention 的 tiling+recomputation、NSA 的 blockwise sparse computation）。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---
