## Arithmetic Intensity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Arithmetic Intensity（算术强度，AI）定义为计算操作的FLOP数除以内存访问的字节数：$AI = \frac{\text{FLOPs}}{\text{Bytes of memory traffic}}$，单位为FLOP/Byte。它是Roofline Model（Williams et al., 2009）的核心指标，用于判断一个操作是compute-bound还是memory-bound。对给定硬件，存在一个ridge point（脊点）：$AI_{\text{ridge}} = \frac{\text{Peak FLOP/s}}{\text{Peak Memory Bandwidth (B/s)}}$。若kernel的AI > ridge point，则为compute-bound（性能受限于计算吞吐）；若AI < ridge point，则为memory-bound（性能受限于内存带宽）。FlashAttention论文的核心贡献之一就是识别出attention操作是memory-bound的——标准attention的arithmetic intensity极低，因为每个element的softmax仅需几次arithmetic操作但需要大量的HBM读写（读写完整的$N \times N$矩阵）。通过tiling和kernel fusion提高weight的data reuse，FlashAttention将有效arithmetic intensity从memory-bound侧推向更接近ridge point，即使总FLOPs增加也因memory traffic减少而加速。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

以NVIDIA A100 GPU（FP16, Peak 312 TFLOPS, HBM BW 2.0 TB/s）为例的roofline分析：
```
                Peak Compute  ─────────────────────────────
              312 TFLOPS     │                    │
                             │  Memory-Bound      │  Compute-Bound
                             │  (slope=BW)        │  (flat)
Achieved                     │   /                │
TFLOPS                       │  /                 │
                             │ /                  │
                             │/                   │
                             ├────────────────────┤
                             ↑                    ↑
                        AI_ridge ≈ 312/2.0 = 156 FLOP/Byte

Standard Attention arithmetic intensity:
  FLOPs ≈ 2N²d (QK^T: 2N²d + softmax: ~3N² + PV: 2N²d)
  Bytes ≈ 2(Nd + N²)dtype_size (S write+read + P write+read + Q,K,V reads + O write)
  AI_standard ≈ O(d) FLOP/Byte (≈ 64-128 FLOP/Byte for typical d)
  → memory-bound (below 156 ridge)

FlashAttention arithmetic intensity (effective):
  FLOPs ≈ same (slightly higher due to recomputation)
  Bytes ≈ O(N²d²/M) (M = SRAM size, greatly reduced)
  AI_flash ≈ O(M/d) FLOP/Byte (≈ 192KB/64B ≈ 3000 FLOP/Byte)
  → much closer to compute-bound
```
FlashAttention并未改变硬件的roofline特性，而是通过reorganize computation改变了kernel在roofline上的位置——从memory-bound侧大幅向右移动，接近ridge point。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Arithmetic intensity分析在实际工程中的使用方法：(1) 使用NVIDIA Nsight Compute的`SpeedOfLight_RooflineChart` section自动测量；计算kernel的FLOPs（通过instruction count）和memory traffic（通过DRAM read/write bytes），绘制在roofline chart上；(2) 手动计算：FLOPs从算法分析得出，Bytes从profiling（如`nvprof --metrics dram_read_bytes,dram_write_bytes`）得出；(3) 优化指导：若kernel在memory-bound侧（AI < ridge），优化应聚焦减少memory traffic（kernel fusion、tiling、mixed precision）；若在compute-bound侧，优化应聚焦提升compute efficiency（减少bank conflicts、增加ILP、使用Tensor core）。FlashAttention论文利用arithmetic intensity分析论证了"减少HBM访问而非减少FLOPs"的核心设计原则。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
