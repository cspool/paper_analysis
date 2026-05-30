## Roofline Analysis for MoE Offloading（MoE Offloading 的 Roofline 分析）

术语是什么？通过联网搜索让回答具体和精准。
Roofline Analysis for MoE Offloading 是 SpecMoEOff 提出的对 MoE 模型 CPU-GPU offloading 推理的分层 Roofline 模型分析。该分析将 MoE 推理中的两种主要计算——MoE layer 和 Attention layer——分别映射到不同的硬件资源组合上：(1) MoE layer 的计算在 GPU，内存访问涉及 GPU HBM 和 CPU-to-GPU transfer（两个不同的带宽约束）；(2) Attention layer 的计算在 CPU，内存访问为 CPU DRAM。通过对比各算子的 arithmetic intensity（Operational Intensity = FLOPs / Bytes accessed）与硬件 peak compute/memory bandwidth，可视化性能瓶颈。

分层 Roofline 模型（Hierarchical Roofline Models）的核心发现：
- MoE layer (GPU compute + GPU HBM): arithmetic intensity 位于 compute-bound 区域，但仅 3.13% GPU peak 利用率
- MoE layer (GPU compute + CPU-GPU transfer): 位于 memory-bound 区域，transfer bandwidth 近乎完全利用
- Attention layer (CPU compute + CPU DRAM): memory-bound, CPU memory bandwidth 是瓶颈
- 结论：CPU-GPU transfer 是 MoE layer 的主要瓶颈，CPU memory bandwidth 是 Attention 的主要瓶颈

从算法pipeline角度拆解术语：
```
# Roofline 成本分析 (Table 1)

# MoE Layer:
computation = 3 × n_activate × b × e   # 3×矩阵乘法, e = h × h_i
memory_access_GpuHBM = n_expert × e      # 大batch: 所有expert可能激活
memory_transfer_CPU_GPU = n_activate × e × r_miss  # r_miss = cache miss rate
# Operational Intensity (GPU HBM axis):
OI = 3 × n_activate × b / n_expert      # b 小时 OI 低 → memory-bound

# Attention Layer:
computation = 2 × b × s × h             # GEMV operations
memory_access = 2 × b × s × h / g       # g = GQA group size factor
# CPU-based attention 优于 GPU-based（因 B_CPU >> B_CPU-GPU）:
# 条件: B_CPU > B_CPU-GPU → attention on CPU 更优 ✓

# Large batch is better when:
# b ≥ n_expert / (n_activate × r_miss)
```

术语一般如何实现？如何使用？
SpecMoEOff 中的 Roofline 分析用于：(1) 确定 MoE layer 和 Attention layer 的最优执行位置（GPU/CPU）；(2) 计算大 batch 与小 batch 方案的最优切换条件；(3) 论证 speculative decoding 的必要性——增大 b 和 k 提升 operational intensity。分析的硬件参数来自实际硬件配置（Table 1: A30/4090D 的 peak TFLOPS、HBM bandwidth、CPU-GPU bandwidth）。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding
