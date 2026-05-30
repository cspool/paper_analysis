## PCIe Bottleneck in MoE Inference (MoE 推理中的 PCIe 瓶颈)

术语解释
PCIe Bottleneck in MoE Inference 是指 memory-constrained MoE 推理中通过 PCIe 总线从 CPU memory 加载 offloaded expert 参数到 GPU 的传输延迟成为端到端推理延迟的主导因素。单个 expert 参数可达数百 MB，PCIe 4.0 带宽仅 ~32 GB/s，单 expert 传输 ~10ms，远超 GPU FFN 计算时间 ~1-2ms，导致推理从 compute-bound 转变为 I/O-bound。

术语是什么？
BuddyMoE 论文量化的关键数据：(1) edge devices 上 Mixtral-8×7B 的 CPU→GPU expert 传输占 85-94% 推理延迟；(2) PCIe 带宽 16-32 GB/s vs GPU HBM TB/s 级——数量级差距；(3) cache miss 同步加载 ~9-10ms vs prefetch 命中 ~0ms；(4) 随 expert 数量增长（8→64→256→2048），GPU cache 命中率下降→miss 率上升→bottleneck 恶化。BuddyMoE 核心解法：用 GPU-resident buddy expert 替代 CPU-resident expert，避免 PCIe 传输。实测 PCIe read bandwidth 比 baseline 减少 ~20%（Figure 8）。

从硬件架构角度拆解术语：
PCIe bottleneck 的硬件层级数据流：CPU DRAM (expert weights) → PCIe 4.0 ×16 bus → GPU HBM → SM。BuddyMoE 的 avoidance path：GPU SM → L2 cache → 查 B_ℓ (GPU HBM) → 查 M mask (register) → atomic CAS (L2 atomic) → 写 S' (register) → FFN GEMM (GPU HBM, no PCIe)，全程 GPU 内部。

术语一般如何实现？如何使用？
- 缓解策略：(1) Hardware: NVLink/CXL/NDP；(2) System: prefetching/buddy replacement/caching；(3) Algorithm: pruning/quantization（减少 expert 字节数）
- 在 edge/mobile devices 上 PCIe bottleneck 更严重（带宽更低）
- PCIe 5.0 (64 GB/s) 理论上 halve 延迟但仍远低于 HBM

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

**补充（来自 Compression Error Sensitivity Analysis）**：该论文量化了 MoE offloading 中 PCIe 瓶颈的严重程度：(1) Mixtral-8x7B FP16 推理需 ~94 GB VRAM，仅约 30% 参数（~27.5 GB）在解码期间活跃使用，其余 66.6 GB 被非激活 expert 占据；(2) PCIe 4.0 (32 GB/s) vs GPU 内存带宽 (~300 GB/s) —— 数量级差距；(3) 论文提出用 error-bounded lossy compression (SZ3/CuSZp) 压缩通过 PCIe 传输的 expert 参数，作为缓解 PCIe bottleneck 的 Algorithm 层面策略，补充了 pruning/quantization 之外的压缩方向。
