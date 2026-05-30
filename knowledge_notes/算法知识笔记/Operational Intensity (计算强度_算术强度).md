## Operational Intensity (计算强度/算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Operational Intensity I（计算强度/算术强度）是 Roofline Model 的核心参数，定义为单位内存访问所执行的计算量：I = FLOPs / Bytes Transferred（单位：FLOPs/Byte）。它衡量计算的"数据复用率"——I 越高，意味着每次从内存取数据后做了更多计算，对内存带宽的依赖越小。Operational Intensity 是判断 kernel 瓶颈类型（compute-bound vs memory-bound）的关键指标：将 I 与硬件 critical intensity Ī = P_peak / B_peak 比较——I ≥ Ī 则为 compute-bound，I < Ī 则为 memory-bound。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Operational Intensity 的计算取决于具体算子：
- **GEMM (M×K · K×N)**：Ops = 2×M×N×K FLOPs, Bytes = (M×K + K×N + M×N) × sizeof(dtype)。当 M,N ≫ K 时，I ≈ min(M,N) / sizeof(dtype)（高数据复用）。
- **Attention (Q·K^T)**：Ops = 2×B×H×S×d FLOPs, Bytes = B×H×(S+d)×sizeof(dtype)。对 decode (S ≫ 1, d ≈ 128)，I ≈ d / sizeof(dtype) × 2S/(S+d) ≈ 常数（极低复用，GEMV 模式）。
- **LayerNorm**：Ops = 5×B×d FLOPs, Bytes = 2×B×d × sizeof(dtype)（每个元素访问一次计算一次），I ≈ 2.5 / sizeof(dtype) ≈ 1.25（FP16，极低）。

在 MoE-Lightning 的 HRM 中，定义了 General Operational Intensity I_x^i——计算任务 x 在内存层次 level i 的操作强度。MoE FFN 的 I 随 batch size N 增大而增大（N 增大意味着更多 token 共享同一组 weights），因此更大的 batch 可以增加 I 从而跨越 PCIe bandwidth roof。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 计算方式：理论分析（从模型配置计算 FLOPs 和 bytes）或 profiling（NVIDIA Nsight Compute 自动报告 kernel 的 arithmetic intensity via Metrics: sm__throughput, dram__bytes）。
- 在 policy 搜索中的应用：MoE-Lightning 使用理论 Operational Intensity（而非 profiling）构建 HRM 性能模型，仅需硬件峰值参数——因为理论计算足以比较不同策略之间的相对效果。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
