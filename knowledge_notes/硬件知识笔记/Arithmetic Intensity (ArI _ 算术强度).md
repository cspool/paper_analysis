## Arithmetic Intensity (ArI / 算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Arithmetic Intensity (ArI) 是衡量计算任务每字节内存访问所执行浮点操作数的指标，单位为 Op/B（Operations per Byte）。ArI 是 Roofline Model 的核心参数，用于判断一个计算任务在特定硬件上是 memory-bound 还是 compute-bound。计算公式：$\text{ArI} = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$。当 ArI 低于硬件的 Ridge Point 时，性能受内存带宽限制（memory-bound），执行时间 = Memory Access / BW；当 ArI 高于 Ridge Point 时，性能受计算吞吐限制（compute-bound），执行时间 = FLOPs / Throughput。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文通过 ArI 分析了 LLM 推理中各层的瓶颈特性并指导系统优化：

**传统 MHA 的 Core-Attention 层 ArI ≈ 1**：decode 时 Score = Q @ K^T，FLOPs = 2×B×n_hd×L×d_hd，Memory Access = 2×B×n_hd×L×d_hd（读 Q 和 K，各 L×d_hd 但 K 无法跨 batch 共享），ArI ≈ 1 Op/B。远低于任何现代加速器的 Ridge Point，始终 memory-bound。

**MLA + Layer Reordering 的 Core-Attention 层 ArI ≈ 100-200**：FLOPs = 2×B×n_hd×L×d_KVco（d_KVco 替代 d_hd），Memory Access = 2×B×(d_KVco×L + n_hd×L)，ArI ≈ (n_hd^-1 + d_KVco^-1)^-1 ≈ 100 Op/B。FlashMLA 通过复用 C_KV 实现 ~200 Op/B，接近 B200 的 Ridge Point 281.25。

**FC 层的 ArI 随 batch size 增长**：decode 时 FC 层为 GEMV（ArI 低），large batch 转换为 GEMM（ArI 高）。MoE 的 FC 层 ArI 由 B × n_k/n_e 决定，需 B ≥ RP_acc × n_e/n_k = B_MoE 才能达到 Ridge Point。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ArI 分析在论文中作为指导系统设计的核心工具：(1) 判断 MLA + reordering 使 attention 从 memory-bound 变为接近 compute-bound，消除了 attention-specialized PIM 硬件的需求；(2) 推导 $B_{\text{RP}} = \max(RP_{\text{acc}} \cdot deg_{\text{DP}}, RP_{\text{acc}} \cdot n_e/n_k)$ 作为 batch size 目标；(3) 评估不同加速器（V100→B200，RP 从 139 升到 281）对层性能的 bound 转换。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
