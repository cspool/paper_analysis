## Memory Reuse Strategy in Distributed Training

术语是什么？
Memory Reuse Strategy 是 MPMoE 提出的通过共享 buffer 减少 MoE 训练 activation memory 占用的技术。核心观察：在 micro-batch pipeline 中，不同 partition 的 tensors（T_DI, T_M, T_DO）在不同时间点激活，产生"memory bubbles"，因此可共享同一个物理 buffer。原本 n 个 partition 各需独立 buffer，共享后仅需 1 份，内存从 O(n·B·M) 降至 O(B·M)。共享 buffer 意味着前向中被覆写的 tensors 需在后向中恢复，MPMoE 提出 4 种恢复策略（S1-S4），组合 CPU offload、通信重放和重计算三种机制。

从算法pipeline角度拆解术语：
4 种策略的恢复方法（Table 2）：

| 策略 | T_DI 恢复方式 | T_M 恢复方式 | 适用场景 |
|------|-------------|------------|---------|
| S1 | CPU offload | CPU offload | N 小（计算瓶颈） |
| S2 | 通信重放 | CPU offload | N 小-中 |
| S3 | CPU offload | 重计算 | N 中-大 |
| S4 | 通信重放 | 重计算 | N 大（通信瓶颈） |

内存节省公式（Equation 5-6）：ΔM_act = ΔM_buf = B * (2M*(n-2)/n + H*(n-1)/n)，n=8 时达 ~38% 节省，最高 vs FasterMoE 节省 53%。

术语一般如何实现？如何使用？
- 类似技术：Gradient Checkpointing（重计算所有 activations）、ZeRO-Offload（offload optimizer states）、vDNN（offload activations）。MPMoE 的创新在于 buffer 共享 + 选择性恢复（组合 offload/recompute/communication replay）联合应用于 MoE pipeline 场景。
- 实现要点：(a) pinned memory 支持异步 D2H/H2D；(b) 不同 CUDA stream 上 overlap；(c) 通信重放依赖原始 T_I 保留在内存中。
- 局限性：(a) PCIe 带宽限制（V100 ~32 GB/s vs HBM ~900 GB/s）；(b) 实际内存节省约理论上限的 95%（Figure 12）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---
