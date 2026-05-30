## 3D Near-Memory Processing (3D NMP) for MoE LLM Inference

术语解释
3D NMP 是通过 Hybrid Bonding 技术将 DRAM die 垂直堆叠在逻辑 die 之上，利用高密度 Cu-Cu 直接键合互连提供高带宽、高能效的近存计算架构。与 GPU 共享内存架构不同，3D NMP 中每个计算节点拥有独立的 local memory bank，通过 2D mesh NoC 互联，形成分布式内存+分布式计算的计算环境。

术语是什么？
3D Near-Memory Processing (3D NMP) 是一种将 DRAM 存储垂直堆叠在逻辑计算芯片之上的 3D 集成架构。与传统的 Processing-in-Memory (PIM) 将计算单元嵌入 DRAM die 内部不同，3D NMP 通过 Hybrid Bonding（Cu-Cu 直接融合键合）技术在 DRAM die 和 logic die 之间建立高密度互连（110,000/mm² at 3µm pitch），使得 logic die 可以独立于 DRAM 工艺节点进行定制设计。3D NMP 的关键优势在于：(1) 每个 memory bank 可被独立并行访问，提供 fine-grained 高吞吐数据访问；(2) 低寄生电容带来高能效（~0.88 pJ/b）；(3) logic die 可使用先进 CMOS 工艺获得更高计算密度。HD-MoE 论文针对 3D NMP 架构的分布式特性（无共享内存、分布式 NoC 互联），提出了混合并行映射策略。

从硬件架构角度拆解术语
在 HD-MoE 论文中，3D NMP 加速器的硬件架构由以下层次组成：
1. **Logic Die**：每个节点包含可配置的 compute unit（2.5/5/10 TFLOPS），执行 expert FFN 的 GEMM 计算（gate_proj, up_proj, down_proj）。
2. **DRAM Die**：通过 Hybrid Bonding 连接到 logic die，存储 expert 权重参数（FP32）。每个节点的 local memory bank 独立寻址，无跨节点共享。
3. **2D Mesh NoC Interconnect**：节点间通过 2D mesh 网络互联（支持 4×4, 4×8, 8×8 拓扑），每个链路带宽可配置（25/50/75 GB/s）。通信使用 XY routing（Manhattan distance 最短路径）。
4. **执行流程**：Token dispatch 阶段，每个 token 的 hidden state 通过 NoC 路由到持有其 activated expert 的节点 → 节点从 local DRAM 读取 expert 权重到 compute unit → 执行 FFN 计算 → 结果通过 NoC all-to-all combine 聚合。
3D NMP 与 GPU 和 GPU 集群的关键区别：无 shared L2 cache、无 global memory、分布式 bank-local memory 组织、有限 NoC 带宽（非 NVLink/NVSwitch）。

术语一般如何实现？如何使用？
3D NMP 的商业化进展包括 Samsung 的 HBM-PIM（在 HBM cube 中嵌入 PE，配合 AMD MI100 GPU）和 SK-Hynix 的 AiM（GDDR6 和 LPDDR 版本）。学术界提出的 3D NMP 加速器包括 H²-LLM（Hybrid-Bonding-based heterogeneous NMP for LLM）和 NeuPIMs 等。HD-MoE 使用自建 Python 离散事件模拟器来评估 3D NMP 上 MoE 推理性能，模型基于 configurable compute throughput + 2D mesh NoC bandwidth + XY routing + priority queue event scheduling。模拟器输入包括：模型参数（expert 数、hidden dim、intermediate size）、硬件配置（mesh 尺寸、TFLOPS、BW）、batch size、expert activation trace（从 MT Bench 统计），输出 Normalized TBT 和 MoE Decomposed Latency。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
