## A Survey on Mixture of Experts in Large Language Models

- baseline方法是什么？
  本论文为综述，其隐含 baseline 是：在没有系统性 MoE 知识组织的情况下，研究者分散地探索 MoE 的算法、系统和应用，缺乏统一的分类框架指导设计选择。具体而言：
  **(1) 算法 baseline**：固定 top-k 门控（k=1 或 k=2）、标准 FFN 专家（无 fine-grained segmentation）、无共享专家、无 PEFT-MoE 融合，训练从头开始。
  **(2) 系统 baseline**：基础 expert parallelism（GShard 式 All-to-All dispatch/combine），无计算优化（无定制 GPU kernel）、无通信优化（无分层/拓扑感知）、无存储优化（所有 expert 常驻 GPU）。
  **(3) 应用 baseline**：MoE 仅用于 NLP 领域的大模型预训练，缺乏跨域（CV/Recommendation/Multimodal）的系统性应用指导。

  **Baseline 全栈执行例子（以 Mixtral-8x7B 推理一个 token 为例）**：
  - 算法层：top-2 token-choice gating（Linear-Softmax-TopK）→ 2 out of 8 FFN experts 激活 → 加权求和输出 → 无 shared expert → L_aux = 0.01 负载均衡
  - 系统框架层：Expert parallelism（每个 GPU 持有部分 experts）→ Gate Routing → All-to-All Dispatch → Expert Computation → All-to-All Combine → Output Decode
  - 编译框架层：论文未明确说明（标准 PyTorch 执行，无 MoE 专用编译）
  - Kernel 调度层：标准 cuBLAS GEMM kernel → 无 block-sparse 优化 → token dropping 由 expert capacity 限制
  - 硬件架构层：标准 NVIDIA GPU HBM → 无 offloading → 全部 expert 参数驻留显存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本论文为**综述**，不提出新方法，而是建立三层分类学（Algorithm- System-Application Taxonomy，Figure 3），系统性地组织和对比现有 MoE 研究，并识别七项关键挑战（Section 7）。

  **综述对 baseline 缺陷的诊断与分类解决路径**：

  | Baseline 缺陷 | 综述识别的方法方向 | 代表性工作 |
  |---|---|---|
  | 固定 top-k 门控导致负载不均和训练不稳定 | 创新门控算法（Expert-Choice, BASE, DSelect-k）、软门控（SMEAR, Lory）、hash/随机门控 | Expert-Choice Gating [92], BASE [72], Lory [39] |
  | FFN 专家粗粒度、知识冗余 | Fine-grained expert segmentation、共享专家、新兴专家架构（MoA, MoH, LoRA experts） | DeepSeekMoE [67], Qwen1.5-MoE [102], MoA [80] |
  | 训练从头开始资源消耗大 | Dense-to-Sparse（Sparse Upcycling）、Sparse-to-Dense（蒸馏/剪枝）、Expert Models Merging（BTX） | Sparse Upcycling [47], BTX [52], DS-MoE [62] |
  | All-to-All 通信成为瓶颈 | 分层通信、拓扑感知路由、计算-通信重叠、架构解耦 | DeepSpeed-MoE [64], Lancet [143], ScMoE [108] |
  | 稀疏运算 GPU 利用率低 | 块稀疏 GEMM kernel（MegaBlocks）、PIT 编译器、ParallelLinear（ScatterMoE） | MegaBlocks [137], PIT [139], ScatterMoE [138] |
  | 专家参数超出单 GPU 显存 | 层级存储 offloading（GPU→CPU→SSD）、预测+预取、低精度加载 | SE-MoE [131], EdgeMoE [148], HOBBIT |
  | 跨域应用缺乏指导 | NLP → CV → Recommender Systems → Multimodal 系统化应用分类 | V-MoE [6], LIMoE [153], MMoE [59] |

  **综述方法论全栈执行例子**：
  本综述的方法论是通过三层分类学自上而下组织知识：
  - **算法层**：Gating Function（Sparse/Dense/Soft）× Expert Network（FFN/Attention/CNN/LoRA）× Training Scheme（Dense-to-Sparse/Sparse-to-Dense/Expert Merging）→ 构成 3×4×3 的设计空间
  - **系统框架层**：Computation（GPU kernel + 负载均衡放置）× Communication（分层 All-to-All + 拓扑感知 + 流水线重叠）× Storage（层级 offloading + 预取）→ 三维度覆盖系统全栈
  - **编译框架层**：论文提及 PIT 编译器（Permutation Invariant Transformation 变换 tile 为 dense 计算）但未深入展开
  - **Kernel 调度层**：Block-sparse kernel（MegaBlocks）、ParallelLinear grouped GEMM（ScatterMoE）、定制 encode/decode kernel（DeepSpeed-MoE/FastMoE/Tutel）
  - **硬件架构层**：层级存储（GPU HBM + CPU Memory + SSD），论文指出稀疏运算在硬件加速器上的非均匀性是关键挑战（Section 7）
