## Communication Compression for Distributed Training

术语是什么？
Mixture-of-Experts (MoE) 是一种在推荐系统中广泛使用的多任务学习架构。其核心思想是：使用多个独立的"专家网络"（expert networks，通常为 MLP + ReLU 激活）并行处理输入，再通过一个或多个可学习的门控网络（gate network）对专家输出进行软性加权组合，最后送入任务特定的预测塔（prediction tower）。相比 Shared-Bottom 等硬共享方法，MoE 通过软路由机制实现了更灵活的参数共享——不同任务可以从同一组专家中提取不同比例的共性信息，从而缓解任务冲突和负迁移。代表作包括 MMoE（Multi-gate MoE，每任务独立 gate）、PLE（Progressive Layered Extraction，引入 task-specific experts 并多层堆叠）和本论文 M3oE（引入 domain experts 和 task experts 的解耦三模块设计）。

从算法pipeline角度拆解术语：
MMoE 的计算流程（以 D 个域、T 个任务为例）：
```
输入: h_d (域d的embedding)
// Expert forward pass（N个共享专家）
for e in 1..N:
    expert_e = ReLU(LayerNorm(W_e @ h_d + b_e))

// Gate forward pass（每个任务t有独立gate）
for t in 1..T:
    gate_t = softmax(W_gate_t @ h_d + b_gate_t)  // shape: (N,)
    weighted_sum_t = sum_{e=1}^{N} gate_t[e] * expert_e

// Task-specific tower
for t in 1..T:
    y_hat_t = Sigmoid(W2_t @ ReLU(W1_t @ weighted_sum_t + b1_t) + b2_t)
```
M3oE 在此基础上扩展为三个专家模块：共享专家（N个）、域专家（D个）、任务专家（T个），并用两级融合机制替代单层 gate，实现 domain-aspect 和 task-aspect 信息的显式解耦建模。

术语一般如何实现？如何使用？
MoE 在推荐系统中通常通过 PyTorch/TensorFlow 实现：专家网络为单层或两层 MLP（含激活和归一化），gate 为线性层 + softmax。多个专家形成一个 ModuleList，gate 输出与 expert 输出做加权求和（等价于 batch matrix multiplication）。PLE 等变体进一步引入 task-specific experts 和多层 CGC（Customized Gate Control）模块堆叠。在工业界（如快手、字节跳动），MoE 被广泛用于点击率/转化率等多任务预估场景。

在 Transformer 语言模型中，MoE 层通常替换 FFN 层（Shazeer et al. 2017, Fedus et al. 2022）。每个 token 通过 Router 动态选择 top-k（通常 k=1-4）个 expert。标准实现使用 batched GEMM 计算所有 expert，但这引入了 expert capacity 约束和 token dropping/padding 问题。MegaBlocks (MLSys 2023) 通过 block-sparse 重表述将 batched GEMM 替换为 block-sparse 矩阵乘法（SDD/DSD/DDS），从根本上消除 token dropping，实现 dropless-MoE (dMoE)。已被用于训练 Mixtral 8×7B 和 DeepSeek V2。

MegaScale-Infer 从 serving 效率角度分析了 MoE 的 decoding 瓶颈：基于 Roofline Model，dense LLM 的 FFN 利用率 = min(B·F/B, 1)，MoE 的 FFN 利用率 = min(top-k/#experts · B·F/B, 1)。以 Mixtral 8×22B 在 A100（312 TFLOPS, 2 TB/s）为例，batch size 至少需 156 tokens 才能使 dense FFN compute-bound，但 MoE sparsity（top-2/8=25%）使有效 batch per expert 仅 39，MFU 仅 25%。更大的 MoE 模型（更多 experts、更低 top-k/#experts 比）sparsity 退化更严重。解耦 attention-expert 部署通过聚合多个 attention node 的请求增大 expert batch size 来逆转此退化。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---
