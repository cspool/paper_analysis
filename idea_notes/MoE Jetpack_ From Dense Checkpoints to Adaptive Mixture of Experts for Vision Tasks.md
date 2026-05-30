## MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

- baseline方法是什么？
  Baseline 为 Soft MoE [6]（from scratch training），即从随机初始化训练 MoE 模型，不使用任何预训练 dense checkpoint。以 ViT-T 架构上的 Soft MoE 为例说明全栈执行路径：
  - **算法层（MoE Routing）**：输入 token X ∈ R^{m×d} → 通过 learnable parameters Φ ∈ R^{d×(e·s)} 将 m 个 token 映射到 e×s 个 slot：X̃ = softmax(XΦ)^T X → e 个 expert（MLP）各自处理 s 个 slot → 输出 Ỹ → 通过 softmax(XΦ) 重组为 m×d token 输出。**缺陷**：(1) MoE 模型从随机初始化训练，缺乏预训练知识的加速，在小数据集上收敛极慢（如 STL-10 仅 67.7% accuracy）；(2) Soft MoE routing 使用 dense Φ 矩阵对所有 token 做 soft assignment，未专门设计以适配从 dense checkpoint 继承的权重分布，导致优化困难和 expert 过度特化（over-specialization）；(3) 所有 expert 大小一致，对重要性不同的 token 分配等量计算资源，计算冗余。
  - **系统框架层**：使用 MMPretrain（OpenMMLab 预训练工具箱）实现，标准 PyTorch 训练循环。训练使用 AdamW optimizer、cosine decay LR schedule、RandAugment/Mixup/CutMix 等数据增强。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：传统 MoE 用 for-loop 逐个 expert 处理 token。论文指出 vanilla MoE 设计不提供运行时加速，需要额外并行策略。论文在 Appendix C 提供了高效的并行 expert 前向实现（合并所有 expert 权重为单个大矩阵，单次 einsum 替代多个逐 expert 操作）。
  - **硬件架构层**：NVIDIA RTX 4090 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 MoE Jetpack 通过两个核心技术逐层解决 Baseline 缺陷：
  1. **Checkpoint Recycling（解决从零训练的收敛慢和资源消耗）**：不随机初始化 MoE expert，而是从预训练 dense checkpoint（ImageNet-21K 预训练的 ViT-S/ConvNeXt-T）中通过 Importance-Based Weight Sampling 提取权重来初始化 MoE expert。具体地：跑一批图像通过 predecessor dense 模型获取每个 channel 和 hidden neuron 的 activation 值，channel 层按跨层平均 activation 排序选 top-d'，hidden neuron 按 activation 概率分布采样分配给不同 expert 以保证 diversity。这使 MoE 模型继承了 dense 模型的预训练知识，大幅加速收敛（ImageNet 上 2× 加速，CIFAR-100 上 8× 加速）并有显著精度提升。相比 Sparse Upcycling [16]（仅复制 MLP），checkpoint recycling 能利用更大/不同的 dense checkpoint 构造不同大小的 expert，灵活性更高。
  2. **SpheroMoE Layer（解决优化困难和 expert 过度特化）**：
     - SpheroMoE Routing：用 cross-attention 替代 Soft MoE 的线性 Φ 分配。随机初始化的 Q 经 L2 normalize 投影到超球面，与从 input token（经继承自 dense checkpoint 的 LayerNorm 处理后）投影得到的 K 计算 cosine similarity，解决了随机初始化 Q 导致的数值不稳定和与 dense checkpoint 分布不一致的问题。
     - Expert Regularization：learnable softmax temperature T（早期大→分散注意力，逐步减小→专精）+ Gaussian noise 到相似度 logits + stochastic expert dropout（概率 p），三者共同防止 expert 对特定 token 的过度聚焦和对特定 expert 的过度依赖。
     - Adaptive Dual-path MoE：利用 checkpoint recycling 赋予的 dense 先验知识区分重要/非重要 token。Core experts（数量少，占总数 1/3，每个完整 hidden dim 4d'）处理高重要性 token；Universal experts（数量多，每个 hidden dim ≈ d'，约 1/4 参数）处理低重要性 token。在保持 FLOPs 不变的前提下提升性能。

  MoE Jetpack 全栈执行路径（与 baseline 同框架对应）：
  - **算法层（Dense→MoE Fine-tuning）**：ImageNet-21K 预训练 ViT-S dense checkpoint → Checkpoint Recycling（Importance-Based Weight Sampling）→ 初始化 V-JetMoE-T 的 expert 权重 → 前 N/2 层保留 dense ViT 结构（继承全部 dense 权重）→ 后 N/2 层为 SpheroMoE 层：input token X → 继承的 LayerNorm → Q L2-norm 超球面投影 → cross-attention 计算 dispatch/combine logits S → Adaptive Dual-path 分离 core/universal token → 并行 expert 前向（合并权重矩阵单次 einsum）→ softmax 重组 token。
  - **系统框架层**：PyTorch 2.1.0 + MMCV 2.1.0 + MMPretrain。训练配置：AdamW（lr=4e-3, weight_decay=0.05, β=(0.9,0.999)）、batch_size=4096（ImageNet）/512（其他）、300 epochs、cosine decay、50 warmup epochs。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文提供了并行 expert 前向实现（Appendix C, Algorithm 1），将所有 expert 的 weight_1 合并为 e×d2×d1 大矩阵，单次 einsum("b e s d1, e d2 d1 -> b e s d2") 平行处理所有 b×e×s 个 slot，替代传统 for-loop。FLOPs 与原始 dense 模型相当（V-JetMoE-T: 1.1G vs ViT-T: 1.1G）。
  - **硬件架构层**：NVIDIA RTX 4090。训练 V-JetMoE-T 在 ImageNet-1K 上需要 120 GPU hours，与原始 dense ViT-T 训练时间几乎相同（论文称 "nearly equivalent training times"）。
