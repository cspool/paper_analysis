## MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE Jetpack，一种将预训练 dense checkpoint 转换为 MoE 模型的 fine-tuning 框架，由两部分组成：
    1. **Checkpoint Recycling（检查点回收）**：将预训练 dense 模型（predecessor）的 MLP 权重通过四种策略分配到 MoE 模型（successor）的多个 expert 中，避免从零训练 MoE。默认策略为 Importance-Based Weight Sampling：通过在 predecessor 上跑一批图像获取每层每个 channel 和 hidden neuron 的 activation 值，对 channel 按跨层平均 activation 选 top-d'，对 hidden neuron 按 activation 概率分布采样分配给不同 expert。其他策略包括 Co-Activation Graph Partitioning（用 Metis 图分割将共激活神经元分入同一 expert）、Uniform Selection（等距采样）和 Random Sampling。
    2. **SpheroMoE Layer（超球面自适应 MoE 层）**：优化 dense checkpoint 到 MoE 的 fine-tuning，包含三个改进：(a) SpheroMoE Routing：用 cross-attention 将 input token 分配到 expert slots，查询向量 Q 随机初始化并 L2-normalize 投影到超球面（避免随机初始化的数值不稳定），key 由 input token 的 LayerNorm 后线性投影得到，在超球面计算相似度 logits；(b) Expert Regularization：learnable softmax temperature T（早期大→均匀分散注意，逐步减小→专精）+ expert noise + stochastic expert dropout（概率 p 随机停用 expert）；(c) Adaptive Dual-path MoE：核心专家（Core experts，数量少参数大）处理高重要性 token + 通用专家（Universal experts，数量多参数约 1/4）处理低重要性 token。通过 checkpoint recycling 获得的 dense 先验知识帮助区分重要/非重要 token。
  - 实验比较：(1) MoE Jetpack vs Dense ViT/ConvNeXt（from scratch 和 ImageNet-21k pretrained）vs Soft MoE（from scratch）；(2) 消融：Checkpoint Recycling + Soft MoE vs 单独 SpheroMoE；(3) Checkpoint Recycling 四种策略 vs Sparse Upcycling [16]；(4) Core Experts Ratio 消融（1/3 最优）；(5) MoE layer 配置消融：层数范围（7:12 最优）、expert 数量、dense checkpoint 基础模型大小。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090。
  - V-JetMoE-T 训练 ImageNet-1K：120 GPU hours；CIFAR-100：2.5 GPU hours。
  - C-JetMoE-F 训练 ImageNet-1K：156 GPU hours；CIFAR-100：2.5 GPU hours。
  - V-JetMoE-S 训练 ImageNet-1K：200 GPU hours；CIFAR-100：8 GPU hours。
  - 论文总训练 GPU hours：约 3300 GPU hours（含探索验证约 8000）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Vision Transformer (ViT-S/16, ViT-T) 和 ConvNeXt (ConvNeXt-T, ConvNeXt-F)。Dense predecessor 使用 ImageNet-21K 预训练权重（来自 timm）。
  - MoE 后继模型：V-JetMoE-T（FLOPs 1.1G, core experts 98, universal experts 196, MoE layers 7:12）、C-JetMoE-F（FLOPs 1.1G）、V-JetMoE-S（FLOPs 4.3G）。
  - 数据集（8 个图像分类）：ImageNet-1K, CIFAR-10, CIFAR-100, Flowers, Pets, STL-10, Food-101, DTD。
  - Benchmark 指标：Top-1 Accuracy (%)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Adlith/MoE-Jetpack（NeurIPS 2024）
  - 框架：PyTorch 2.1.0 + MMCV 2.1.0 + MMPretrain（OpenMMLab）
  - 算法 Pipeline 伪代码：

```
# === Phase 1: Checkpoint Recycling（离线，一次性） ===

# 输入：predecessor dense 模型 P (N layers, channel dim d, hidden dim 4d)
# 输出：successor MoE 模型 S (N layers, channel dim d', hidden dim 4d', 
#        前半 N/2 dense layers + 后半 N/2 SpheroMoE layers)

# Importance-Based Weight Sampling:
images = sample_batch(dataset)  # 一批图像
activations = forward_pass(P, images)  # 获取每层各 channel/hidden neuron 的 activation

# Channel 选择：跨层平均 activation，取 top-d'
for c in range(d):
    A_c = mean([activations[l][c] for l in range(N)])
selected_channels = top_k(A_c, d')  # 选 activation 最高的 d' 个 channel

# Hidden neuron 选择：按 activation 概率分布为每个 expert 采样
for h in hidden_neurons:
    P_h = A_h / sum(all_A)  # 概率分布
for each expert e:
    expert_neurons[e] = sample_from_distribution(P_h, 4d')

# 从 predecessor 权重矩阵中提取相应行/列构造 expert 的 MLP 权重

# === Phase 2: SpheroMoE Layer 前向传播（训练 + 推理） ===

# 输入：X ∈ R^{b×n×d}（batch, token_num, channel）
# Q ∈ R^{e*s×d}（随机初始化，e experts * s slots per expert）

def spheromoe_forward(X, Q, T, core_experts, univ_experts):
    # 1. 继承 dense checkpoint 的 LayerNorm
    X_norm = inherit_layer_norm(X, dim=-1)  # b×n×d
    
    # 2. 超球面投影：Q 通过 LayerNorm + L2 normalize
    Q_norm = l2_norm(inherit_layer_norm(Q, dim=-1))  # e*s×d
    
    # 3. Key 投影
    K = linear(X_norm, W_k)  # b×n×d
    
    # 4. 超球面相似度计算（点积）
    S = einsum(K, Q_norm, "b n d, e s d -> b n e s")  # b×n×e×s
    
    # 5. Expert Regularization
    S = S + normal_noise(S) * noise_mult  # 加噪声
    dispatch = softmax(S / T, dim=1)       # temperature-scaled, b×n×e×s
    combine = softmax(S / T, dim=[-1,-2])  # b×n×e×s
    
    # 6. Token 分发到 expert slots
    X_hat = einsum(dispatch, X_norm, "b n d, b n e s -> b e s d")  # b×e×s×d
    
    # 7. Adaptive Dual-path: 分离 core 和 universal experts
    X_core = X_hat[:, :core_num, :, :]    # b×core_num×s×d
    X_univ = X_hat[:, core_num:, :, :]    # b×univ_num×s×d
    
    # 8. 并行 expert 前向（合并所有 expert 权重为一个大矩阵，单次 matmul）
    # parallel_expert_forward 等价于:
    #   x = einsum(x, experts.weight_1, "b e s d1, e d2 d1 -> b e s d2")
    #   x = x + rearrange(experts.bias_1, "e d2 -> () e () d2")
    #   x = experts.act(x)
    #   x = einsum(x, experts.weight_2, "b e s d1, e d1 d2 -> b e s d1")
    #   x = x + rearrange(experts.bias_2, "e d1 -> () e () d1")
    Y_core = parallel_expert_forward(X_core, core_experts)
    Y_univ = parallel_expert_forward(X_univ, univ_experts)
    Y_hat = concat([Y_core, Y_univ], dim=1)  # b×e×s×d
    
    # 9. Expert dropout（随机停用 expert）
    Y_hat = expert_dropout(Y_hat, p)
    
    # 10. Token 重组
    Y = einsum(combine, Y_hat, "b n e s, b e s d -> b n d")  # b×n×d
    return Y
```

- 关键设计要点：
  - **继承 LayerNorm**：X 的 LayerNorm 直接从 dense checkpoint 继承，Q 也通过相同 LayerNorm + L2 norm，保证 MoE 层与 dense checkpoint 的分布一致性。
  - **超球面相似度**：Q 经 L2 normalize 后 ‖Q_norm‖ = 1，与 K 做点积等价于 cosine similarity（因 ‖K‖ 未归一化保留了 scale 信息），解决了随机初始化 Q 的数值不稳定。
  - **并行 Expert 前向**：将所有 expert 的 weight_1 合并为一个大矩阵（shape e×d2×d1），通过单次 einsum 完成 b×e×s 个 slot 的并行计算，替代传统 for-loop 逐 expert 处理。
  - **Adaptive Dual-path**：core expert 数量 = 总 expert 数的 1/3（最优比例来自消融实验），core expert 有完整 hidden dim 4d'，universal expert hidden dim ≈ d'（约 1/4 参数）。
