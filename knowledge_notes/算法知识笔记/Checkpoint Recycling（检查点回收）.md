## Checkpoint Recycling（检查点回收）

术语是什么？
Checkpoint Recycling 是 MoE Jetpack 框架的核心组件，一种将预训练 Dense 模型（predecessor）的权重直接转化为 MoE 模型（successor）中各 Expert 初始权重的方法。它不复制整个 MLP，而是从前驱模型的 MLP 权重中通过重要性采样等策略选择性地提取 weight 子集来构造 expert，从而将 dense checkpoint 的预训练知识注入 MoE 模型，避免从零训练 MoE 的高昂开销。该过程为一次性离线操作（在 RTX 4090 上对 30000 张图像进行 activation profiling 仅需约 5 分钟）。

从算法pipeline角度拆解术语：
给定 predecessor dense 模型 P（N 层、channel dim d、hidden dim 4d），目标 successor MoE 模型 S（N 层、channel dim d' ≤ d、前半 N/2 dense layers + 后半 N/2 SpheroMoE layers）。Importance-Based Weight Sampling（默认策略）流程：

```
# Step 1: Activation Profiling
images = sample_batch(dataset)        # 一批图像通过 predecessor
for layer l in P:
    A_c[l] = activation_of_channel(c)  # 每层每个 channel 的 activation
    A_h[l] = activation_of_neuron(h)   # 每层每个 hidden neuron 的 activation

# Step 2: Channel Selection（跨层平均，选 top-d'）
for channel c in [0..d-1]:
    A_c = mean([A_c[l] for l in range(N)])
selected_channels = top_k(A_c, d')     # activation 最高的 d' 个 channel

# Step 3: Neuron Sampling（按 activation 概率分布为每个 expert 采样）
P(h|H) = A_h / sum(all_A_h)           # activation → 概率分布
for each expert e:
    expert_neurons[e] = sample(P(h|H), 4d')  # 采样不同 neuron 保证 expert 多样性

# Step 4: Weight Extraction
for each expert e:
    W_expert[e] = W_predecessor[selected_channels][expert_neurons[e]]
```

其他策略：(a) Co-Activation Graph Partitioning：构造 neuron 共激活图，用 Metis 图分割将频繁共激活的 neuron 分入同一 expert；(b) Uniform Selection：等距采样；(c) Random Sampling：随机选。

术语一般如何实现？如何使用？
在 PyTorch 中实现：加载 timm 预训练 dense checkpoint（如 ViT-S/16 ImageNet-21k），通过 forward hook 捕获中间层 activation 值，按策略索引提取权重子矩阵构造 expert 的 Linear 层权重，保存为 MoE checkpoint。使用时作为 MoE 模型初始权重加载，然后执行标准 fine-tuning。开源实现见 https://github.com/Adlith/MoE-Jetpack。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
