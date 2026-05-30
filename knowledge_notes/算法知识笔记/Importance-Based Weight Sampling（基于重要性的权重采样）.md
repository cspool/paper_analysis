## Importance-Based Weight Sampling（基于重要性的权重采样）

术语是什么？
Importance-Based Weight Sampling 是 Checkpoint Recycling 的默认权重选择策略。通过在前驱 dense 模型上跑一批图像获取每层各 channel 和 neuron 的 activation 值，然后按 activation 大小选择最重要的 channel（确定性 top-d'）和按 activation 概率分布采样 neuron（随机性保证 expert 多样性），从 dense checkpoint 中提取对应权重子矩阵构造 MoE expert。

从算法pipeline角度拆解术语：

```
# Channel 选择：跨层平均 activation，确定性取 top-d'
# Activation 收集：在 ImageNet 训练集 30K 子集上推理
for each batch B in calibration_set:
    for each layer l in [0..N-1]:
        A_channel[l] += activation_of_channels(B)  # (d,)
A_avg = mean(A_channel, dim=layer)                  # (d,) 跨层平均
top_channels = argsort(A_avg, descending=True)[:d']  # 选最重要的 d' 个 channel

# Hidden Neuron 选择：按 activation 概率分布独立采样每个 expert
for each layer l:
    A_neuron = activation_of_neurons[l]              # (4d,)
    prob = A_neuron / sum(A_neuron)                  # 概率分布
    for each expert e:
        neurons[e] = sample(prob, size=4d')          # 独立采样保证 diversity
```

术语一般如何实现？如何使用？
在 PyTorch 中通过 forward hook 在 predecessor 模型的每层 MLP 后注册 hook 捕获 activation 值。activation profilling 使用校准集（如 ImageNet 子集）进行一次前向传播。选出的 channel/neuron index 用于从 state_dict 中提取对应权重子矩阵构造 expert。作为 Checkpoint Recycling 默认策略（消融实验表明在所有策略中效果最好），替代 Sparse Upcycling 的 naive 复制。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
