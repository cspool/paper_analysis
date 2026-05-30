## Expert Regularization（专家正则化，MoE 上下文）

术语是什么？
在 MoE Jetpack 的 SpheroMoE Layer 中，Expert Regularization 是一组防止 MoE expert 在 fine-tuning 过程中过度特化（over-specialization）和防止输出过度依赖单一 expert 的正则化技术组合，包括：(1) Learnable Softmax Temperature T：初期 T 大→logits 平滑→expert 均匀分散注意力；逐步减小 T→expert 聚焦特定特征；(2) Gaussian Noise：加在相似度 logits S 上，提升泛化能力；(3) Stochastic Expert Dropout：以概率 p 随机停用 expert，防止任一 expert 成为输出瓶颈。

从算法pipeline角度拆解术语：
```
# 在 SpheroMoE 前向传播的相似度计算后应用
S = einsum(K, Q_norm, "b n d, e s d -> b n e s")     # 原始相似度 logits

# 1. Gaussian Noise
noise = torch.randn_like(S) * noise_multiplier
S = S + noise

# 2. Learnable Temperature（训练过程中动态变化）
# T 初始化为较大值（如 5.0），随训练 epoch 逐步减小
dispatch = softmax(S / T, dim=1)                      # T 大→均匀分布，T 小→尖锐分布

# 3. Stochastic Expert Dropout
mask = torch.bernoulli(torch.ones(e) * (1 - p))      # 每个 expert 以 p 概率被 drop
Y_hat = Y_hat * mask.view(1, e, 1, 1)                # 被 drop 的 expert 输出归零
```

术语一般如何实现？如何使用？
Temperature T 实现为 nn.Parameter，随模型一同训练，初期设为较大值（5.0 或更高），训练中通过标准梯度下降自动调节。Noise multiplier 可设为小值（0.01-0.1），dropout rate p 类似标准 dropout 设为 0.1-0.2。三者组合使用，共同确保 fine-tuning 稳定性。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
