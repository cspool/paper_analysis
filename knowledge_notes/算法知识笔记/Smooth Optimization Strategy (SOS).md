## Smooth Optimization Strategy (SOS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Smooth Optimization Strategy (SOS) 是 I&S-ViT 提出的 ViT PTQ 三阶段训练策略。设计动机源自不同量化粒度下 loss landscape 分析：(1) channel-wise 权重量化 + layer-wise 激活量化 → 粗糙高 loss landscape，易误导优化进入局部极小值；(2) 全精度权重 + channel-wise 激活量化 → 平滑低 loss landscape，优化稳定。SOS 先利用平滑 landscape 优化，再通过 scale reparameterization 无缝转为 layer-wise 量化保持推理效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段流程：
```
# Stage 1: 全精度权重 + 细粒度激活量化优化
for iter in range(1000):
    # weights: FP32 (不量化)
    # post-LayerNorm activations: channel-wise quantizer
    # other activations: layer-wise quantizer
    L = ||X_l_fp - X_l_hat||_2   # block-wise reconstruction
    optimizer.step()  # Adam, lr=4e-5, cosine decay

# Stage 2: 无损 Scale Reparameterization
# 将 channel-wise 量化转为 layer-wise (权重FP32, 无损)
for each post-LayerNorm activation:
    s̃ = Mean(s); z̃ = Mean(z)
    β̃ = (β + s⊙(z-z̃)) / (s/s̃); γ̃ = γ / (s/s̃)
    W̃ = (s/s̃)⊙W; b̃ = b - (s⊙(z-z̃))·W

# Stage 3: 全量化微调
for iter in range(additional_iters):
    # weights: channel-wise quantizer; all activations: layer-wise
    L = ||X_l_fp - X_l_hat||_2; optimizer.step()
```
SOS 与 BRECQ 的两阶段策略的区别：(1) SOS 先全精度权重+量化激活，BRECQ 先量化权重+全精度激活；(2) SOS 包含 ViT 特有的无损 Scale Reparameterization 转换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SOS 在 PyTorch 中通过三阶段训练循环实现。Adam 优化器，权重 lr=4e-5，cosine 衰减，WD=0。Stage 1 使用 channel-wise 量化器（每个通道独立 scale/zero-point），Stage 2 通过修改 LayerNorm affine 参数和后续层权重实现等价转换，Stage 3 切换为 layer-wise 量化器进行微调恢复权重量化损失。量化参数在初始校准后固定不优化。I&S-ViT 实验：SOS 单独贡献 +41.83% 准确率提升（DeiT-S W3A3，从 3.36% 到 45.19%）。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---
