## Expert-Specific Learning Rate Scaling in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Learning Rate Scaling 是 Hunyuan-Large 提出的 MoE 训练策略：为 shared expert 和 specialized experts 分配不同的学习率（而非统一学习率），以解决不同 expert 处理的 token 数量不平衡导致的 effective batch size 差异问题。核心思想：MoE 中 shared expert 处理所有 token，而每个 specialized expert 仅处理约 1/n 的 token（n=specialized experts 数量），因此它们的 effective batch size 不同，需要不同的最优学习率。

基于 AdamW 的最优学习率公式（Li et al., 2024a）：

$$\epsilon_{opt}(B) = \frac{2\epsilon_{max}}{\sqrt{\frac{B_{noise}}{B} + \sqrt{\frac{B}{B_{noise}}}}}$$

其中 ε_max 是 AdamW 的最大学习率，B_noise 是训练速度与数据效率的 trade-off 点。shared expert 使用 ε_opt(B)，specialized expert 使用 ε_opt(B/n)。在 Hunyuan-Large 中（n=16, B=实际 batch size），比例 ε_opt(B)/ε_opt(B/n) ≈ 0.31。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hunyuan-Large 的 Expert-Specific LR 设置
# 全局参数: 最大学习率 ε_max, 噪声 batch size B_noise, 实际 batch size B
# specialized experts 数量 n=16

# Step 1: 计算 shared expert 最优学习率
B_shared = B                                          # shared expert 处理所有 token
shared_lr = 2*ε_max / (sqrt(B_noise/B) + sqrt(B/B_noise))

# Step 2: 计算 specialized expert 最优学习率
B_specialized = B / 16                                # 每个 specialized expert 仅 1/16 token
specialized_lr = 2*ε_max / (sqrt(B_noise/B_specialized) + sqrt(B_specialized/B_noise))

# Step 3: 分配学习率
# optimizer param groups:
#   - shared_expert.params: lr = shared_lr
#   - specialized_experts.params: lr = specialized_lr = shared_lr * 0.31 (approximately)
#   - other params (attention, embedding, etc.): lr = shared_lr
```

Hunyuan-Large 使用 AdamW optimizer。SFT 阶段的学习率从 2e-5 衰减到 2e-6（3 epochs）。Annealing 阶段在最后 5% tokens 将学习率降至 peak 的 1/10。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中通过 optimizer 的 `param_groups` 实现：为 shared expert 参数和 specialized expert 参数创建不同的 param group，设置不同的 `lr` 值。关键在于计算合理的比例因子——该因子取决于 `B_noise`（需通过小规模实验估计）和 specialized experts 数量 n。此方法适用于所有使用 shared + specialized experts 架构的 MoE 模型（如 DeepSeek-V2/V3 也可受益）。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
