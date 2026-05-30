## Upcycling (Sparse Upcycling / 稠密到MoE转换)

术语是什么？

Upcycling（在 MoE 语境中）是将已预训练的稠密（dense）语言模型转换为稀疏 MoE（Mixture of Experts）模型的训练技术。核心流程是：(1) 复制稠密模型的 MLP（FFN）权重来初始化 MoE 的多个 expert；(2) 随机初始化 Router；(3) 用相对较少的 token（通常是预训练 token 数的 10% 以内）继续训练，使 Router 学会合理的路由策略、expert 逐步分化。Upcycling 的目标是：与从头训练 MoE 相比，大幅降低总计算量；与续训稠密模型相比，利用 MoE 架构获得更高模型容量，从而得到更好的下游任务性能。

其理论基础在于：预训练的稠密 MLP 层已经学到了丰富的通用知识（language understanding、reasoning 等），将这些知识作为多个 expert 的初始状态比随机初始化更高效。每个 expert 从相同的起点出发但因为在 upcycling 阶段收到的 token 不同，逐渐通过梯度更新发生分化，最终形成不同专长的专家。

从算法pipeline角度拆解：

Upcycling 的完整流程（以 Nemotron-4 15B → E8G1T2 为例）：

```
# === 阶段 0: 准备稠密 Checkpoint ===
dense_model = load_checkpoint("Nemotron-4-15B")
# dense_model 已预训练 8T tokens, MMLU 59.3

# === 阶段 1: 初始化 MoE 架构 ===
# 对每个需要替换为 MoE 的 Transformer 层 (每 2 层中的 1 层):
for layer in moe_layers:
    # 复制 MLP 权重 E 次 → 初始化为 E 个 expert
    layer.experts = [copy(dense_model[layer].mlp) for _ in range(E)]
    # 随机初始化 Router
    layer.router = random_init((d_model, E))
    # 应用 Weight Scaling
    scale = (E * G**2 / T) ** (1/3)
    for expert in layer.experts:
        expert.W1 *= scale
        expert.W2 *= scale

# === 阶段 2: Upcycling 训练 ===
# 训练数据: 续训数据 blend，1T tokens
# 学习率: warmup → peak 3e-4 → cosine decay → 1/100 of pretraining min LR
# Load balancing aux loss: coeff = 1e-2
# 分布式策略: Megatron-LM (DP + TP + EP)
for batch in training_data:
    # 标准 MoE forward pass
    output = moe_model(batch)
    loss = L_LM + 0.01 * L_aux  # 语言模型损失 + 负载均衡损失
    loss.backward()
    optimizer.step()
```

关键设计决策：
1. **学习率重置**：upcycling 必须使用高学习率（如 2e-4 或 3e-4），而非 fine-tuning 的小学习率。原因是 MoE 从稠密模型的局部最优出发，高学习率帮助逃离该局部最优，促进 expert 分化。
2. **Router 设计**：推荐 softmax-then-topK 而非 topK-then-softmax，因为前者保留了 Router 输出的绝对值信息。
3. **大批量**：推荐 4M+ tokens 的 batch size，因为每个 expert 只收到部分 tokens，大量样本能稳定梯度并降低负载均衡损失的噪声。
4. **Weight Scaling**：对 fine-grained MoE 至关重要，补偿因 expert 拆分导致的输出缩放。

术语一般如何实现？

NVIDIA 开源实现位于 Megatron-LM: [moe/upcycling](https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling)。NeMo 也集成了 online upcycling 功能（用户提供 dense checkpoint + parallel training config，自动产生 MoE 模型）。后续工作 "Llama 3 Meets MoE" 将 upcycling 应用于 Llama 3-8B，使用 <1% 预训练计算量实现 ~1.2% benchmark 平均提升。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
