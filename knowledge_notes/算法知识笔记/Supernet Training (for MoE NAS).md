## Supernet Training (for MoE NAS)

术语解释
Supernet Training 是 Neural Architecture Search (NAS) 中的核心技术，通过构建一个包含搜索空间中所有可能子架构的"超级网络"（Supernet），并通过权重共享（weight sharing）联合训练所有子架构。AutoMoE 首次将 Supernet training 扩展到 MoE 架构的搜索空间。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Supernet = 搜索空间中的最大模型配置。在 AutoMoE 中，Supernet 是：
- 每层 M 个 expert（M = 最大 expert 数）
- 每个 expert 的 FFN 中间维度 = 最大可选值（3072）
- Decoder 层数 = 最大值（6）
- Attention heads = 最大值（8）
- Hidden/Embedding size = 最大值（640）

Supernet 训练过程：
1. 随机从搜索空间采样一个子架构（subnet）
2. 从 Supernet 中提取对应子架构的权重（通过 weight sharing 机制）
3. 用提取的权重前向传播 + 反向传播（仅更新被提取的部分）
4. 重复采样和训练直至训练 budget 耗尽

训练收敛后，任何子架构的性能可通过从 Supernet 提取权重并在验证集上评估来快速估计，无需单独训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Supernet 构建（搜索空间中的最大配置）
Supernet = Transformer(
    encoder_layers=6, decoder_layers=6,
    embedding_size=640,
    experts_per_layer=M,           # 每层 M 个 expert
    expert_ffn_size=3072,          # 最大 FFN 中间维度
    attention_heads=8
)

# Supernet 训练循环
for step in range(total_steps):
    # Step 1: 随机采样一个子架构
    subnet_config = {
        'dec_layers': random_choice([1,2,3,4,5,6]),
        'emb_size': random_choice([512, 640]),
        'attn_heads': random_choice([4, 8]),
        'enc_experts': [random_choice(1..M) for _ in range(6)],
        'dec_experts': [random_choice(1..M) for _ in subnet_config['dec_layers']],
        'enc_ffn_sizes': [[random_choice([1024,2048,3072]) for _ in range(e)] 
                          for e in subnet_config['enc_experts']],
        'dec_ffn_sizes': [[random_choice([1024,2048,3072]) for _ in range(e)] 
                          for e in subnet_config['dec_experts']]
    }
    
    # Step 2: 从 Supernet 提取子架构权重（front rows/columns）
    subnet_weights = extract_subnet_weights(Supernet, subnet_config)
    
    # Step 3: 前向 + 反向传播
    loss = forward(subnet_weights, batch)
    loss.backward()
    optimizer.step()  # 仅更新被提取的权重部分

# 训练后：评估任意子架构（无需额外训练）
def evaluate_subnet(config):
    weights = extract_subnet_weights(Supernet, config)
    val_loss = forward(weights, val_set)
    return val_loss
```

术语一般如何实现？如何使用？
- 基于 fairseq toolkit 实现（AutoMoE）
- Supernet 训练 40K steps，与最终模型训练相同步数（fair comparison）
- Weight sharing 使搜索效率极高：单个 Supernet 涵盖数千个子架构，搜索 + 训练仅需 224 GPU-hours（vs Evolved Transformer 的 2,192,000 GPU-hours）
- 局限：Supernet 中所有子架构共享权重可能导致子架构性能估计有偏；Sandwich sampling 和 inplace knowledge distillation 可改进 Supernet 训练质量
- 后续工作：Mixture-of-Supernets (MoS, ACL 2024 Findings) 用 MoE 增强 Supernet 表达力

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---
