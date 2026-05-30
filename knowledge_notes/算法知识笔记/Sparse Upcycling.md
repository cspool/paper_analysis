## Sparse Upcycling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Upcycling 是一种将预训练 dense checkpoint 转换为稀疏激活 MoE (Mixture-of-Experts) 模型的训练初始化技术。由 Komatsuzaki et al. (ICLR 2023) 首次提出，核心思想是：将 dense 模型中某些 FFN (Feed-Forward Network) 层的权重复制 N 次，初始化 N 个 expert，同时添加一个随机初始化的 router（门控网络），其余层（embedding、attention、norm 等）直接从 dense checkpoint 复制。随后仅需少量继续训练（<1% 原始预训练 compute），router 学会将不同 token 路由到不同 expert 组合，expert 在 fine-tuning 中逐渐分化。该方法避免从头训练 MoE 的高昂成本（数据需求大、训练不稳定、expert collapse 等），同时复用已投入的预训练 GPU 小时。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Sparse Upcycling 的算法 pipeline（以 Llama 3-8B → E8T2 为例）：

```
# 输入: dense checkpoint Θ_dense (Llama 3-8B), N=8 experts, K=2 (Top-K)
# 输出: initialized MoE checkpoint Θ_moe

def sparse_upcycling(Θ_dense, N, K, moe_layer_indices):
    Θ_moe = deep_copy(Θ_dense)  # 复制所有非 MoE 权重

    for layer_idx in moe_layer_indices:  # 每隔一层替换 FFN
        W_ffn = Θ_dense[layer_idx].ffn    # 原始 FFN 权重

        # Step 1: 复制 FFN N 次初始化每个 expert
        for i in range(N):
            Θ_moe[layer_idx].expert[i] = deep_copy(W_ffn)

        # Step 2: 随机初始化 router
        Θ_moe[layer_idx].router.W_g = random_init()      # gating weights
        Θ_moe[layer_idx].router.W_noise = random_init()   # noise weights

    return Θ_moe
```

Upcycling 后继续训练的 MoE 前向传播（Mixtral-type router）：

```
# Router: KeepTopK → Softmax (确保初始输出 = dense 输出)
H(x) = x @ W_g + StandardNormal() * Softplus(x @ W_noise)
G(x) = Softmax(KeepTopK(H(x), k=K))

# Expert FFN (SiLU-gated):
for token x routed to experts (i1, i2):
    gate_i1 = G(x)[i1], gate_i2 = G(x)[i2]
    y = gate_i1 * E_i1(x) + gate_i2 * E_i2(x)

# Expert Capacity 硬约束:
capacity = (tokens_per_batch / N) * CF
# 溢出 token 跳过该 MoE 层，直接传递 residual
```

Router 类型选择：Mixtral-type (KeepTopK→Softmax) 优于 ST-type (Softmax→KeepTopK)，因为 upcycling 后初始前向输出与 dense 模型完全一致，训练初始 loss 更低、收敛更快。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式（基于 NVIDIA NeMo + Megatron-Core）：
1. Online Upcycling：按并行训练配置（TP/EP/PP/DP）分片 dense checkpoint，各设备独立完成权重复制和 router 初始化，无需跨设备通信
2. 训练配置：CF=4, EP=8, TP=2, PP=4, VPP=8, DP with ZeRO-1, bfloat16 精度
3. 学习率调度：初始 LR=3e-5，余弦退火至 3e-7，100 warmup steps
4. 仅需 100B tokens 训练（<1% 预训练 compute），512 H100 GPU 上消耗 11K GPU hours
5. 适用场景：已有预训练 dense checkpoint，希望在有限 compute budget 下提升模型性能（如 MMLU 0-shot +2%）

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling
