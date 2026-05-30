## Load Balancing Auxiliary Loss (负载均衡辅助损失)

术语是什么？

Load Balancing Auxiliary Loss（负载均衡辅助损失）是 MoE 训练中用于防止"专家坍塌"（expert collapse，即大部分 token 被路由到极少数 expert 而其他 expert 闲置）的辅助损失函数。最早由 Switch Transformer (Fedus et al., 2022) 和 ST-MoE (Zoph et al., 2022) 提出。其核心思想是：在语言模型 loss 之外添加一项鼓励均匀路由的惩罚项，使 Router 在优化语言模型的同时也更均匀地分配 token。

标准公式（本文使用）：
$$L_{\text{aux}} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

其中：
- $f_e = \frac{1}{T} \sum_{t \in \text{batch}} \mathbf{1}[\text{token } t \text{ routed to expert } e]$（expert e 实际收到的 token 比例）
- $P_e = \frac{1}{T} \sum_{t \in \text{batch}} \text{softmax\_prob}[t, e]$（Router 分配给 expert e 的 softmax 概率均值）
- $E$ 为 expert 总数

当所有 expert 被均等利用时 $f_e = P_e = 1/E$，$L_{\text{aux}} = 1$。$L_{\text{aux}}$ 越大表示负载越不均。

从算法pipeline角度拆解：

```
# 在每个训练 step 的 forward pass 中:
gate_probs = softmax(x @ W_router)     # (S, E), Router softmax 输出
topk_probs, topk_idx = topk(gate_probs, T)  # Top-T 选择

# 计算 f_e (实际负载):
f_e = zeros(E)
for t in range(S):
    for k in range(T):
        f_e[topk_idx[t, k]] += 1 / (S * T)

# 计算 P_e (Router 分配比例):
P_e = gate_probs.mean(dim=0)           # (E,)

# 辅助损失:
L_aux = E * sum(f_e * P_e)             # 标量

# 总损失:
L_total = L_LM + alpha * L_aux
# 本文: alpha = 1e-2, 不带 Z loss
```

关键 hyperparameter 权衡：
- alpha 太小 (如 1e-4)：不充分的负载均衡 → 出现 "dead experts"（某些 expert 永远未被路由），导致训练 loss 提早 plateau
- alpha 太大 (如 1e-1)：aux loss 主导 language modeling loss → 模型质量下降
- 本文推荐范围：1e-2 到 1e-3

术语一般如何实现？

在 Megatron-LM / NeMo 等训练框架中，aux loss 在 MoE 层的 forward pass 中计算并加到 total loss 中。训练时与主 loss 同步反向传播。Upcycling 场景下特别重要，因为初始 Router 是随机初始化的，没有 aux loss 会导致少数 expert 迅速接收大部分 token 而其他 expert 完全不被训练。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
