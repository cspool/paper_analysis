## Router Entropy Loss in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Entropy Loss 是 MoE 中用于控制路由稀疏性的辅助损失函数，定义为 gate 概率分布的负熵：

$$L_{\text{entropy}} = N \sum_{i=1}^{N} P_i \cdot \log(P_i)$$

其中 N 为 expert 总数，$P_i$ 为 router 对每个 token 分配到 expert i 的平均 softmax 概率。该 loss 惩罚 router 输出过于均匀的概率分布（即接近 uniform 的高熵状态），从而**抑制**模型激活过多 expert。在 HMoE 的 Top-P routing 场景中，router entropy loss 尤为重要——Top-P routing 允许动态激活任意数量的 expert，训练中可能逐步增加激活数量（router 倾向于输出更均匀的概率以"安全地"激活更多 expert），导致稀疏性退化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# Router Entropy Loss 计算
def router_entropy_loss(gate_probs, N):
    """
    gate_probs: [B, S, N] router 输出的 softmax 概率（未做 top-k/top-p masking）
    """
    # P_i: 各 expert 的平均路由概率
    P = gate_probs.mean(dim=(0, 1))  # [N]

    # L_entropy = N * Σ P_i * log(P_i)
    # 当 P_i 接近 uniform (1/N) 时 loss 最大 → 不期望
    # 当 P_i 接近 one-hot (某个 expert 概率近 1，其余近 0) 时 loss 最小 → 期望
    loss = N * (P * P.log()).sum()
    return loss
```

HMoE 的最终 training loss（Top-P routing 时）：
```
L_final = L_lm + α * L_P-Penalty + β * L_entropy
# α = 0.1 (P-Penalty coefficient)
# β = 3e-2 (Entropy loss coefficient)
```

Router entropy loss 的效果：(1) 防止 Top-P routing 在训练中激活 expert 数量无限制增长（entropy 高时 router 对 expert 的区分度低，Top-P 可能需累加更多 expert 才达到 threshold p）；(2) 鼓励 router 对少数 expert 给出高置信度（低熵），使 Top-P 更准确地按需选择真正相关的 expert。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 HMoE 中实现为每个 MoE 层 forward 后对 router softmax 输出（未经 top-k/top-p masking 的原始概率分布）计算 entropy loss，累加到总 loss。仅用于 Top-P routing（Top-K 固定激活数量，无需 entropy 控制）。系数 β 需权衡——太小则无法抑制 expert 数量增长，太大则 router 过度集中于少数 expert 导致负载不均衡。HMoE 使用 β=3e-2。首次由 Huang et al. (2024) 在 Top-P routing MoE 论文中提出。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
