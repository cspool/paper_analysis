## Router Z-Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Z-Loss 是 Switch Transformer (Fedus et al., 2022) 引入的一种辅助损失函数，用于稳定 MoE 训练中 router（门控网络）的 logits 输出。其数学形式为：
$$L_z = \frac{1}{B} \sum_{i=1}^{B} \left(\log \sum_{j=1}^{E} \exp(x_i \cdot W_r)_j\right)^2$$
其中 B 为 batch token 数，E 为 expert 数，x_i 为第 i 个 token 的 hidden state，W_r 为 router 权重矩阵。实质是对 router logits 的 log-sum-exp 值（即归一化前的 softmax 分母的对数）施加 L2 惩罚，鼓励 router 输出值保持较小，防止 logits 漂移过大导致训练不稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Joint MoE Scaling Laws 论文中，Router Z-Loss 的计算流程（每 MoE layer 前向）：

```
# 输入: x [B, L, d_model]
# Router: W_r [d_model, E]
# z_loss_coefficient = 0.001

# 1. Router logits 计算
router_logits = x @ W_r  # [B*L, E]

# 2. Z-Loss 计算
log_z = log(sum(exp(router_logits), dim=-1))  # [B*L], softmax 分母的对数
z_loss = (1 / (B*L)) * sum(log_z ** 2)         # scalar

# 3. 总 loss = cross_entropy + load_balancing_loss + z_loss_coefficient * z_loss
```

Z-Loss 与 Load Balancing Loss 的区别：Load Balancing Loss 惩罚的是 expert 间 token 分配的不均衡，Z-Loss 惩罚的是 router logits 幅值的过大增长。两者协同：Load Balancing 确保 token 均匀分布，Z-Loss 确保 router 数值稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Z-Loss 在 MoE 训练中的典型使用：
- Switch Transformer 原论文：z_loss_coefficient = 0.01（推荐值）
- Joint MoE Scaling Laws：z_loss_coefficient = 0.001（更保守，配合 truncated normal initialization scale=0.1）
- ST-MoE (Zoph et al., 2022)：进一步提出 router z-loss 的变体和改进
- 通常与 Load Balancing Loss 共同作为辅助损失，系数通过网格搜索确定
- 对训练稳定性的影响：无 z-loss 可能导致 router logits 发散 → softmax 退化为 one-hot → 训练崩溃

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
