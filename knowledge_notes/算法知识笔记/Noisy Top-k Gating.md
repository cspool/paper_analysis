## Noisy Top-k Gating

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Noisy Top-k Gating 是 MoE 模型中 router（门控网络）的核心机制，通过注入可学习的 Gaussian 噪声到 router logits 中，再从 N 个 expert 中选 Top-K 个进行激活。数学形式（Shazeer et al., ICLR 2017）：

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

其中 W_g 是 gating 权重矩阵，W_noise 是可训练的噪声权重矩阵。StandardNormal() 注入随机性，Softplus 保证噪声非负。噪声的作用：(1) 鼓励探索——防止 token 始终走相同路径；(2) 辅助负载均衡——打破"富者愈富"的正反馈循环；(3) 防止 expert collapse——少数 expert 支配训练而其余退化。

KeepTopK(v, k)_i = v_i if v_i in top-k else -∞。Softmax 将 -∞ 映射到 0，实现真正的稀疏激活——仅 K 个 expert 有非零权重，其余完全不被计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Noisy Top-k Gating 在 MoE 层 forward 中的流程
def moe_router_forward(x, W_g, W_noise, N, K):
    # Step 1: 计算 clean logits
    clean_logits = x @ W_g  # [B*S, N]

    # Step 2: 注入可学习噪声
    noise_std = softplus(x @ W_noise)  # [B*S, N]，保证非负
    noise = torch.randn_like(clean_logits) * noise_std
    noisy_logits = clean_logits + noise

    # Step 3: KeepTopK → 仅保留 Top-K 的 logits，其余设为 -inf
    topk_vals, topk_indices = torch.topk(noisy_logits, K, dim=-1)
    mask = torch.full_like(noisy_logits, float('-inf'))
    mask.scatter_(-1, topk_indices, topk_vals)

    # Step 4: Softmax （在 Mixtral-type 中 KeepTopK 在前）
    gate_weights = F.softmax(mask, dim=-1)  # 仅 K 个非零

    return gate_weights, topk_indices
```

Load balancing 辅助损失（GShard 风格）：
$$L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 f_i = fraction of tokens dispatched to expert i, P_i = average gating probability for expert i, α 为损失系数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

现代框架实现（Megatron-Core）支持：
- switch_load_balancing_loss_func：GShard 风格辅助损失
- Sinkhorn routing：基于最优传输的负载均衡
- Group-limited top-k routing：限制 expert 选择范围在 device/node 子集内（减少 All-to-All 通信）
- Mixtral-type (KeepTopK→Softmax) 和 ST-type (Softmax→KeepTopK) 两种路由顺序。Upcycling 场景下 Mixtral-type 更优，因为初始输出与 dense 模型一致

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling
