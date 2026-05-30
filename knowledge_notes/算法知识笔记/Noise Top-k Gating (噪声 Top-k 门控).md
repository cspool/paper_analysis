## Noise Top-k Gating (噪声 Top-k 门控)

术语解释
Noise Top-k Gating 是 MoE router 的一种门控机制，由 Shazeer et al. (2017) 提出。在标准 Top-k 选择前向 gate logits 加入可训练的 Gaussian noise，使得负载均衡通过噪声扰动自然实现而非完全依赖 auxiliary loss。

术语是什么？
公式定义：
$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$
$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$
$$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$

核心组成：
- **Gate logits**: `x · W_g` 产生每个 expert 的基本得分
- **Noise term**: `StandardNormal() · Softplus(x · W_noise)` 对 logits 加入可训练的高斯噪声，`W_noise` 可学习噪声幅度
- **KeepTopK**: 保留 top-k logits，其余设为 `-∞`（softmax 后概率为 0）
- **Softmax + Weighted Sum**: 归一化后加权聚合 expert 输出

噪声作用：(1) 在训练中通过随机扰动打破固定 routing 模式，鼓励更多样化的 expert 选择；(2) 配合 auxiliary loss 实现负载均衡；(3) 使 gate probability 分布趋于平滑，减少 router 坍缩到少数 expert 的风险。

从算法pipeline角度拆解术语：
```
# Noise Top-k Gating Forward Pass (per token)
def noise_topk_gating(x, W_g, W_noise, k, N):
    # x: [d] input token embedding
    # W_g: [d, N] gate weight
    # W_noise: [d, N] noise weight
    
    # Step 1: Clean logits
    clean_logits = x @ W_g              # [N]
    
    # Step 2: Noise logits (trainable)
    noise_std = softplus(x @ W_noise)   # [N], always positive
    noise = randn(N) * noise_std        # [N], Gaussian noise
    H = clean_logits + noise            # [N], noisy logits
    
    # Step 3: Top-k selection
    topk_vals, topk_idx = topk(H, k)    # select k experts
    mask = -inf * ones(N)
    mask[topk_idx] = H[topk_idx]
    
    # Step 4: Softmax + aggregate
    G = softmax(mask)                   # [N], sparse gate probs
    y = sum(G[i] * E_i(x) for i where G[i] > 0)
    return y
```
训练时 `W_g` 和 `W_noise` 均可学习；推理时噪声可关闭以确定性选择 expert。

术语一般如何实现？如何使用？
- Llama-MoE (Zhu et al., 2024) 使用 Noise Top-k Gating 作为其 router 机制
- 噪声项通过 `Softplus` 保证标准差始终为正，允许模型学习每个 expert 的噪声幅度
- 在 KD 场景中，SAR 方法更新 `W_g` 和 `W_noise`（仅 router 部分）使其更适应 student 的学习需求
- 负载均衡：Noise Top-k 配合 auxiliary loss `L_b = CV(m)^2 + CV(P)^2` 实现 balanced routing

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
