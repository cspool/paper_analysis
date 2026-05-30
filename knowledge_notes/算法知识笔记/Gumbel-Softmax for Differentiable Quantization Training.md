## Gumbel-Softmax for Differentiable Quantization Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Gumbel-Softmax（Jang et al., 2016）是一种将离散分类采样重参数化为可微操作的技术。给定类别概率 $\pi_1, \dots, \pi_k$，标准 Softmax 输出概率分布，Gumbel-Softmax 通过引入 Gumbel 噪声 $g_i \sim \text{Gumbel}(0,1)$ 并使用温度参数 $\tau$ 控制输出的连续松弛：$y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)}$。当 $\tau \to 0$ 时，$y$ 趋近 one-hot 向量；当 $\tau$ 较大时，$y$ 是平滑的连续向量。这使得离散量化过程可端到端梯度优化。

在 CommVQ 中，Gumbel-Softmax 用于编码器输出层，使编码器将连续的 KV 向量映射到离散的二进制序列 s_i ∈ {0,1}^{N_c} 的过程保持可微。训练时 τ 较大（平滑梯度），推理时 τ → 0（硬量化）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 中 Gumbel-Softmax 编码器训练流程**：

```
encoder = Sequential(
    Linear(d, hidden_dim),
    ReLU(),
    Linear(hidden_dim, N_c),
    GumbelSoftmax(tau=1.0, hard=False)  // 训练模式：连续松弛
)

for t_i in kv_cache_batch:  // t_i: [d]
    s_i = encoder(t_i)      // s_i: [N_c], 连续值 ∈ (0,1)
    t_hat_i = s_i @ C       // C: [N_c, d] 码本
    loss = MSE(t_i, t_hat_i)
    loss.backward()         // 同时更新 encoder 参数和 codebook C

// 推理阶段：切换到 hard mode
encoder[-1].hard = True     // τ → 0, 输出近 one-hot
s_i = encoder(t_i)          // s_i: [N_c], 精确 {0,1}
store(s_i)                  // 每维 1 bit
```

术语一般如何实现？如何使用？

PyTorch 实现：`F.gumbel_softmax(logits, tau=1.0, hard=False)`。训练时通常使用温度退火（temperature annealing）：初始 τ 较高（1.0-5.0）以保证梯度平滑流动，随训练逐步降低 τ 使输出接近离散。在 VQ-VAE、DALL-E（码本学习）、CommVQ（KV cache 编码器）中广泛使用。CommVQ 的编码器为每层每 token 独立运行，prefill 阶段一次性编码，不增加 decoding 阶段开销。

Dynamic-LLaVA 将 Gumbel-Softmax + STE 用于 token pruning：两个 predictor（Image/Output predictor）各自输出一个 [N, 2] 维的决策矩阵 D，沿第二维做 Gumbel-Softmax 得到 D†（连续松弛），forward 时 argmax 生成离散 mask M，backward 时 STE 将 ∂L/∂M 直接传递给 ∂L/∂D†，绕过 argmax 的不可微问题。τ 从 1 指数衰减至 0.1。与 CommVQ 的差异：CommVQ 用于连续向量→离散码本映射（量化），Dynamic-LLaVA 用于 token keep/discard 二分类决策（剪枝）。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification for Codebook Learning (码本学习的EM算法)
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

**Elastic Attention 中的 Gumbel-Sigmoid for Attention Routing**：不同于 CommVQ 的 Gumbel-Softmax（多类别码本映射）和 Dynamic-LLaVA 的 Gumbel-Softmax（token keep/discard 二分类），Elastic Attention 使用 Gumbel-Sigmoid（二分类 Gumbel-Softmax）对每个 head 做 FA vs SA 二选一路由。具体：z ∈ R^{H×2} 经 Gumbel-Sigmoid 得 r_soft ∈ R^{H×2}（连续松弛），argmax 得 r_hard（离散路由），STE 传导梯度。温度 τ 按 τ(t) = max(τ_min, τ_init · exp(-r·p)) 退火（r=0.6）。早期高 τ 鼓励探索，后期低 τ 逼近离散 Bernoulli。与 Lagrange 乘子协同优化。
