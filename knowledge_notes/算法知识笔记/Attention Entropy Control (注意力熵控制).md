## Attention Entropy Control (注意力熵控制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Entropy Control 是一类通过控制 softmax 后注意力分布熵来实现长上下文泛化的技术族。核心问题：从训练长度 $N_{\text{train}}$ 扩展到推理长度 $N_{\text{inf}} \gg N_{\text{train}}$ 时，标准 attention 分布趋于均匀（高熵），注意力浪费在无关 token 上。

Token 范围 $[t_1, t_2]$ 内的熵定义为：
$$H_{t_1}^{t_2} = -\sum_{t=t_1}^{t_2-1} \frac{\tilde{A}_t}{Z_{t_1}^{t_2}} \log\left(\frac{\tilde{A}_t}{Z_{t_1}^{t_2}}\right)$$

现有方法分类：
1. **全局缩放**（LogN）：$L_t = s\log N \cdot S_t$ —— 位置无关
2. **加性偏置**（ALiBi）：$L_t = S_t - m \cdot t$ —— 线性刚性
3. **位置依赖变换**（Scale-invariant）：$L_t = a_t S_t + m_t$ —— 局部恒等、全局稀疏

从算法pipeline角度拆解术语：

三种方法的 entropy-scaling 行为比较（IID Gaussian logits）：

| 方法 | $H_t^{t\Delta}$ | 局部注意力保持 | 全局稀疏性 |
|------|----------------|--------------|-----------|
| 无缩放 | $\Theta(\log t)$ | 差 | 无 |
| LogN | sub-log | 差 (随 $N$ 衰减) | 强 |
| Scale-invariant | $\sim \sqrt{\log t}$ | 好 | 弱 (sub-linear) |

术语一般如何实现？如何使用？

通过 attention score modification 实现（FlexAttention score_mod）。选择指南：LogN 适合不需要精细局部控制的场景（实现最简单）；ALiBi 适合极端长度外推（零额外参数）；Scale-invariant 适合需同时保持局部稠密和全局稀疏的场景（仅增一个超参数 $\tau$）。

涉及论文标题：
- Scale-invariant Attention
