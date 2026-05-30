## Scale-invariant Attention (尺度不变注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Scale-invariant Attention 是一种对 Transformer attention logits 施加位置依赖变换的注意力机制，由 Anson et al. (2025) 提出。核心思想源于自然图像的尺度不变（scale-invariant）统计特性——图像中存在所有空间尺度的重要特征。类比到文本 attention，token 范围也划分为不同尺度（1-10 tokens、10-100 tokens、100-1000 tokens 等），每个尺度的信息都应被保留。

Scale-invariant Attention 满足两个数学性质：
1. **Scale-invariant Total Attention**：在任意 token 范围 $[t, t\Delta)$ 内的 expected total unnormalized attention $\mathbb{E}[Z_t^{t\Delta}] = \Theta(1)$，即各范围的总注意力渐进恒定。
2. **Weak Scale-invariant Attention Sparsity**：$\mathbb{E}[H_t^{t\Delta}] = o(\log t)$，即注意力熵随 $t$ 亚对数增长，稀疏性随上下文变长而增加。

实现方式：在 attention score $S_t$ 上施加位置依赖变换 $L_t = a_t \cdot S_t + m_t$，其中（在 IID Gaussian logits 假设下，边界条件 $a_0^2=1, m_0=0$ 得 $\alpha=\beta=e^{0.5}$）：
$$a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}, \quad m_t = -a_t^2 + \beta/\alpha$$

唯一超参数 $\tau$（长度尺度，最优约 10）控制"局部区域"大小。$t \ll \tau$ 时 $a_t \approx 1, m_t \approx 0$（局部近似标准 attention）；$t \gg \tau$ 时 $a_t^2$ 对数增长（分布尖锐化），$m_t$ 对数下降（压低远距离总权重）。

从算法pipeline角度拆解术语，给出具体例子。

**Scale-invariant attention forward pass（使用 FlexAttention）**：

```
输入: Q [B, H, T, d], K [B, H, T, d], V [B, H, T, d]
超参数: τ = 10, α = β = e^{0.5}

# score_mod 函数（FlexAttention）
def scale_invariant_score_mod(score, b, h, q_idx, kv_idx):
    t = q_idx - kv_idx  # 距离
    if t >= 0:
        f_t = log(t/τ + 1) - log(α)
        a_t = sqrt(2 * (f_t + β/α))
        m_t = -a_t**2 + β/α
        return a_t * score + m_t
    return score

# FlexAttention 自动处理 causal mask、block-sparse 编译、反向传播
output = flex_attention(Q, K, V, score_mod=scale_invariant_score_mod)
```

**与 LogN/ALiBi 的关键区别**：
- LogN: $L_t = s\log N \cdot S_t$（位置无关，全局缩放）→ 牺牲局部注意力
- ALiBi: $L_t = S_t - m \cdot t$（线性刚性偏置）→ 无法灵活控制熵
- Scale-invariant: $L_t = a_t S_t + m_t$（位置依赖）→ 局部稠密 + 全局稀疏

术语一般如何实现？如何使用？

基于 modded-nanogpt（PyTorch），使用 FlexAttention API 定义 score_mod。论文在 GPT-2-style 162M/304M 和 Llama 2 7B 上验证。训练：FineWeb 数据集，Muon（线性层）+ Adam（embedding）。评估：验证 loss + needle-in-a-haystack。性能（162M Train@4k/Val@64k）：Val loss=3.247 vs LogN+RoPE 3.378 vs ALiBi 3.270，Needle@64k 准确率 0.969。

涉及论文标题：
- Scale-invariant Attention

---
