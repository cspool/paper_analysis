## Hybrid-Head Architecture (混合头架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid-Head Architecture 是 Hymba (NVIDIA, 2024) 提出的一种 LLM 架构设计，在**同一 Transformer 层内并行**放置 Attention heads 和 SSM (Mamba) heads，两者同时处理相同输入，输出经 learnable per-channel 重缩放后融合。与传统 sequential hybrid（如 Samba/Jamba 交替堆叠 Attention 层和 Mamba 层）不同，hybrid-head 的并行设计使两种算子互补：SSM heads 提供全局上下文摘要（类比 fading memory），Attention heads 提供高分辨率局部召回（类比 snapshot memory）。统一对称公式为：

$$Y = W_{\text{out\_proj}} \left( \beta_1 \cdot \text{norm}(M_{\text{attn}} \tilde{X}) + \beta_2 \cdot \text{norm}(M_{\text{ssm}} \tilde{X}) \right)$$

其中 $M_{\text{attn}} = \text{softmax}(QK^T) W^V$，$M_{\text{ssm}} = G \odot \alpha(A,B,C,\Delta) W^{SSM}$，$\beta_1, \beta_2$ 为可学习 per-channel 重缩放向量。Hymba 发现 SSM heads 输出幅度始终大于 attention heads（Fig. 12），因此引入归一化+重缩放保证训练稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hymba hybrid-head 单层前向：

```
X̃ = concat([R, X], dim=0)          # prepend 128 meta tokens

# 统一输入投影
Q, K, V = W^Q @ X̃, W^K @ X̃, W^V @ X̃
X_ssm = W^{SSM} @ X̃
G = W^G @ X̃

# Attention heads（仅 3 层 global，其余 sliding window）
Y_attn = softmax(Q @ K^T / √d_head + causal_mask) @ V

# SSM heads（Mamba-style recurrent）
h_0 = 0
for i in 1..N+m:
    Δ_i = softplus(W_Δ @ X_ssm[i])
    Ā_i = exp(Δ_i ⊗ A)
    h_i = Ā_i ⊙ h_{i-1} + (Δ_i ⊗ (W_B @ X_ssm[i])) ⊙ X_ssm[i]
    y_i = (W_C @ X_ssm[i]) @ h_i
Y_ssm = G ⊙ Y

# 融合
Y = W_out_proj(β₁ ⊙ norm(Y_attn) + β₂ ⊙ norm(Y_ssm))
```

Hymba-1.5B 配置：32 layers, 25 attn heads (5 GQA groups), attn:mamba 参数比 1:5.23, 3/32 layers 为 global attention, window_size=1024。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hybrid-head 场景：(1) 高效小模型（<2B）——Hymba-1.5B 超越 Llama-3.2-3B，cache 缩小 11.67×；(2) 长上下文——SSM recurrent 支持外推；(3) 端侧部署——小 cache（79MB at 8K）。实现基于 PyTorch + Mamba selective scan + FlashAttention。局限：单层参数量略大于纯 Transformer/Mamba；训练需维护两种机制。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models
