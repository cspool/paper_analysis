## Cross-Attention for Multi-Modal Fusion (交叉注意力多模态融合)

术语解释
Cross-Attention 是 Transformer 架构中的一种注意力机制变体，其中 Query 来自一个模态/序列，Key 和 Value 来自另一个模态/序列，实现跨模态信息交互。在 BrainMoE 中，Cross-Attention 用于将原始 FC 矩阵（结构信息）融合到 cognition embeddings（语义信息）中。

术语是什么？
标准 Self-Attention: Q, K, V 来自同一输入。Cross-Attention: Q 来自一个源，K, V 来自另一个源。

在 BrainMoE Cognition Adapter 中：
- Q = I · α̂_h：来自 FC 矩阵 I∈R^{M×M}（脑连接组结构），通过线性投影到 C_hid 维
- K = Z̄ · β̂_h：来自 cognition embeddings + task queries 的混合表示
- V = I · γ̂_h：同样来自 FC 矩阵

Cross-attention 使每个 task query 能根据 brain connectivity 模式自适应地 attend 到相关的 cognition embedding 部分。本质上将"脑区之间如何连接"（FC 矩阵）作为上下文来解读"脑在特定认知状态下如何活动"（cognition embedding）。

从算法pipeline角度拆解术语。
```
# BrainMoE Cross-Attention 的具体操作
# 输入: Z_bar [(k+P), C_hid], FC [M, M]
# 输出: updated Z_bar

for head h in range(num_heads):
    # Query 来自FC矩阵（brain structure）
    Q_h = FC @ alpha_hat_h       # [M, C_hid]
    
    # Key 来自混合表示（cognition semantics）
    K_h = Z_bar @ beta_hat_h     # [(k+P), C_hid]
    
    # Value 来自FC矩阵（brain structure）
    V_h = FC @ gamma_hat_h       # [M, C_hid]
    
    # Scaled Dot-Product Attention
    A_h = Softmax(Q_h @ K_h^T / sqrt(C_hid))  # [M, (k+P)]
    O_h = A_h @ V_h              # [M, C_hid]

# 合并多头 + 残差
O = concat([O_1, ..., O_H]) @ W_out
Z_bar = Z_bar + O[:k+P]         # 取前(k+P)行作为残差
```

核心张量维度：
- α̂_h, γ̂_h ∈ R^{M×C_hid} (M=116 regions)
- β̂_h ∈ R^{C_hid×C_hid}
- Attention: M brain regions attend to (k+P) token queries

术语一般如何实现？如何使用？
- 通用使用场景：多模态融合（文本-图像：Q=text, K=V=image；音频-文本等）、编码器-解码器 attention（Q=decoder, K=V=encoder）
- BrainMoE 的特殊用法：以脑结构信息（FC）为 Q 和 V，以认知语义信息（cognition embeddings）为 K，实现"structure-aware"的认知融合
- 不同于标准 cross-attention 中 Q 来自 decoder（目标模态）——这里 Q 来自 FC 矩阵，起到"通过脑连接组结构的镜头来理解认知状态"的作用

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
