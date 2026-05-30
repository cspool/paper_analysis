## Duo FFN Module (Heterogeneous Per-Layer Expert Pair)

术语解释
Duo FFN Module 是 Duo-LLM 的核心架构设计：在 Transformer 的每层 FFN 中并排放置一个大 FFN 和一个小 FFN（16x 尺寸差异），两者共享同一个 attention 模块。这是实现 token 级别自适应计算的最小异构 MoE 单元。

术语是什么？
每层包含：(1) 共享的 Multi-Head Attention；(2) Big FFN（如 inner_dim=10240，SwiGLU）；(3) Small FFN（inner_dim=640，16x smaller，SwiGLU）。两个 FFN 的输入/输出维度相同（d_model=2560），仅中间维度不同。训练时以 p=0.5 随机路由 token 到 Big 或 Small FFN，使两者学习互换性。

从算法pipeline角度拆解术语：
```
# Duo FFN Module per layer
# d_model = 2560
# Big FFN:   inner_dim = 10240, params ≈ 2560×10240×3 ≈ 78.6M/layer
# Small FFN: inner_dim = 640,   params ≈ 2560×640×3   ≈ 4.9M/layer

def duo_ffn_layer_forward(x, route_decision):
    # x: [batch, d_model]
    
    # 1. Shared Attention (RMSNorm + MHA + residual)
    x_norm = RMSNorm(x)
    attn_out = MultiHeadAttention(x_norm)
    x = x + attn_out
    
    # 2. Duo FFN (RMSNorm + FFN choice + residual)
    x_norm = RMSNorm(x)
    if route_decision == 'big':
        # Big FFN: gated SwiGLU
        gate = x_norm @ W_gate_big    # [batch, 10240]
        up   = x_norm @ W_up_big      # [batch, 10240]
        act  = SiLU(gate) * up        # [batch, 10240]
        ffn_out = act @ W_down_big    # [batch, 2560]
        flops = 5 * 2560 * 10240      # ≈ 131M FLOPs
    else:
        # Small FFN: gated SwiGLU (16x smaller)
        gate = x_norm @ W_gate_small  # [batch, 640]
        up   = x_norm @ W_up_small    # [batch, 640]
        act  = SiLU(gate) * up        # [batch, 640]
        ffn_out = act @ W_down_small  # [batch, 2560]
        flops = 5 * 2560 * 640        # ≈ 8.2M FLOPs
    
    x = x + ffn_out
    return x, flops  # Big: 131M, Small: 8.2M (16x reduction)
```

术语一般如何实现？如何使用？
- Duo-LLM 使用 12 层该架构，总计 1.399B 参数（big FFN: 944M, small FFN: 59M, attention: 314M, embedding: 82M）
- 训练策略：random routing (p=0.5) 从零训练优于 freeze big + fine-tune small
- 训练数据：300B tokens (FineWeb, Wiki, Flan/Dolma, Python Stack-v2)
- 推理时 router 动态选择每层每个 token 的 FFN 路径
- 论文提及 Megablocks 的 block-sparse matmul 可高效执行此架构，但未提供具体实现
- 该架构是一种简化的异构 MoE：仅有 2 个 expert（big/small），但允许 fine-grained per-layer per-token routing

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---
