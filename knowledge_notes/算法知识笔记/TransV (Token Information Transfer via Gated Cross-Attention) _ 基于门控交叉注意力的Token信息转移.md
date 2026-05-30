## TransV (Token Information Transfer via Gated Cross-Attention) / 基于门控交叉注意力的Token信息转移

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TransV 是 TimeViper 提出的 LLM 内部视觉 token 压缩模块，通过 Gated Cross-Attention 将冗余视觉 token 的信息显式转移到指令 token 中，再丢弃原始视觉 token。核心思想：不同于传统的 token dropping（不可逆丢失信息）或 token compression into new special tokens（破坏 token 身份），TransV 先通过 cross-attention 将被丢弃的 vision tokens 作为 KV、instruction tokens 作为 Q 计算信息增量，再通过门控因子 tanh(α_l) 控制转移强度，将信息"存"进 instruction tokens，最后安全地丢弃 vision tokens。TransV 增加约 100M 参数（相对于 9B backbone 约 1.1%），学习率 5e-5（高于 LLM backbone 的 1e-5）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TransV at layer l: compress and transfer vision tokens -> instruction tokens
def TransV(X_0, X_1, l):  # X_0: vision tokens [T_0, D], X_1: instruction tokens [T_1, D]
    # Step 1: Token dropping strategy
    if l == shallow_layer:  # e.g., layer 7
        X_0_kept, X_0_dropped = UniformDrop(X_0, rate=0.5)
    elif l == deep_layer:   # e.g., layer 39
        attn_scores = Attention(X_1[-1], X_0)  # last inst token as query
        keep_ids = TopK(X_0, score=-attn_scores, k=T_0 * 0.1)
        X_0_kept = X_0[keep_ids]
        X_0_dropped = X_0[~keep_ids]
    # Step 2: Gated cross-attention transfer
    X_1_tilde = CrossAttn(Q=X_1, KV=X_0_dropped)  # [T_1, D]
    # Step 3: Gated addition
    alpha = tanh(alpha_l)  # alpha_l init=0, range [-1, 1]
    X_1_new = X_1 + alpha * X_1_tilde
    return X_0_kept, X_1_new
```

Annotations: T_0 = 16×N_frames (after ToMe); shallow TransV: uniform dropping（first attention前Mamba层attention score不可靠）；deep TransV: attention-guided dropping（attention已可靠）；α_l init=0 确保初始时instruction理解不受影响；总压缩: (1-0.5)×(1-0.9) = 5% vision tokens保留。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TimeViper 配置：浅层 TransV 在第 7 层（uniform, p=50%），深层 TransV 在第 39 层（attention-guided, p=90%）。两阶段训练中 TransV 仅在 Stage 2 启用。关键消融：(1) TransV vs token dropping：TVG 上 38.1 vs 26.1；(2) 浅层位置：第 7 层 vs 第 2 层 MCQ 上 +0.6 但 VDC 上 -0.8；(3) 压缩率：50% vs 90% 导致 MCQ 从 56.7 降到 53.4。TransV 也可用于 Qwen2.5 Transformer backbone，但 Qwen 的 VDC 下降更大 (1.3 vs Nano's 0.6)。代码：https://github.com/xiaomi-research/timeviper (TBD)。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
