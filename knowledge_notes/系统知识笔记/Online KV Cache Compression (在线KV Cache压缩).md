## Online KV Cache Compression (在线KV Cache压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online KV Cache Compression 是 Dynamic-LLaVA 在 decoding with KV cache 模式下实现的、基于当前 token 特征逐 token 决策是否将 KV activations 加入 cache 的方法。与传统 KV cache 压缩方法（如 H2O）的根本区别：H2O 从历史 KV cache 中基于 attention scores 淘汰旧 token，是"回顾性"压缩；Dynamic-LLaVA 对每个新生成的 token 通过 output predictor 做"是否保留"决策，决定其 KV 是否写入 cache，是"前瞻性"在线压缩。核心优势：不需要访问历史 KV cache 内容（无需额外 attention 计算），仅依赖当前 token 的 embedding 特征，因此额外开销极小（<1%）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**Dynamic-LLaVA 在线 KV Cache 压缩流程（batch=8, 1×A100 80G）**：

```
// Mini-batch KV cache 集合
KV_batch = {{S_l^{K(b)}, S_l^{V(b)}} | b=1..B}

for each decoding step t:
    for each batch b in 1..B:
        // Step 1: 当前 token 的 embedding → predictor 决策
        token_emb = S_{l,N^{OT}}^{OT(b)}              // [d]
        D_b = P^{OT}(token_emb)                        // [2]
        M_b = argmax(D_b)                              // 0 或 1

        // Step 2: 标准 attention（始终包含当前 token）
        Q_b, K_new, V_new = W^{Q,K,V} · token_emb
        O_b = Attention(Q_b, S_l^{K(b)} ∪ K_new, S_l^{V(b)} ∪ V_new)

        // Step 3: 在线决策——是否持久化 KV
        if M_b == 1:
            S_l^{K(b)} ∪= K_new, S_l^{V(b)} ∪= V_new  // 保留
        else:
            // 不加入 KV cache（但仍参与当前 attention）
            // 后续层的 KV cache 大小相应减少

    // Batch-parallel: LeftPadding + TopkArgmax
    // padded_KV: [B, max_len, d] → 批量 Attention
```

术语一般如何实现？如何使用？

与 H2O 的对比：H2O 在每次 decoding 时计算当前 Q 与所有历史 KV 的 attention scores → 淘汰 score 最低的历史 token → 需要显式 attention scores（与 FlashAttention 等隐式计算 operator 不兼容）→ 额外开销大。Dynamic-LLaVA 的 output predictor 仅需当前 token 特征 → 与 FlashAttention 完全兼容 → 额外开销 <1%。

在 MLLM 场景下，H2O 的 attention-score-based 淘汰策略在混合模态（vision+language）下严重退化（SciQA -16.3%, MMBench 仅 1.4），因为 vision token 和 language token 的 attention 模式差异大。Dynamic-LLaVA 通过端到端训练学习模态感知的保留策略，MLLM 场景下 PPL=4.90 vs H2O 78.95。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
