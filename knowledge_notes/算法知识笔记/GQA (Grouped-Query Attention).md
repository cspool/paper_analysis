## GQA (Grouped-Query Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA) 是 Ainslie et al. (2023, EMNLP) 提出的 attention 变体，介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间。GQA 将 query heads 分为多个组，每个组共享一组 key-value heads：

- MHA：H 个 query heads, H 个 KV heads（KV cache = 4 × H × d_h × l）
- GQA：H 个 query heads, G 个 KV heads, G < H（KV cache = 4 × G × d_h × l）
- MQA：H 个 query heads, 1 个 KV head（KV cache = 4 × 1 × d_h × l）

GQA 在保持接近 MHA 表达能力的同时大幅减少 KV cache。在 Hunyuan-Large 中，设置 80 个 query heads、8 个 KV groups（G=8），相比 MHA 的 80 个 KV heads 将 KV cache 减少 10 倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GQA forward: H query heads, G KV groups
# 输入: x [B, L, d_model]
# KV heads 数: G (每组有 H/G 个 query heads 共享 KV)

# 投影
Q = x @ W_q  # [B, L, H*d_k]  — 标准 MHA query
K = x @ W_k  # [B, L, G*d_k]  — 仅 G 组 KV
V = x @ W_v  # [B, L, G*d_k]

# Reshape
Q = reshape(Q, [B, L, H, d_k])
K = reshape(K, [B, L, G, d_k])
V = reshape(V, [B, L, G, d_k])

# 计算 attention: query head h 使用 KV group h // (H/G)
for h in range(H):
    g = h // (H // G)                        # 确定 KV group
    score = Q[:,:,h,:] @ K[:,:,g,:].T / sqrt(d_k)
    attn = softmax(score + mask)
    out[:,:,h,:] = attn @ V[:,:,g,:]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA 已被广泛采用——LLaMA 2/3、Mistral、Gemma、Hunyuan-Large 等主流 LLM 均使用 GQA。实现时可以复用 MHA 的代码框架，仅需调整 KV head 数量和 repeat KV 的维度。在 HuggingFace Transformers 中，`num_key_value_heads` 参数即指定 GQA 的 KV group 数。GQA 也可与 FlashAttention 结合使用，通过 `flash_attn_func` 的 GQA 模式支持。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
