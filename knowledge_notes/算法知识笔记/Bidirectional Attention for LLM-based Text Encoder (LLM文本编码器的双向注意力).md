## Bidirectional Attention for LLM-based Text Encoder (LLM文本编码器的双向注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bidirectional Attention for LLM-based Text Encoder 是 LLM2CLIP 在 CC Fine-tuning 阶段对 LLM 架构的关键修改：移除 LLM 原生的 causal (autoregressive) attention mask，改为完全的 bidirectional attention，使每个 token 可以 attend 到序列中所有其他 token（前后双向）。原始 LLM 使用 causal mask（下三角矩阵）以确保自回归生成能力——每个 token 只能看到自身及之前的 token。但在纯编码场景（文本嵌入提取），生成能力不需要，双向建模能更充分地捕获文本的双向语义关系。具体实现：将 Transformer 层的 attention mask 从 causal（lower triangular）替换为全 1 矩阵 [1, L, L]（仅保留 padding mask）。LLM2CLIP 消融显示 bidirectional 与 causal attention 性能相近（Avg I2T 80.4 vs 80.0），但 bidirectional 理论上有更好的文本理解能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Causal Attention (原始 LLM)
def causal_attention(Q, K, V):
    # Q, K, V: [B, H, L, d_h]
    scores = Q @ K.transpose(-2, -1) / sqrt(d_h)  # [B, H, L, L]

    # Causal mask: 下三角矩阵, token i 只能 attend token 0..i
    causal_mask = torch.tril(torch.ones(L, L))     # 下三角=1, 上三角=0
    scores = scores.masked_fill(causal_mask == 0, -inf)

    attn = softmax(scores, dim=-1)
    return attn @ V

# Bidirectional Attention (LLM2CLIP Stage 1 & 2)
def bidirectional_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1) / sqrt(d_h)  # [B, H, L, L]

    # 仅使用 padding mask (无 causal mask)
    # pad_mask: [B, L], 1=有效token, 0=padding
    pad_mask_expanded = pad_mask[:, None, None, :]  # [B, 1, 1, L]
    scores = scores.masked_fill(pad_mask_expanded == 0, -inf)

    attn = softmax(scores, dim=-1)  # 所有 token 互相 attend
    return attn @ V

# 在 Llama 3.1 8B 中启用 bidirectional attention:
# 将 attention_mask 从 causal + padding → padding only
# 对应的 FlashAttention-2 调用中设置 is_causal=False
```

Annotations: causal mask 限制每个 token 只能看过去（prevents information leakage from future）；bidirectional attention 让 [CLS]-like pooling 可以汇聚双向语境信息。Bidirectional attention 在前向时可能导致 attention scores 对称化，但移除 causal mask 本身不改变计算量（仍为 O(L^2)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLM2CLIP 的实现中：Stage 1 CC Fine-tuning 和 Stage 2 的 LLM 编码均使用 bidirectional attention。具体实现方式：调用 HuggingFace Transformers 的 `model.forward(input_ids, attention_mask=padding_mask)` 时，如果模型原生有 causal mask，可通过 `use_cache=False` 和设置 `output_attentions=False` 配合自定义 attention mask 覆盖。或直接修改模型 config 中的 `is_causal` 属性。LLM2CLIP 消融 (Table 6/Table A5) 显示：bidirectional vs causal 差异很小（80.4 vs 80.0 Avg I2T），原因是 caption 文本通常较短（≤ 512 tokens），且双向语义信息已在 LLM 预训练的 causal objective 中被隐式学习（通过多层堆叠间接获取反向信息）。Bidirectional attention 的使用场景：任何将自回归 LLM 转为纯编码器（embedding model）的场景，如 LLM2Vec、NV-Embed-v2 均采用此策略。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
