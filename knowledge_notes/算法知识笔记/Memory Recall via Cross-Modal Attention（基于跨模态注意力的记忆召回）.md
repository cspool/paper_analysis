## Memory Recall via Cross-Modal Attention（基于跨模态注意力的记忆召回）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Recall via Cross-Modal Attention 是 FlexMem 中从 Visual Memory Bank 检索最相关记忆的机制。原理：在 memory encoding 阶段，如果 Tq（问题文本）被包含在 MLLM 输入中，则 MLLM 的 self-attention 会自然产生 Tq→Vi 的 cross-modal attention weights。这些权重反映了模型在理解问题时关注了哪些视觉区域，因此天然可度量 clip-问题的相关性。FlexMem 对这些 attention weights 求和作为 relevance score g_i = Σ_{l=3→L} Σ_{j∈Tq} Σ_{k∈Vi} a_{jk}^l，仅取深层（≥第3层），因为浅层 attention 分布均匀无区分力。最后选择 g_i 最高的 na 个连续 clip 的 memory 作为召回结果。该方法零额外计算（复用已有 forward pass 的 attention），但代价是必须在 encoding 阶段就传入 Tq。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Encoding-based Memory Recall ===
for i, Mi in enumerate(M_bank):
  g_i = 0
  for layer in 3..L:
    A = saved_attention_matrix[layer]  # 来自encoding的forward pass
    g_i += sum(A[j, k] for j in Tq_positions for k in Vi_positions)

start = argmax(sum(g[start:start+na]))  # 找na个连续最高分clip
Y = MLLM.decode(M_bank[start:start+na], Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该方法完全在 MLLM 的 forward pass 中实现——attention weights 是 self-attention 的中间结果，仅需在深层保存 Tq→Vi 的注意力子矩阵。论文实验（Table 1）均使用 encoding-based reading，在五个 long VideoQA benchmark 上取得 SOTA。消融（Table 5 Block 3）表明 memory recall 远优于 indiscriminate loading of all memory，验证了选择性召回的價值。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
