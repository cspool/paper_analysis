## Question-Independent KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Question-Independent KV Cache Compression 是 KV-Distill 定义的两大 KV Cache 压缩范式之一（另一为 Question-Aware）。在问题无关范式中，上下文在不知道具体问题时被压缩为紧凑表示，供后续任意多轮问答复用。问题感知范式可利用问题信号（问题→上下文的 attention）定位关键信息。问题无关压缩更难：必须在不了解未来查询时预测哪些信息值得保留。核心应用场景：固定知识库/长文档压缩一次供反复查询（如 RAG 文档索引压缩）。

从算法pipeline角度拆解：

```
// H2I (问题无关, 无训练): 仅 context self-attention
scores = Σ softmax(Q_ctx @ K_ctx^T)  // context 内部
// LLAMA-3 SQuAD 25%: 56.6%

// KV-Distill (问题无关, 可训练): scorer 学会预测重要性
s = FFN_scorer(hidden_η)  // 从 hidden states 学习
top_k = argtopk(s, k)
X_comp = LM_θ.encode(context, selected=top_k)
// LLAMA-3 SQuAD 25%: 86.6% (vs H2I 56.6%, H2A 84.0%)
```

术语一般如何实现？如何使用？

训练数据拆分为 (Context, Instruction, Answer) 三元组，压缩仅应用于 Context 部分。评估时，context 压缩后与 question 拼接送入模型。长上下文使用 folding 技巧（pad→reshape batch→分别压缩→unfold）。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---
