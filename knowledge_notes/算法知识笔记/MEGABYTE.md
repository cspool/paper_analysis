## MEGABYTE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MEGABYTE 是 Yu et al. (Meta, NeurIPS 2023) 提出的字节级多尺度 Transformer 架构，采用与 Block Transformer 相似的全局到局部分层结构（patch-level global decoder + byte-level local decoder）。关键差异：(1) MEGABYTE 针对 byte-level 数据（无 tokenizer），Block Transformer 针对 subword-level 推理优化；(2) MEGABYTE 认为 local module 应尽可能小（建议 6:1 global:local 参数比），Block Transformer 证明 1:1 更优且更大 token decoder 提升吞吐量；(3) MEGABYTE 使用 summation 注入全局信息，Block Transformer 使用 prefix。Block Transformer 的 token-level MEGABYTE reimplementation 显示，Block Transformer 在吞吐量上超 MEGABYTE 1.5×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MEGABYTE vs Block Transformer 关键结构对比：
```
# MEGABYTE: 6:1 ratio, summation
ctx_bias = Linear(ctx_emb).reshape(B, LB, D)
tok_input = tok_embs + ctx_bias  # summation, no refinement

# Block Transformer: 1:1 ratio, prefix
prefix = Linear(ctx_emb).view(B, P, D)
tok_input = Concat([prefix, tok_embs])  # prefix, allows self-attention refinement
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block Transformer 论文重新实现 token-level MEGABYTE（summation, 6:1 ratio, Pile 300B tokens training）以进行公平比较。MEGABYTE 官方实现为 byte-level：https://github.com/facebookresearch/megabyte。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---
