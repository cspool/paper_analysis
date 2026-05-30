## Attentive Pooling（注意力池化）

术语是什么？
Attentive Pooling 通过可学习的 query 向量对 token 序列执行 cross-attention，将变长序列聚合为固定维度表示。与 average pooling（等权平均）和 max pooling 不同，attentive pooling 学习 token 的重要性权重，自动聚焦判别性区域。OV-Encoder 采用 multi-head attention pooling（源自 SigLIP），用少量可学习 query tokens 对 ViT 输出的全部 2048 个 spatiotemporal tokens 做 cross-attention，生成 compact embeddings 用于 cluster discrimination 损失。

从算法pipeline角度拆解术语：

```
# Multi-Head Attention Pooling
Q = learnable_query[N_queries, d]        # N_queries << M (通常1-4)
K, V = Z @ W_k, Z @ W_v                  # Z: ViT输出 [M, d]
attn = softmax(Q @ K.T / sqrt(d))        # [N_queries, M]
pooled = attn @ V                         # [N_queries, d]

# OV-Encoder 直接作用于 ViT 最后一层所有 patch tokens（无[CLS]）
# Codec Patchification 输入端筛选 + Attentive Pooling 输出端加权
```

术语一般如何实现？如何使用？
PyTorch 实现：`nn.MultiheadAttention` + 可学习 `nn.Parameter` query tokens。源自 SigLIP 的 MAP 设计。query tokens 随 ViT 一起优化。相比 [CLS] token：query 不与输入 token 耦合，可增加 N_queries 提升表达力。应用：任何需要将变长序列聚合为固定向量的 ViT/Transformer 输出端。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
