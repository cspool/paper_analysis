## MLP-MoE (Vision-Language Connector MoE)

术语解释
MLP-MoE 是 CuMo 论文提出的将 Top-K 稀疏门控 MoE 块集成到多模态 LLM 中 vision-language MLP 连接器的设计。MLP 连接器通常为两层线性 MLP，将 visual tokens 投影到 word embedding 空间。MLP-MoE 将此 MLP 替换为 Top-2-in-4 稀疏 MoE 块。

术语是什么？
标准 MLP connector：`visual_tokens → Linear1 → GELU → Linear2 → word_embedding_tokens`。MLP-MoE 将其替换为：Router（线性层 → Softmax → Top-2）选择 2/4 experts（每个 expert 同为两层 MLP），加权求和输出。

```
# MLP-MoE connector forward
def mlp_moe_connector(visual_tokens):  # [N, d_v]
    W = Softmax(Linear_router(visual_tokens))  # [N, 4]
    W_K_values, W_K_indices = TopK(W, K=2)
    W_K = Softmax(W_K_values)                  # [N, 2]

    word_embeddings = zeros(N, d_llm)
    for i in range(2):
        expert_idx = W_K_indices[:, i]
        expert_out = ExpertMLP_i(visual_tokens[expert_idx])
        word_embeddings[expert_idx] += W_K[expert_idx, i] * expert_out

    return word_embeddings  # 输入给 LLM
```

从算法pipeline角度拆解术语：
MLP-MoE 位于视觉编码器（CLIP-MoE）和 LLM 之间，负责维度转换。每个 expert 仅包含两个线性层（参数量小），因此 MLP-MoE 的总参数仅 0.10B（激活 0.05B）。CuMo 的消融实验（Table 3）表明：(1) 随机初始化 MLP-MoE → 性能明显下降；(2) Upcycling 初始化 → 边际提升；(3) 加入 bzloss → MMVet 明显提升；(4) Top-2-in-8 → 性能略降（数据不足以训练 8 个均衡专家）。

术语一般如何实现？如何使用？
- 微小的额外参数成本：总参数从 7.25B → 7.65B（+0.40B），激活参数从 7.25B → 7.60B（+0.35B）
- Co-Upcycling 从预训练 MLP connector 权重初始化
- 配合 bzloss（与 CLIP-MoE 的 bzloss 独立应用）
- 推理时 Router 仅增加极小的计算开销（一个 Linear 层 + Softmax + Top-K）

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---
