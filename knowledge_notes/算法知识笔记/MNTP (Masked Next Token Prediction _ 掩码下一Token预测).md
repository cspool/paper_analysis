## MNTP (Masked Next Token Prediction / 掩码下一Token预测)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MNTP (Masked Next Token Prediction) 是 LLM2Vec (BehnamGhader et al., 2024) 提出的一种 LLM 训练方法，灵感来源于 BERT 的 Masked Language Modeling。核心思想：在序列中 mask 掉特定 token，让 LLM 预测被 mask token 的下一个 token（而非被 mask token 本身），以对齐 LLM 的 next-token prediction 预训练惯例。与 BERT MLM 的区别：BERT 预测被 mask 的 token 本身（基于双向上下文），MNTP 预测 mask 位置之后的 token（利用 LLM 原生 next-token prediction 能力）。LLM2CLIP 评估了 MNTP 在 CLIP 跨模态场景中的效果：单独 MNTP (Avg I2T 70.1) 远低于 SimCSE (80.4)，MNTP + SimCSE 组合 (79.7) 也不优于 SimCSE alone，因此 LLM2CLIP 默认不使用 MNTP。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MNTP 训练流程
# 输入: 文本序列 "a cat sitting on a mat"
def mntp_training(llm, text_sequences):
    # 1. Mask tokens (e.g., mask "sitting")
    # tokens:    [a, cat, [MASK], on, a, mat]
    # labels:    [-,  -,  sitting, -, -,  -]  (只计算 mask 位置)
    masked_seq, labels = apply_mask(text_sequences)

    # 2. LLM 前向 (causal attention)
    logits = llm(masked_seq)  # [B, L, vocab_size]

    # 3. 仅在 mask 位置计算 next-token prediction loss
    # token "sitting" 的预测基于其之前的 tokens [a, cat, [MASK]]
    loss = CrossEntropy(logits[mask_positions], labels[mask_positions])
    return loss

# LLM2CLIP 中 MNTP 评估:
# 与 SimCSE 对比 (Table A5):
#   MNTP alone:            Avg I2T/T2I = 70.1/67.0
#   Unsupervised SimCSE:   Avg I2T/T2I = 59.2/57.7
#   Supervised SimCSE:     Avg I2T/T2I = 80.4/77.9
#   MNTP + SimCSE:         Avg I2T/T2I = 79.7/77.2
```

Annotations: MNTP 的 mask 位置由随机选择或启发式策略确定；MNTP 的优势在纯文本任务中已被 LLM2Vec 验证有效，但在跨模态场景（需要 caption 间语义区分能力）中，SimCSE 的对比式训练更直接地针对 embedding 可分离性优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MNTP 在 LLM2Vec 中的使用：通过 MNTP 激活 LLM 的双向上下文理解能力（尽管仍用 causal attention），使 LLM 的 hidden states 对上下文更敏感。LLM2CLIP 的评估显示 MNTP 在跨模态场景效果有限：(1) MNTP 训练目标是 token-level prediction，而 CLIP 需要的是 sentence-level 语义可分离性；(2) SimCSE 通过 sentence-level 对比损失直接优化 embedding 空间的分离度，对 CLIP 跨模态对比训练更匹配；(3) MNTP + SimCSE 组合未带来额外收益（79.7 vs 80.4），说明 SimCSE 已充分激话 LLM 的文本区分能力。MNTP 可能适用于：需要 LLM 输出 token 对上下文中词级语义更敏感的场景（如信息检索、文本匹配），而在跨模态 embedding 场景中 SimCSE 是更优选择。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
