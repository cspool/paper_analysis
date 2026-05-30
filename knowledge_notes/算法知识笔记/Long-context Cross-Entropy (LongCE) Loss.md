## Long-context Cross-Entropy (LongCE) Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LongCE（Long-context Cross-Entropy）Loss 由 Fang et al.（2025, ICLR 2025）提出，发表于论文 "What is Wrong with Perplexity for Long-context Language Modeling?"。核心发现：标准 perplexity 对所有 token 等权平均，无法区分对长上下文理解关键（key tokens）和无关键（ordinary tokens）的 token，导致 perplexity 与长上下文 benchmark 性能相关性差（Pearson 接近 0）。LongCE 通过 long-short context contrastive 方法识别 key tokens（在长上下文中预测显著不同于短上下文预测的 token），并在 CE loss 中对 key tokens 施加更高权重（weight > 1），对 ordinary tokens 维持 weight ≈ 1。这使得模型在长上下文继续预训练时自动聚焦于对长程依赖关键的 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LongCE Loss 计算流程：
```
# 给定: 长上下文模型 P_long(token|ctx_N), 短上下文模型 P_short(token|ctx_k)
# k << N, 例如 k=4096, N=65536

For each token position i in sequence:
    # Compute prediction discrepancy between long and short context
    p_long = P_long(x_i | x_{i-N:i-1})
    p_short = P_short(x_i | x_{i-k:i-1})
    
    # Key token identification: token hard to predict without full context
    discrepancy = |log(p_long) - log(p_short)|
    
    # Dynamic weight assignment
    w_i = 1 + λ * discrepancy  # λ controls weighting strength
    
    # Weighted cross-entropy
    loss_i = -w_i * log(P_long(x_i | context))

LongCE_loss = mean(loss_i)  # average over sequence
```

在 RWKV-X 中，LongCE 被用于 long-context continual pretraining 阶段（ProLong-64K 数据集，64K context）。消融实验（Table 4）：S-NIAH-2 8K 上 w/ LongCE 99.8 vs w/o 67.0；S-NIAH-3 8K 上 w/ LongCE 95.6 vs w/o 62.6。LongCE 在深层长上下文推理任务（S-NIAH-2/3）上效果显著，在简单任务（S-NIAH-1 passkey retrieval）上无差异（两者均为 100%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongCE 作为 plug-and-play 训练策略，可直接替换标准 CE loss 用于任何 LLM 的长上下文继续预训练。长/短上下文模型的对比可通过：(1) 使用两个独立的模型（long context + short context checkpoints）；(2) 使用同一模型在不同 context window 下的预测差异。在 RWKV-X 中，关键 token 识别基于 ProLong 论文的 long-short context contrastive 方法。代码开源：https://github.com/PKU-ML/LongPPL。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---
