## Token Sensitivity-Aware KV Cache Management

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Sensitivity-Aware KV Cache Management（token敏感度感知的KV cache管理）是一种识别和保护高敏感度token的KV cache压缩增强策略。核心思想：并非所有token对模型输出质量贡献相同——某些token包含任务关键信息（如问题中的实体名、特定指令token），其KV状态在合并或丢弃时会对生成质量造成显著负面影响。这些token被定义为"高敏感度token"。FlowMM将sensitivity定义为token对模型输出保真度的贡献度——若合并某token的KV状态导致后续生成准确度/相关性显著下降，则该token为高敏感度。直接通过逐一扰动测试测量敏感度计算成本过高，因此FlowMM使用attention scores作为sensitivity的零开销近似：attention scores直接量化token对当前生成步骤的影响，可在正常attention计算中免费获得。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Sensitivity-Adaptive Token Matching (FlowMM公式9-10):
# Step 1: 计算token间余弦相似度
for token i in K^n (non-pivot set):
    for token j in K^p (pivot set):
        u_{i,j} = (k_i^T · k_j) / (||k_i|| · ||k_j||)

# Step 2: Sensitivity-gated nearest neighbor matching
for token i in K^n:
    # 仅在低敏感度pivot (I_j ≤ τ) 中搜索最近邻
    j* = Argmax_{j∈K^p, I_j ≤ τ}(u_{i,j})
    # 高敏感度pivot (I_j > τ) 被保护，不接受任何合并
    merge(K_i, V_i) → K_{j*}, V_{j*}

# 设计逻辑：
# - 低敏感度pivot: 可接受合并 → 信息聚合点
# - 高敏感度pivot: 不接受合并 → 信息保护
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sensitivity evaluation使用proxy tokens方法（FlowMM公式8）：选择prompt末尾少量token作为proxy（这些token通常capture任务特定上下文信息），对每个token i计算其从所有proxy tokens收到的attention scores之和：I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}。相比使用全局累积attention（可能biased），proxy token方法提供更公平的token重要性估计。敏感度阈值τ需校准：过高→保护过多pivot→合并候选不足→压缩效率降低；过低→高敏感度token未受保护→任务性能下降。FlowMM消融实验（Table 4）：移除sensitivity protection在TextNeedle任务上性能下降最显著（-3.68%），因为该任务需要精确保留特定文本token信息。该策略可与其他KV cache压缩方法组合使用。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference
