## Positional Bias in Visual Token Pruning（视觉Token剪枝的位置偏见）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Positional Bias in Visual Token Pruning 是指基于 attention weights 的 inner-LLM 视觉 token 剪枝方法系统性地对序列末尾位置的 token 赋予更高的"重要性"分数，而忽略 token 的实际语义内容。在 LVLM 中，visual token 按空间位置顺序排列（通常从上到下、从左到右扫描）进入 LLM，因此位置偏见具体表现为：过度保留图像底部位置的 visual token，而丢弃图像上部/中部的语义重要 token。V2Drop 通过定量分析（Figure 3）证实：在 LLaVA-1.5-7B 和 Qwen2-VL-7B 上，FastV 和 SparseVLM 等 attention-based 方法在剪枝 50% token 后，末尾 20% 位置的 token 保留概率远高于前部 token，形成"end-of-sequence bias"。这一偏见与 token 的实际语义内容无关（content-agnostic），严重时加剧多模态幻觉——保留不相关 token 同时丢弃关键视觉信息。

产生原因：Transformer 的 causal attention mask 和 positional encoding 使得 LLM 天然倾向于关注序列后部位置，attention weights 将这种位置偏好投射为"重要性"信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
位置偏见通过以下量化方法度量：

```
# Quantifying positional bias
# 1. Apply attention-based pruning (e.g., FastV) at layer k
# 2. Partition vision tokens into N equal intervals by position index
# 3. For each interval j: compute retention probability
#    P_retain[j] = (# tokens retained in interval j) / (total tokens in interval j)
# 4. Visualize: P_retain vs. position interval

# Expected (no bias): P_retain[j] ≈ pruning_ratio, uniform across j
# Attention-based (with bias): P_retain[j] increases monotonically with j
```

V2Drop 证明 variation-based 评分产生近乎均匀的空间保留分布——高 variation 区域可以出现在图像的任何位置（上部、中部、下部），不受 token 序列位置影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
避免位置偏见的策略：(1) 使用与位置无关的 token 重要性信号（如 variation、duplication、entropy），而非 attention weights；(2) 对 attention weights 进行位置去偏处理（如减去 position-only baseline attention）；(3) 在 positional encoding 层面引入 spatial prior（如 2D RoPE）。目前 variation-based 方法（V2Drop）和 duplication-based 方法（DART）是最有效的 position-agnostic 方案。该概念对设计新一代 token compression 方法具有指导意义——任何依赖 LLM attention weights 的 token 重要性评估都应考虑位置偏见的去偏。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
