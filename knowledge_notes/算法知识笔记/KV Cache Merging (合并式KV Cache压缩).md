## KV Cache Merging (合并式KV Cache压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Merging是一种KV cache压缩范式，与eviction（丢弃）和quantization（量化）并列。核心思想：在保留KV cache预算有限时，不直接丢弃低重要性token的KV状态，而是将其**合并（merge）**到被保留的高重要性token的KV状态中，从而以紧凑表示保留更丰富的上下文信息。形式化：给定KV cache K_t, V_t ∈ R^{L×d}和目标压缩比B，首先选出top-B个pivot tokens的K^p, V^p（保留完整信息），然后将剩余的non-pivot tokens K^n, V^n按相似度合并到pivot tokens中：K^{merged}, V^{merged} = f_merge(K_t, S), g_merge(V_t, S)，其中S ∈ R^{L×L}为token间的相似度矩阵。典型合并操作使用余弦相似度匹配最近邻后做加权平均。Merging相比eviction的优势：即使被合并token的信息被压缩，其内容仍部分保留在pivot token中（而非完全丢失），减少context fragmentation和hallucination。挑战：(1) 不准确的相似度计算可能将无关token合并导致语义混淆；(2) 在多模态场景下，不同模态token的分布偏移（distributional divergence）使简单的余弦相似度合并不可靠。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 通用KV Cache Merging流程：
Input: K ∈ R^{L×d}, V ∈ R^{L×d}, budget B (保留比例)
Output: K^{merged} ∈ R^{B·L×d}, V^{merged} ∈ R^{B·L×d}

# Step 1: Token重要性评估（基于累积attention scores）
importance[i] = Σ_h Σ_{j∈recent_tokens} α_{j→i}^h

# Step 2: 选出pivot set和non-pivot set
sorted_indices = argsort(importance, descending=True)
pivot_indices = sorted_indices[:B·L]        # top-B 保留
non_pivot_indices = sorted_indices[B·L:]    # 其余将被合并
K^p = K[pivot_indices];    V^p = V[pivot_indices]

# Step 3: 基于相似度的最近邻合并
for each i in non_pivot_indices:
    similarities = cosine_similarity(K[i], K^p)  # [B·L]
    j_star = argmax(similarities)                # 最近邻pivot
    weight = attention_based_weight(i, j_star)
    K^p[j_star] = weight · K^p[j_star] + (1-weight) · K[i]
    V^p[j_star] = weight · V^p[j_star] + (1-weight) · V[i]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV Cache Merging的典型实现：在每层attention计算后，对KV cache做后处理合并。PyTorch实现使用torch.cosine_similarity和加权平均操作。常用变体：(1) KVMerge（Wang et al., 2024）——基于模型自身指示的合并位置决策；(2) CaM（Zhang et al., 2024）——将eviction候选合并到保留状态中；(3) MiniCache（Liu et al., 2024a）——利用层间KV相似度做intra-layer压缩；(4) LOOK-M（Wan et al., 2024b）——multimodal-specific的KV cache合并方法。FlowMM在此基础上引入了跨模态信息流引导的层自适应合并策略和敏感度自适应的token匹配，解决多模态场景下统一合并策略的不足。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference
