## Multi-Trial Logit Aggregation / Cross-Refinement (多试次Logit聚合与交叉验证)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Trial Logit Aggregation 是 T3S 的输出融合策略，用于将 m 个独立视频子序列推理得到的 m 个 logit 向量聚合为最终预测。论文提出三种聚合策略：(A) Mean Logits——直接对各试次 logit 取均值，参数无关且可靠性高；(B) Confidence-Weighted Aggregation——根据各试次预测分布的逆熵加权（低熵=高置信度=高权重）；(C) Two-Trial Cross-Refinement (m=2)——非对称验证方案：试次 1 提出 top-k 候选 token 集合，试次 2 在候选集上重新排序选出最优。方法 (C) 被证明在 m=2 时最优：试次 1 的采样保留率（α₁=0.5）高于试次 2（α₂=0.3），试次 1 拥有更多视觉信息适合提出候选，试次 2 用更稀疏但不同的视角验证。消融实验表明 Two-Trial Cross-Refinement 在所有 benchmark 上均优于 Mean Logits 和 Confidence-Weighted（VideoMME: 65.2 vs 65.1 vs 64.7; LongVideoBench: 62.3 vs 62.0 vs 61.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === 三种Logit聚合策略 ===
# 输入: logits o₁, o₂, ..., oₘ ∈ R^D (D=词表大小)

# (A) Mean Logits (默认, m任意)
def mean_aggregation(o_list):
    o_avg = sum(o_list) / len(o_list)
    return argmax(o_avg)

# (B) Confidence-Weighted (m任意)
def confidence_weighted(o_list):
    weights = []
    for o_i in o_list:
        pi = softmax(o_i)                    # 预测概率分布
        H_i = -sum(pi * log(pi))              # 熵 (越低越确定)
        weights.append(1.0 / H_i)             # 逆熵权重
    weights = normalize(weights)
    o_weighted = sum(w_i * o_i for w_i, o_i in zip(weights, o_list))
    return argmax(o_weighted)

# (C) Two-Trial Cross-Refinement (m=2, k=2)
def cross_refinement(o1, o2, k=2):
    # 第一阶段: 试次1 (α₁=0.5, 信息更全)提出top-k候选
    K = argsort(o1, descending=True)[:k]      # TopK(o₁, k)
    # 第二阶段: 试次2 (α₂=0.3, 但覆盖不同帧)重新排序
    t_star = argmax_{t∈K} o2[t]
    return t_star

# 直觉: 试次1做"生成"(宽覆盖), 试次2做"验证"(不同视角检验)
# 消融结果(Table 6): Two-Trial > Mean > Confidence-Weighted
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现中，聚合策略通过参数配置选择。推荐使用 Two-Trial Cross-Refinement（m=2 时）或 Mean Logits（m>2 时）。Confidence-Weighted 在 MLVU M-Avg 上表现最好（69.5），但整体略逊于 Cross-Refinement。Top-k 参数 k 对性能不敏感（2-100 范围波动 <1%），论文推荐 k=2 以获得最高的推理效率。在自回归生成中，每一步都需要执行 logit 聚合以决定下一个 token，因此聚合策略的计算开销必须极低（O(m·D)），T3S 的三种策略均满足此要求。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
