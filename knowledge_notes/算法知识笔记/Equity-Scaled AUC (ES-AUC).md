## Equity-Scaled AUC (ES-AUC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equity-Scaled AUC（ES-AUC，公平缩放 AUC）是哈佛医学院团队提出的性能-公平性联合度量指标，用于量化模型在整体性能和组间公平性之间的权衡。与单独使用 AUC（仅衡量性能）或 DPD/EOD（仅衡量公平性）不同，ES-AUC 将两者统一为单一指标。

定义（Fair-MoE 论文中的表述）：

$$ES\text{-}AUC_s = \frac{AUC_s}{1 + \sum_a |AUC_s - AUC_{s,a}|}$$

其中 AUC_s 是属性 s 上的整体 AUC，AUC_{s,a} 是属性 s 中组 a 的组内 AUC。分母中的惩罚项 Σ|AUC_s - AUC_{s,a}| 衡量各组 AUC 偏离整体 AUC 的程度——偏离越大，惩罚越重，ES-AUC 越低。当所有组 AUC 完全相等时，ES-AUC = AUC（无惩罚）；各组差距越大，ES-AUC 衰减越多。

在 Fair-MoE 论文中，ES-AUC 是 primary evaluation metric（因它同时衡量 effectiveness 和 fairness）。FairMoE/l14 在 Race 上 ES-AUC=72.53（+5.00% vs FairCLIP/l14）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_ES_AUC(y_scores, y_true, groups):
    """
    y_scores: 模型预测分数 [N]
    y_true: 真实标签 [N]
    groups: 受保护属性组标签 [N]
    """
    from sklearn.metrics import roc_auc_score
    
    # 整体 AUC
    auc_overall = roc_auc_score(y_true, y_scores)
    
    # 各组 AUC
    auc_groups = {}
    for g in unique(groups):
        mask = (groups == g)
        auc_groups[g] = roc_auc_score(y_true[mask], y_scores[mask])
    
    # 惩罚项 = 各组 AUC 偏离整体 AUC 的绝对值之和
    penalty = sum(abs(auc_overall - auc_g) for auc_g in auc_groups.values())
    
    # ES-AUC
    es_auc = auc_overall / (1 + penalty)
    return es_auc
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ES-AUC 最初在 Fair Identity Normalization (FIN) 和 Fair Adaptive Scaling (FAS) 论文中提出，用于青光眼和糖尿病视网膜病变筛查。ES-AUC 的优势在于它直接反映"提升整体性能是否会以牺牲某组性能为代价"——如果某组性能下降，即使整体 AUC 提升，ES-AUC 也会因惩罚项增大而不升反降。这使得 ES-AUC 成为多组公平性场景下的首选联合评估指标。Fair-MoE 在所有受保护属性和 backbone 上均以 ES-AUC 为主要对比指标。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
