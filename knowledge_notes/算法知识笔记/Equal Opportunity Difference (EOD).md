## Equal Opportunity Difference (EOD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equal Opportunity Difference（EOD，机会均等差异）是比 DPD 更细粒度的公平性度量，同时考虑真正例率（TPR）和假正例率（FPR）在不同受保护属性组之间的差异。EOD 是 Equalized Odds 的差异版本，确保模型在各组上的分类错误类型分布一致。

在 Fair-MoE 论文中：

$$EOD_s = \max_{a,b \in s, a \neq b} (|P(\hat{y}=1|G=a,y=1)-P(\hat{y}=1|G=b,y=1)|, |P(\hat{y}=1|G=a,y=0)-P(\hat{y}=1|G=b,y=0)|)$$

第一项为 TPR 差异（真实患病者中获得正确诊断的概率差异），第二项为 FPR 差异（健康人中被误诊的概率差异）。取两者中的最大值。EOD 比 DPD 更严格，因为它要求模型在真正例和假正例两个维度上都公平。

在 Harvard-FairVLMed 上，CLIP/b16 的 Race EOD=18.47，FairMoE/l14 的 Race EOD=4.25（↓77%）。EOD 特别适用于医疗诊断场景——确保模型不会系统性地对某组产生更多假阳性（过度诊断）或更多假阴性（漏诊）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_EOD(y_pred, y_true, groups):
    tpr = {}  # True Positive Rate per group
    fpr = {}  # False Positive Rate per group
    for g in unique(groups):
        mask = (groups == g)
        pos = mask & (y_true == 1)   # 该组真实患病
        neg = mask & (y_true == 0)   # 该组真实健康
        tpr[g] = mean(y_pred[pos] == 1)
        fpr[g] = mean(y_pred[neg] == 1)
    max_tpr_diff = max(tpr.values()) - min(tpr.values())
    max_fpr_diff = max(fpr.values()) - min(fpr.values())
    return max(max_tpr_diff, max_fpr_diff)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EOD 在 FairLearn、AIF360 中作为标准指标实现。与 DPD 相同，EOD 通常是评估指标而非直接优化目标。在医疗 AI 公平性研究中，EOD 是最受关注的指标之一——因为它能同时暴露过度诊断（高 FPR）和漏诊（低 TPR）的组间不公平。Fair-MoE 通过 FO-MoE 和 FOL 间接优化 EOD。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
