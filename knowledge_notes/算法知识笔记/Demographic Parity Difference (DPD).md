## Demographic Parity Difference (DPD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Demographic Parity Difference（DPD，人口均等差异）是机器学习公平性的基础度量指标，衡量不同受保护属性组（如不同种族、性别）获得正向预测结果的概率差异。DPD 越小表示模型越公平（理想值为 0）。

在 Fair-MoE 论文中，对于受保护属性 s 的所有组 a 和 b：

$$DPD_s = |\max_a P(\hat{y}=1|G=a, y=1) - \min_b P(\hat{y}=1|G=b, y=1)|, \quad a \neq b$$

其中 ŷ=1 表示正向预测（诊断患病），G=a 表示属于属性组 a，y=1 表示真实患病。DPD 关注的是"在真实患病的人群中，不同组获得正确诊断的概率是否一致"。DPD < 0.1 通常被认为公平。

在 Harvard-FairVLMed 青光眼诊断任务中，CLIP/b16 的 Race DPD=14.57（高度不公平），FairMoE/l14 的 Race DPD=2.63（接近公平）。DPD 适用于任何存在受保护属性的分类任务，是 most widely used group fairness metric。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DPD 在模型评估 pipeline 中的计算（以二分类医学诊断为例）：

```
# 输入: predictions ŷ, true labels y, protected attribute groups G
def compute_DPD(y_pred, y_true, groups):
    # 仅考虑真实患病人群 (y=1)
    positive_mask = (y_true == 1)
    group_probs = {}
    for g in unique(groups):
        group_mask = positive_mask & (groups == g)
        group_probs[g] = mean(y_pred[group_mask] == 1)
    # DPD = 最大组概率 - 最小组概率
    return max(group_probs.values()) - min(group_probs.values())
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPD 在 FairLearn（Microsoft）、AIF360（IBM）、FairTorch 等公平性工具包中均有标准实现。在训练中，DPD 通常不作为直接优化目标（不可微），而是作为评估指标或通过代理 loss（如 adversarial debiasing、contrastive fairness loss）间接优化。Fair-MoE 未直接优化 DPD，而是通过 FOL 优化 gate weight 的方差差异，间接降低 DPD。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
