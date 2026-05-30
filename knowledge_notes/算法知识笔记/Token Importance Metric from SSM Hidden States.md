## Token Importance Metric from SSM Hidden States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Importance Metric for SSMs是Rethinking Token Reduction论文提出的从Mamba SSM隐藏状态评估每个token重要性的度量方法。度量公式为 `S = Σ_{d=1}^{D'} max(0, y_{::d}) / D'`，其中y ∈ R^{B×N×D'}是SSM层输出隐藏状态，max(0,·)（ReLU clip）只保留正向激活通道值，沿特征维D'求和除以D'得平均重要性。选择SSM隐藏状态的原因：SSM拥有高维通道空间（D'），能对每个token进行细粒度的多通道关注度分析，不同于Transformer的单一attention矩阵。clip操作优于ℓ1/ℓ2 norm和unclipped版本：只关注正向激活更有信息量。消融证实：clip版本Mamba-2-2.7B达PPL 17.96、Avg Acc 58.7%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
y = SSM_forward(A, B, C, x)               # y ∈ R^{B×N×D'}
S_clipped = sum(max(0, y), dim=-1) / D'   # ∈ R^{B×N×1} (论文最优)
S_l1 = sum(abs(y), dim=-1) / D'           # ℓ1-norm 对比
S_l2 = sqrt(sum(y^2, dim=-1)) / D'        # ℓ2-norm 对比

# Mamba-2-2.7B @20% FLOPS:
# Clip: PPL 17.96, Acc 58.7%
# ℓ1:   PPL 17.96, Acc 58.6%
# ℓ2:   PPL 19.86, Acc 58.6%
# 无Clip: PPL 18.17, Acc 58.5%
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码：https://github.com/wuyushuwys/ToR_SSM。实现为hook读取Mamba block的selective_scan中间tensor，无需额外模型修改或权重存储。与DeciMamba（用Δ_t）、LongMamba（用Δ_t区分全局/局部通道）的重要差别：直接使用SSM输出的hidden states作为信号源。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
