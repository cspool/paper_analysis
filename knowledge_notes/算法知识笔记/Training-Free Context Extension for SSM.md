## Training-Free Context Extension for SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-Free Context Extension for SSM指无需额外训练即可使预训练Mamba SSM处理远超训练长度输入序列的方法类别。传统Transformer上下文扩展（位置插值、Attention Sink等）因SSM缺少显式position encoding和attention机制而无法直接应用。当前方法包括：(1) DeciMamba——深层token pruning减少序列长度；(2) LongMamba——per-channel token filtering扩大全局通道感受野；(3) MambaExtend——校准离散化缩放因子。共同特征：不改变模型权重，仅修改推理时前向传播，通过离线标定少量超参数适配更长序列。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
通用范式:
阶段1 (离线标定): 使用训练集分析内部统计量 → 标定超参数
阶段2 (推理干预): 在SSM递归循环中插入条件逻辑 → 调整衰减/更新行为

LongMamba特化:
- 标定: Δ_t分布统计 + global/local通道分类 (θ search)
- 干预: global通道跳过Δ_t<g的token
- 对齐: ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongMamba可直接应用任何预训练Mamba/Mamba2/Zamba2模型，仅需~5条校准序列。局限：需访问训练集数据标定Δ_t分布、超参数需per-model搜索（如Mamba-1.4B θ=10^{-30} vs Mamba2-1.3B θ=5×10^{-2}）。适用于快速将预训练Mamba部署到长上下文场景。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---
