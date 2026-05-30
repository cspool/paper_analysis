## Token Filtering for SSM Context Extension

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Filtering是LongMamba的核心技术——跳过不重要token的隐藏状态更新来扩大Mamba全局通道的感受野。当S≫L时，对每个全局通道c，若Δ_t[c] < g_c(S)则设置Ā'_t[c]=1、B̄'_t[c]=0（H_t[c]=H_{t-1}[c]，不衰减也不更新）。阈值g_c(S)是per-channel查找表（1000-token间隔），通过Pile采样序列标定Δ_t分布并数值求解使筛选后∏Ā'_i≈∏_{trained}Ā_i来确定。核心insight：Δ_t可解释为token"重要性"——大Δ=重要token应保留更新，小Δ=不重要可跳过以减少衰减累积。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LongMamba Token Filtering (推理时，per Mamba layer):
For t = 1..S:
    Δ_t = Softplus(Linear_Δ(X_t))
    Ā_t = exp(Δ_t ⊙ A);  B̄_t = Δ_t ⊗ B_t
    For each channel c:
        if is_global[c] and Δ_t[c] < g_c(S):
            Ā'_t[c] = 1;  B̄'_t[c] = 0   # 跳过该token
        else:
            Ā'_t[c] = Ā_t[c];  B̄'_t[c] = B̄_t[c]
    H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t

# 对齐条件: ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
标定：5条Pile随机序列，grid search确定clamping百分位C∈{0,5,10,15,20}。查找表1000-token间隔预计算，推理S向下取整到最近间隔。延迟开销极小（A100 prefill增加≤3.8%）。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---
