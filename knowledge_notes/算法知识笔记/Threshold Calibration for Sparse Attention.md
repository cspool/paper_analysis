## Threshold Calibration for Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Threshold Calibration for Sparse Attention是BLASST提出的自动阈值选择方法。由于固定阈值在不同context length下会产生不一致的sparsity（如λ=1e-3在4K context下sparsity仅23%，在64K下达到75%），需要动态适配阈值。校准过程（Algorithm 2）：(1) 在校准数据集D上做一次forward pass，计算所有attention scores；(2) 对每个候选threshold λ_j和每个样本(x_i, L_i)，从同一次attention scores中提取该λ_j下的achieved sparsity s_ij；(3) 记录数据点(λ_j·L_i, s_ij)；(4) 拟合指数模型 λ·L = α·exp(β·s)。一次forward pass即可覆盖所有候选threshold（因sparsity可从相同的attention scores离线计算不同λ的结果）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

校准算法pipeline：

```
# Input: calibration dataset D, threshold set Λ, sparsity bounds [s_min, s_max]
# Output: calibration parameters α, β

P = []  # data points
for (x_i, L_i) in D:
    # 单次forward pass → 获取所有attention scores
    all_scores = forward_pass(x_i)  # 使用dense FlashAttention
    
    for λ_j in Λ:
        # 从scores离线统计sparsity（不做重复forward pass）
        s_ij = measure_sparsity(all_scores, λ_j)  # 统计满足m̃-m < ln(λ)的block比例
        if s_min ≤ s_ij ≤ s_max:
            P.append((λ_j * L_i, s_ij))

# 拟合指数模型: λ·L = α·exp(β·s)
α, β = fit_exponential_model(P)

# 推理时: target sparsity S, context length L → λ = α·exp(β·S)/L
```

发现：λ与L成反比关系 λ = a/L，其中a = α·exp(β·S)。理论依据：attention scores按行归一化到和为1，更长序列中每个token的平均score更低，需要更小的threshold才能实现相同的sparsity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际部署流程：(1) 用~1000条RULER样本在不同context length（4K-64K）上执行一次dense forward pass，(2) sweep λ∈[1e-6, 1e-1]范围计算每个λ下的sparsity，(3) 拟合α,β参数，(4) 推理时仅需指定target sparsity S（如50%或75%），系统自动按λ=α·exp(β·S)/L设置threshold。校准后sparsity偏差仅~1.2%（Table 6），远优于固定threshold的~27%偏差。校准参数a在不同数据集间表现一致（Table 12），无需per-task retuning。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
