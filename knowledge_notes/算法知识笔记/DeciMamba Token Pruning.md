## DeciMamba Token Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DeciMamba（Ben-Kish et al., 2024, ICLR 2025）是首个探索Mamba上下文长度外推的training-free方法。核心机制：在Mamba深层（如第12层）利用Δ_t值作为token重要性度量，仅保留top-k个平均Δ_t最大的token进行后续处理。关键发现：Mamba隐含学习了与训练长度绑定的有效感受野（ERF），当序列超过训练长度时隐藏注意力矩阵变稀疏并在~10K token后崩溃。DeciMamba通过减少深层序列长度缓解这一问题。与LongMamba的per-channel区分策略不同，DeciMamba对所有通道统一prune token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
For layer l in decimating_layers (e.g., [12]):
    importance[t] = mean(Δ_t)  # 沿d_e维度
    k = min(max(decimation_min_seq_len, base), current_seq_len)
    kept_indices = topk(importance, k)
    X_l = X_l[kept_indices]    # 仅保留重要token继续

# 典型配置: decimation_beta=0.5, decimating_layers=[12],
#            decimation_min_seq_len=20, decimation_max_p_L_base=2000
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/assafbk/DeciMamba。LongMamba实验表明DeciMamba在PG-19 60k tokens上perplexity仍>30（vs LongMamba <20），因为无差别pruning在所有通道上丢弃token降低了局部上下文建模能力。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement
- Rethinking_Token_Reduction_for_State_Space_Models

---
