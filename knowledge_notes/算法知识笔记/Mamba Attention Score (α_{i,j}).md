## Mamba Attention Score (α_{i,j})

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba Attention Score由Ali et al. (2024)提出，是衡量Mamba SSM中不同token间信息流动强度的per-channel指标。展开Mamba的递归计算后，输出Y_i = Σ_{j=1}^i α_{i,j} ⊙ X_j，其中α_{i,j} = C_i^T (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}。与Transformer的标量attention score不同，Mamba的α_{i,j}是per-channel向量，可捕捉不同通道对不同token pair的差异化关注模式。LongMamba利用此概念进行per-channel感受野分析和全局/局部通道分类。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 推导 (Eq. 8-11):
H_i = Σ_{j=1}^i (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ⊙ X_j
Y_i = C_i^T H_i = Σ_{j=1}^i [C_i^T · (∏_{k=j+1}^i Ā_k) ⊙ B̄_j] ⊙ X_j
α_{i,j} = C_i^T · (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}

# 对比: Transformer score = q_i^T k_j (标量 per head)
#        Mamba score = α_{i,j} (向量 per-channel)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
仅用于离线分析/可视化，不参与训练或推理优化。DeciMamba也使用此概念分析Mamba的ERF和注意力稀疏性。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---
