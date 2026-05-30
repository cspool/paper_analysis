## Intra-layer Token Reduction (Hidden States + Residual Connections)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intra-layer Token Reduction是指在同一模型层内对多个计算分支（hidden states和residual connections）分别应用不同token reduction策略的设计。Rethinking Token Reduction发现：(1) hidden states承载SSM处理后新信息，适合hybrid（pruning+merging）；(2) residual传递前层原始信息，只能merging不能pruning（pruning永久丢失残差信号）。关键问题：若hidden和residual以不同方式/不同步调reduction，重组时出现"index misalignment"（hidden中删除token在residual仍存在，维度不匹配）。UTRC通过统一M_A/M_B分类解决此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
y = SSM(A, B, C)(x)                    # hidden states
residual = T_{l-1}                      # residual

M_B, M_A = importance_classify(y)      # 共享分类

# Hidden: Hybrid (q=0.5)
y_reduced = utr_hybrid(y, M_B, M_A, q=0.5)
# Residual: Merge-only
res_reduced = utr_merge_only(residual, M_B, M_A)

output = proj(y_reduced) + res_reduced  # 维度一致!
# M_A在两个分支中被同步删除以保证alignment
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为Mamba block hook中的并行reduction路径，共享M_A/M_B索引保证dimension一致性。通用化原则：任何具有多计算分支的block（如Transformer的attention+MLP分支），hidden/states分支和residual分支需要解耦reduction策略以保证对齐。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
