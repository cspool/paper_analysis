## Hybrid Token Reduction (Pruning + Merging)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Token Reduction是Rethinking Token Reduction论文提出的将token pruning和token merging以特定比例组合用于同一层内token缩减的策略。对保留的相似连接对，q比例执行pruning（直接删除低重要性token），(1-q)比例执行merging（将低重要性token信息平均融合到高重要性对应token：`f_i = (a_i + f_i)/2`）。q=0.5效果最优。技术原理：pruning消除纯冗余token（相似度极高，被counterpart完全代表），merging保留有独特信息的token（通过融合保留其语义贡献）。纯pruning信息损失大，纯merging在残差上会破坏前层信息——hybrid在两者间取得最优平衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
num_prune = int(q * num_keep_pairs)
num_merge = num_keep_pairs - num_prune
# Pruning (前q比例):
for (a_i, f_i) in keep[:num_prune]:
    M_A.remove(a_i)                       # 纯删除
# Merging (后1-q比例):
for (a_i, f_i) in keep[num_prune:]:
    T[f_i] = (T[a_i] + T[f_i]) / 2       # 平均融合
    M_A.remove(a_i)
# Hidden states: q=0.5 ← 消融最优
# Residual connections: q=0 (纯merge) ← 保护残差信息
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
q值需消融优选。论文消融（Table 5）Mamba-2-2.7B @30% FLOPS：hidden q=0.5 + residual M-only → PPL 40.61, Acc 54.7%（最优）；hidden P-only + residual P-only → PPL 42.65, Acc 53.9%；hidden M-only + residual M-only → PPL 42.61, Acc 54.0%。通用原则：hidden states可用hybrid，residual branches应保守（仅merge）。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
