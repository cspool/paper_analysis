## Hierarchical Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Token Reduction是一种跨模型层的渐进式token缩减策略——不在每层都执行reduction，而是每隔固定层数（如5层）执行一次。Rethinking Token Reduction论文对Mamba-2-2.7B在[12,17,22,27,32,37,42]层执行。动机：(1) 相邻层token重要性分布相似，每层reduction冗余；(2) 浅层representation不成熟，不适合过早reduction；(3) 更早层reduction产生更大累积FLOPs节省。消融（Table 4）：[12,17,...]配置PPL 17.96/Acc 58.7%优于更深起始[20,25,...]的PPL 18.88/Acc 57.8%，证明适中提早策略最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Hierarchical Schedule:
config = {
    "Mamba-2-2.7B/2.8B": [12, 17, 22, 27, 32, 37, 42],
    "Mamba-2-1.3B/1.4B": [10, 15, 20, 25, 30, 35],
}

for layer_id in model.layers:
    x = forward_pre_ssm(layer, x)
    if layer_id in reduction_layers:
        x = reduce_tokens(x, r)            # Token数递减
    x = forward_post_ssm(layer, x)

# Per-layer vs Hierarchical:
# Per-layer:   每层reduction → 高开销、冗余计算
# Hierarchical: 每5层一次 → 低开销、累积效果好
# 起始层选择: 1/4~1/3总层数处, 间隔≈5层
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为层索引检查条件，reduction_layers通过消融grid search确定。通用原则：起始层≈总层数1/4处，间隔≈5层，最后一层不做reduction。适用于任何多层Transformer/SSM架构的token reduction场景。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
