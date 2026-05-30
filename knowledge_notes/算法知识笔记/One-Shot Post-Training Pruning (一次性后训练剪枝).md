## One-Shot Post-Training Pruning (一次性后训练剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Shot Post-Training Pruning 是一类无需重训练的模型压缩方法：在预训练模型上用少量校准数据（128 条 C4）一次前向计算权值重要性，生成并应用稀疏 mask，不更新权重。代表方法：SparseGPT（Hessian 逆+权重更新）、Wanda（仅激活范数）、MoE-Pruner（router 感知激活范数）。

从算法pipeline角度拆解术语：
```
X = sample(C4, 128)
for layer in model.layers:
    X = forward_until(layer, X)
    S = pruning_metric(W, X)       # Wanda: |W|*∥X∥; MoE-Pruner: |W|*∥X*Gate∥
    mask = row_topk_mask(S, p%)    # 每输出神经元保留 top-(1-p%)
    W = W * mask                   # 无权重更新
    X = forward_layer(layer, X)
```
MoE-Pruner 的 one-shot 特性：无需 retraining/weight update，O(d_hidden²) 复杂度，128 seqs 校准。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 单卡 H100 数分钟至数十分钟。MoE-Pruner 剪枝后用 Expert-wise KD 恢复（1小时/1000样本）。局限：对校准数据质和量敏感；非结构化稀疏无硬件加速；剪枝决策不可逆。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
