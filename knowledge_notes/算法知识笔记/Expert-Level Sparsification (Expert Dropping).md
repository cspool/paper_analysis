## Expert-Level Sparsification (Expert Dropping)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Level Sparsification（专家级稀疏化，亦称 Expert Dropping/Expert Pruning）是专门针对 SMoE 架构的模型压缩技术，通过识别和移除整个 expert 子网络（包括其权重和 router entry）来减少模型总参数量和内存占用。核心理念：SMoE 中存在显著的 expert redundancy——部分 expert 对模型性能至关重要（dominant experts），而另一些则高度冗余。Jaiswal et al. (2025) 的实验显示 Mixtral-8×7B 中某些 expert 被单独丢弃后 perplexity 急剧上升，另一些几乎无影响。与传统 weight pruning 不同，Expert Dropping 丢弃的是整个结构化计算单元（FFN expert），不改变剩余 expert 的内部参数结构，因此可直接获得实际推理加速（50% sparsity → 1.27× speedup, ≤0.55× memory usage）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Dropping Pipeline
def expert_drop_per_layer(M, l, s, criterion, X_calib):
    """
    M: SMoE model, l: layer index, s: droprate
    criterion: MC-Suite criterion function
    """
    scores = [criterion(M, l, e, X_calib) for e in range(n)]
    n_drop = int(n * s)
    drop_set = argsort(scores)[:n_drop]  # lowest importance
    
    # Remove from router: W_G^{d×n} → W_G^{d×(n-n_drop)}
    keep_mask = ones(n, dtype=bool)
    keep_mask[drop_set] = False
    M.layers[l].router.W_G = M.layers[l].router.W_G[:, keep_mask]
    for e in drop_set:
        del M.layers[l].experts[e]
```

**与 Weight Pruning 的对比** (Mixtral-8×7B @ 50% sparsity, zero-shot avg):
- Random Weight Pruning (2:4): 27.27 (Base) / 31.94 (Instruct)
- Wanda Weight Pruning (2:4): 52.91 (Base) / 62.28 (Instruct)
- Min-EAN Expert Pruning (r=4): 56.62 (Base) / 63.95 (Instruct)
- Full Mixtral (r=8): 70.32 (Base) / 76.31 (Instruct)

Expert-level sparsification 整体优于 weight pruning，尤其在 ARC-c 上提升 ~16.2%。Base model 上的表现优于 Instruct model（建议在 instruction tuning 前做 expert dropping）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Per-layer uniform dropping：每层丢弃相同比例 expert 以避免 bottleneck layers
- 必须同步修改 router gating 函数（删除对应 expert 入口），否则 router 可能将 token 路由到已删除 expert
- 丢弃后 router 矩阵直接修改导致负载分布偏斜，需要 finetuning 或 load rebalancing 校正
- 相关方法：REAP (Lasby et al., 2025) 用 router gate-value × activation norm；DERN (Zhou et al., 2025) 重组合 pruned expert 的神经元到 retained expert

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
