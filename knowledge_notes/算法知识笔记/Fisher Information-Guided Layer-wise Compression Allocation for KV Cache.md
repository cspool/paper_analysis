## Fisher Information-Guided Layer-wise Compression Allocation for KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Fisher Information 引导的逐层压缩率分配继承自 Palu (2024)，通过计算每层 Key/Value 投影矩阵的 Fisher Information F(θ)=E[(∂log p(y|x;θ)/∂θ)²] 来估计层重要性。高 Fisher 层保留更多 rank，低 Fisher 层可更激进压缩，实现在固定全局 budget 下的最优分配。

从算法pipeline角度拆解术语：

```
// Fisher 计算: 在 X_calib 上前向传播
for x in X_calib:
    loss = compute_loss(model, x)
    fisher = (∂loss/∂W_kv)²  // gradient squared
// 分配: r[layer] ∝ fisher[layer] / sum(fisher)
```

ReCalKV 使用与 Palu 相同的策略，为 Key 和 Value 分别或联合分配 rank。256 WikiText-2 样本完成计算。

术语一般如何实现？如何使用？

PyTorch: `loss.backward()` + `param.grad`，128-256 样本，offline 一次性计算。不仅用于 KV 压缩，可推广到 weight pruning per-layer sparsity 分配等场景。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
