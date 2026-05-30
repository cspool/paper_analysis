## Tensor Index Slicing (Index Select) for MoE Dispatch（张量索引切片MoE调度）

术语是什么？
Tensor Index Slicing（index_select）是 PPMoE 中替代 all-to-all 的 token dispatch 机制。由于 PPMoE 将所有 experts 放置在同一 TP group（同一节点）内，且各 TP rank 持有相同的 hidden states（经 TP all-reduce 同步）和相同的 dispatching order（相同的 gating 输入），dispatch 仅需本地 index_select 操作——无通信开销。这是 PPMoE 消除 all-to-all 的关键技术。

从kernel调度角度拆解术语：
```
// PPMoE dispatch: index_select (替代 all-to-all)
// 输入: hidden_states [8, h/T], indices [8]（dispatching order）
// indices = [2, 3, 1, 2, 0, 3, 2, 0]
// N=4 experts on this TP rank (E=64, T=8)

// 按 expert 分组 token
X0 = hidden_states[[4, 7], ...]   // expert 0: tokens 4,7
X1 = hidden_states[[2], ...]      // expert 1: token 2
X2 = hidden_states[[0, 3, 6], ...] // expert 2: tokens 0,3,6
X3 = hidden_states[[1, 5], ...]   // expert 3: tokens 1,5

// 然后串行执行各 expert FFN
```

对比 DPMoE 的 all-to-all dispatch（跨节点传输 b*s*h 数据，走 InfiniBand 12.5 GB/s），PPMoE 的 index_select 是本地 PyTorch 操作，零通信，利用 NVLink 300 GB/s 的 all-reduce 替代 InfiniBand all-to-all 完成 gather。

术语一般如何实现？如何使用？
PyTorch 的 `torch.index_select` 或高级索引 `tensor[indices]` 实现。要求：(1) 所有 experts 在同一节点内；(2) 各 TP rank 持有相同输入（通过 TP 的 copy_to_tensor_parallel_region 保证）；(3) gating network 的参数在 TP ranks 间需同步（仅 h×E 大小，可忽略）。适用场景：PPMoE 或任何将 experts 集中在单节点内的 MoE 并行架构。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
