## Data Parallelism (DP / 数据并行)

术语是什么？
Data Parallelism 是最基础且最广泛使用的分布式训练并行技术。每个设备持有完整模型副本，输入数据按 batch 维度切分（micro-batches）分配到各设备，各自独立执行 forward+backward，完成后通过 all-reduce 同步梯度。DP 不切分模型参数，仅切分数据，因此每设备需能容纳完整模型。

从kernel调度角度拆解术语：
```
// DP 训练一个 iteration（D 个 DP ranks）
每个 rank r 独立执行:
    loss_r = model.forward(micro_batch_r)
    loss_r.backward()        // 计算本地梯度
// 梯度同步
all_reduce(gradients)        // 所有 ranks 的梯度求平均
// 优化器更新
optimizer.step()             // 各 rank 独立更新（梯度已同步 → 参数一致）
```

在 DPMoE 中，DP 与 EP 绑定——每 DP rank 持有 E/D 个 experts，EP 的 all-to-all 通信发生在 DP ranks 之间的 MoE layers。在 PPMoE 中，DP 与 EP 解耦——DP 仅用于扩展 global batch size，不影响 expert 分布。

术语一般如何实现？如何使用？
PyTorch DDP (DistributedDataParallel) 或 DeepSpeed ZeRO。PPMoE 实验中，Dense/backbone 使用 DP 扩展（DP=32 时 5120 tok/s/GPU），但 PPMoE 因 PP 已提供足够的 batch scaling 而省略 DP（DP=1）。ZeRO optimizer 可与 DP 结合使用以降低 per-rank 内存占用。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
