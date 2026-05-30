## All-Reduce Communication in Distributed Training（分布式训练中的全归约通信）

术语是什么？
All-Reduce 是分布式训练中最核心的集合通信原语之一。将各设备上的数据（通常是梯度或 hidden states）求和/平均后广播到所有设备。在 TP 中用于同步 FFN/attention 的输出和梯度；在 DP 中用于同步梯度。All-Reduce 的通信复杂度为 O(2(N-1)/N · data_size)，在 ring 算法下每设备收/发 2(N-1)·data_size/N 数据。节点内 all-reduce（NVLink）带宽远高于节点间（InfiniBand），因此 TP 的 all-reduce 开销远低于 EP 的 all-to-all。

从kernel调度角度拆解术语：
```
// Ring All-Reduce (N 个设备)
// 分两步：Reduce-Scatter + All-Gather
每个设备 data = local_tensor [size M]

// Step 1: Reduce-Scatter (N-1 步)
for step in 1..N-1:
    send chunk[(rank-step)%N] to (rank+1)%N
    recv chunk[(rank-step-1)%N] from (rank-1+N)%N
    reduce recv_chunk into local accumulator

// Step 2: All-Gather (N-1 步)
for step in 1..N-1:
    send reduced_chunk to (rank+1)%N
    recv chunk from (rank-1+N)%N
```

在 PPMoE 中，MoE 层的 all-reduce 通信量 = 2×b×s×h per global batch（与 TP FFN 完全相同），走 NVLink (300 GB/s)，远低于 DPMoE 的 all-to-all（走 InfiniBand 12.5 GB/s）。PPMoE 将 MoE all-reduce 时间降至仅比 FFN all-reduce 多 1.9% of total forward time。

术语一般如何实现？如何使用？
NCCL (NVIDIA Collective Communications Library) 提供 `ncclAllReduce`，自动选择最优算法（ring/tree/collnet）根据拓扑。PyTorch 通过 `torch.distributed.all_reduce` 调用。PPMoE 中 MoE 层的 all-reduce 通过 Megatron 的 `reduce_from_tensor_parallel_region` 封装。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
