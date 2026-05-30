## FSDP (Fully Sharded Data Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FSDP (Fully Sharded Data Parallelism) 是 PyTorch 分布式训练中的一种 Data Parallelism 变体（源自 ZeRO-3 的思想），其核心是将模型参数、梯度和优化器状态完全分片（shard）到所有 data parallel device 上，而非每个 device 持有完整副本。训练时，参数按需通过 AllGather 物化（materialize），用后立即释放；梯度通过 ReduceScatter reduce 后各 device 更新其 shard；优化器状态也仅维护 shard 部分。FSDP 显著减少了每 device 的内存占用（参数量/DP_degree），使得训练超大 dense 模型成为可能。

然而，当直接应用于 MoE layer 时，FSDP 变得极低效：一个包含 |ℰ| 个 expert 的 MoE layer 会产生 |ℰ|× 的通信开销（因为每个 expert 都需要 AllGather），难以与计算重叠。Hecate 的 FSSDP 受 FSDP 启发，但以不同粒度分片 MoE layer（以 expert 而非参数矩阵为分片单位），并用稀疏通信原语 (SparseAllGather/SparseReduceScatter) 替代 AllGather/ReduceScatter，使通信量降为 O(λS)（λ << 1 时远小于 FSDP 的 O(S)）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FSDP 在 dense model 单层中的执行流程：

```
FSDP Forward Pass (single dense layer):
  AllGather(layer_params)   // 从所有 DP ranks 收集参数分片
                             // 通信量: O(S) (S = 总参数大小)
  y = layer(x)              // 前向计算
  discard(layer_params)     // 立即释放非本 rank 的参数分片

FSDP Backward Pass:
  AllGather(layer_params)   // 重新物化参数 (backward 需要)
                             // 通信量: O(S)
  grad = layer.backward(y)  // 反向计算
  discard(layer_params)
  ReduceScatter(grad)       // 将梯度 reduce 并 scatter 到各 rank
                             // 通信量: O(S)

总通信量: 3 × O(S) per layer
重叠策略: AllGather/ReduceScatter 可与前一层的计算重叠
```

FSDP 在 MoE layer 上的问题：
- MoE layer 有 |ℰ| 个 expert × 3 FFN matrices (W_gate, W_up, W_down)
- FSDP 将每个 expert 视为独立参数块 → |ℰ|× 通信开销 vs 1 个 dense FFN
- 通信量过大，难以与 computation 重叠

FSSDP 的解决方案：
- 以 expert（非参数矩阵）为 sharding 单位
- SparseAllGather 仅物化当前 placement 需要的 expert 子集 → O(λS)，λ << 1
- 通信可与 Attention computation 重叠（不在 MoE 层内）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch FSDP 官方实现：`torch.distributed.fsdp` 模块，通过 `FullyShardedDataParallel` wrapper 包裹模型。
- 支持 mixed precision、activation checkpointing、CPU offloading 等优化。
- 在 Hecate 中，FSDP 的思想被创新性应用于 MoE training，通过 FSSDP 实现类似的内存节省效果（全局仅一份 optimizer states）但避免了 FSDP 在 MoE layer 上的通信爆炸问题。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (Zhao et al., VLDB 2023)
