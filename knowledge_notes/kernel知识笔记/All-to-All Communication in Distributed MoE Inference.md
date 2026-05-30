## All-to-All Communication in Distributed MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

All-to-All Communication 是分布式 MoE expert parallelism (EP) 的核心集合通信操作。每 MoE 层需要两轮 All-to-All：(1) Dispatch——将各 GPU 的 tokens 按 gate 选择的 expert index 发送到 hosting GPU；(2) Combine——将 expert 计算完成的输出 tokens 返回原 GPU。通信量与 token 数、hidden dimension、EP degree 成正比。GRACE-MoE 实测 multi-node 场景中 cross-node All-to-All 占单 MoE 层执行时间的 70%+，端到端延迟约 40%。Flat global All-to-All 在 heterogeneous 链路（NVLink + Ethernet）中产生 straggler effect——所有 ranks 等待最慢 link，amplifying synchronization overhead。

从kernel调度角度拆解：

```
# Flat All-to-All (NCCL alltoallv, blocking):
sent = alltoall_dispatch(local_tokens, counts_per_rank)  # GPU SM idle
expert_out = fused_moe(sent, local_experts)
combined = alltoall_combine(expert_out, counts_per_rank) # GPU SM idle

# GRACE-MoE HSC (hierarchical + sparse):
# Stage 1: cross-node sparse P2P (global group, zero-padded)
#   GPU aggregates tokens by dest node → single send
# Stage 2: intra-node NVLink redistribution (overlapped w/ routing comp)
# Combine: symmetric reverse

# 通信时间模型:
# T_flat ≈ 2 × n_token × d_model / BW_cross_node
# T_HSC ≈ n_unique_dest × d_model / BW_cross_node
#        + max(0, n_token × d_model / BW_intra_node - T_routing)
```

术语一般如何实现？如何使用？

- NCCL AlltoAll/AlltoAllv 是 PyTorch distributed 的底层实现（ring/tree topology）
- MoE 专用库：DeepEP（NVSHMEM, one-sided put/get + warp specialization）、FUSCO（transformation-communication fusion）
- EP degree 通常 = 节点数（节点内 GPU 分配给其他并行维度）
- GRACE-MoE HSC 在 Megablocks 上用 NCCL global group + manual zero-padding 实现 logical sparsity
- 优化原则：minimize cross-node traffic（dedup, affinity grouping），maximize intra-node BW utilization

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
