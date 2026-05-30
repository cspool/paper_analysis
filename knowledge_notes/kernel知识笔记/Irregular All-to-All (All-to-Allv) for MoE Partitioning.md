## Irregular All-to-All (All-to-Allv) for MoE Partitioning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Irregular All-to-All（MPI 术语中称为 All-to-Allv）是 MoE 分布式训练中的通信模式，其中每个设备向不同目标设备发送不同数量的数据（非均匀 partition）。在 Lancet 的算子分区方案中，input batch 沿 batch 维度分区为 micro-batch，每个 micro-batch 经过 gating 后，向某个 expert 发送的 token 数从 0 到 C（expert capacity）不等，但所有 partition 的总 token 数之和等于 C。这种不规则性源于：special gating operator 在 partition 间传递容量信息——当第一个 partition 使用 3/4 C 容量时，后续 partition 动态调整 remaining capacity 为 1/4 C。Lancet 使用双趟 All-to-All 实现：第一趟交换各 GPU 间实际传输的 data size，第二趟按已知 size 传输实际数据。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Lancet 的 Irregular All-to-All 实现（论文 Fig. 10）：

```
# 设 G 个 GPU 参与 all-to-all, 每 GPU 分配 E^l 个 expert (G = E/E^l)
# Input Buffer: [G, C] 固定形状, 仅部分填充
# Output Buffer: [G, C] 固定形状

def irregular_all_to_all(tokens_by_expert, gating_result):
    # tokens_by_expert[g][e]: GPU g 上属于 expert e 的 tokens (variable size)
    # gating_result 包含每个 (src_gpu, dst_expert) 的 token count
    
    # Phase 1: 交换 data sizes
    # send_sizes[g] = [count_0, count_1, ..., count_{G-1}]  (发给每个 GPU 的 token 数)
    send_sizes = compute_send_sizes(tokens_by_expert)
    recv_sizes = all_to_all_sizes(send_sizes)  # 第一趟 all-to-all: 只交换 size 信息
    
    # Phase 2: 传输实际数据
    # 基于 recv_sizes 知道从每个 src 收多少数据
    # 基于 send_sizes 知道向每个 dst 发多少数据
    for dst_gpu in range(G):
        if send_sizes[dst_gpu] > 0:
            ncclSend(tokens_buffer[dst_gpu], size=send_sizes[dst_gpu], target=dst_gpu)
    for src_gpu in range(G):
        if recv_sizes[src_gpu] > 0:
            ncclRecv(output_buffer[src_gpu], size=recv_sizes[src_gpu], source=src_gpu)
    
    return received_tokens
```

与 Uniform All-to-All 的对比：

```
# Uniform: 每 GPU 向每 GPU 发送固定 C 个 token
# 总通信量: G * G * C * token_size

# Irregular: 每 GPU 向 GPU g 发送 s_g 个 token, Σ s_g = G*C
# 总通信量: G * C * token_size (实际数据) + G*G*sizeof(int) (size info)
# 不规则的总通信量更低（不传输 padding tokens）
```

Pipeline 调度中的不规则 All-to-All：

```
# Pipeline Stage 中的 3-partition 例子:
# Partition 0: NonMoE₀ → [IrregA2A₀ out] → ...
# Partition 1:        NonMoE₁ → [IrregA2A₁ out] → ...
# Partition 2:               NonMoE₂ → [IrregA2A₂ out] → ...

# IrregA2Aₖ 的通信量取决于 gating output 在 partition k 中的 token 分布
# PipelineScheduler 使用 static-shape approximation (C/k) 预估时间
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lancet 基于 NCCL Send/Recv primitives 实现（grouped communication），不使用 `ncclAllToAll`。Input/Output buffer 按最大容量（C×G）静态分配，运行时仅部分填充。该实现不传输 padding tokens，因此总通信量可低于 uniform all-to-all。类似的 irregular all-to-all 实现在 DeepSpeed-MoE、FasterMoE、Lina、Tutel 等系统中也有出现，各自有不同的优化策略（如 Tutel 的 2D-Hierarchical All-to-All 利用 NVLink intra-node 和 network inter-node 的分层拓扑）。Lancet 的 static-shape approximation 虽不能精确预测不规则 all-to-all 的绝对时间，但误差仅 3.83%，足够引导 DP 搜索选择正确的 partition range。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
