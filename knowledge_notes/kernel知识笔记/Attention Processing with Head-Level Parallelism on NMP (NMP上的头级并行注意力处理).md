## Attention Processing with Head-Level Parallelism on NMP (NMP上的头级并行注意力处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Processing with Head-Level Parallelism 是 Stratum 将 Transformer 的 multi-head attention 计算映射到 NMP processor 上的调度策略。由于 attention heads 之间无数据依赖，可以高度并行执行。Stratum 将 16 个 PUs 划分为多个 PU groups（通过 ring topology 上的 neighboring PUs），每个 group 负责处理一组 attention heads。两个 heads 分配给同一 group 以支持 interleaved processing（一个 head 执行 MatMul 时另一个执行 Softmax）。Key/Value 矩阵沿 sequence length 维（而非 head dim）分片到 PU group 内的各 PU，因 sequence length（512-32k）远大于 head dim（64-128），partition along sequence length 提供更好的负载均衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Processing on Stratum NMP (8 heads, 4 PU groups, 2 heads/group)
# Assumes KV cache stored in intermediate-speed Mono3D DRAM tier

# For each PU Group (e.g., PUs 0-3, heads H0, H1):

# --- Head H0 processing ---
# Step 1: Sub-ring all-gather Query Q to all PUs in group
For PU_i in group (parallel):
  Q_i = all-gather(Q_slice)    # ring network, replicate full Q

# Step 2: Score = Q @ K^T (K partitioned along seq_len dim)
For PU_i in group (parallel):
  S_i = Q @ K_i^T              # [1 × S/P], PE Tensor Core (GeMV mode)
  # S_i contains scores for sequence range [i*S/P, (i+1)*S/P)

# Step 3: Softmax (3-stage with inter-PU scalar communication)
# Stage 1: Local max
local_max_i = row_max(S_i)     # Special Function Engine
global_max = ring_scalar_exchange_max(local_max_i)  # scalar only!

# Stage 2: Local exp_sum
local_exp_i = sum(exp(S_i - global_max))  # Special Function Engine
global_sum = ring_scalar_exchange_sum(local_exp_i)

# Stage 3: Normalize
S_soft_i = exp(S_i - global_max) / global_sum

# Step 4: O = Softmax(S) @ V (V partitioned along seq_len dim)
O_i = S_soft_i @ V_i           # [1 × d_head], PE Tensor Core

# Step 5: Reduce-scatter O across PUs in group
O = reduce_scatter(O_i)        # ring network

# --- Head H1 processing (interleaved with H0) ---
# While H0 is in Softmax Stages 1-2, H1 executes Step 2 (Scores = Q@K^T)
# While H0 is in Step 4, H1 executes Softmax Stages 1-3
# H1's reduce-scatter overlaps with: nothing (H0 already done)
# But H0's reduce-scatter overlaps with H1's Step 4 (Attn@V)

# Key: Softmax's inter-PU scalar communication (2 values per PU) is negligible
# compared to tensor data transfer, enabling clean interleaving.
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Stratum 的头级并行注意力的关键实现考量：(1) PU group formation——flexible grouping 使系统能适应不同的模型架构（如 MHA, GQA, MQA），每组大小根据 head count 和 request concurrency 动态调整；(2) K/V round-robin placement——新生成的 KV pairs 按 round-robin 分布到 group 内不同 PUs，避免单个 PU 的 KV cache size 过大；(3) Scalar-only inter-PU communication——Softmax 所需的 global max/sum 仅需标量交换（每 PU 2 个值，总共 8 values per group），通过 ring network 的标量通信通道完成，latency 极小；(4) Head interleaving——2 heads per group 的设计确保一个 head 的 Softmax（低计算强度，高延迟）被另一个 head 的 MatMul（高计算强度）完全隐藏。Multi-head scheduling 的全流程由 compiler 在 offline 时根据模型配置预计算并作为静态调度嵌入 NMP 的 finite state machine。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
