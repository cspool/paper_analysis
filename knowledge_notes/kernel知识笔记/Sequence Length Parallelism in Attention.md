## Sequence Length Parallelism in Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sequence Length Parallelism in Attention（注意力中的序列长度并行）是FlashAttention-2提出的一种GPU线程块级并行策略：除batch和head维度外，额外沿sequence length维度并行化attention计算。FlashAttention v1仅将不同(head, batch)组合分配给不同thread block（即1 thread block per attention head），在长序列场景（batch size小、head数少）下thread block总数远低于GPU SM数量，导致occupancy不足。FlashAttention-2观察到forward pass的外循环（over KV column blocks）对不同row block是embarrassingly parallel（各row block独立计算其output chunk），因此将不同row block分配给不同thread block并行处理，thread block数从batch×heads增至batch×heads×T_r（row block数）。例如N=8192, B_r=128时T_r=64, 若batch=2, heads=32, 则thread blocks从64增至4096 >> A100的108 SMs, occupancy大幅提升。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Forward pass 序列长度并行调度（thread block视角）：**
```
// FlashAttention v1: 仅batch×heads并行
grid_dim = (batch_size, num_heads)  // 例如 (2, 32) = 64 thread blocks
每个thread block处理: 1个attention head的完整forward（所有row blocks串行内循环）

// FlashAttention-2: batch×heads×sequence_length并行
grid_dim = (batch_size * num_heads, T_r)  // 例如 (64, 64) = 4096 thread blocks
每个thread block处理: 1个(row_block_i, head)组合
  // Thread block (head_h, row_i):
  load Q_i from HBM                  // Q的第i个row block [B_r, d]
  for j = 1..T_c:                    // 遍历所有KV column blocks
      load K_j, V_j from HBM
      compute O_i partial update     // online softmax + matmul
  write O_i, L_i to HBM output
  // 无需与其他thread block通信！
```

**Backward pass 序列长度并行调度（thread block视角）：**
```
// FlashAttention-2 backward: 列并行
grid_dim = (batch_size * num_heads, T_c)  // 沿column blocks并行
每个thread block处理: 1个(column_block_j, head)组合
  // Thread block (head_h, col_j):
  load K_j, V_j from HBM
  for i = 1..T_r:                    // 遍历所有row blocks
      load Q_i, O_i, dO_i, L_i, D_i from HBM
      recompute S_ij = Q_i @ K_j.T   // 在SRAM中重计算
      recompute P_ij = exp(S_ij - L_i)
      accumulate dV_j += P_ij.T @ dO_i
      accumulate dK_j += dS_ij.T @ Q_i
      dQ_i += dS_ij @ K_j            // atomicAdd to HBM! (跨thread block通信)
  write dK_j, dV_j to HBM
```
注意反向dQ需要跨thread block的atomicAdd，因为多个column blocks (j)的thread blocks同时更新同一个dQ_i。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashAttention-2通过CUDA kernel launch configuration实现：(1) 前向用2D grid `dim3(batch*heads, T_r)`，每个block的blockIdx.y标识row block；(2) 后向用2D grid `dim3(batch*heads, T_c)`，每个block的blockIdx.y标识column block。K/V block加载在单个thread block内串行（前向row parallel）或Q/O/dO block加载在单个thread block内串行（后向column parallel）。解码阶段采用不同策略：由于query length=1，bottleneck变为KV cache加载速度，因此将KV cache分片到不同thread block并行加载（split KV cache loading across thread blocks）以saturate HBM bandwidth，中间结果写入HBM后通过separate reduce kernel合并。

涉及论文标题：
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
