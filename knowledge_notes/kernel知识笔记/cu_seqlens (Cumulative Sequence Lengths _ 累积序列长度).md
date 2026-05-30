## cu_seqlens (Cumulative Sequence Lengths / 累积序列长度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cu_seqlens是Mamba库（https://github.com/state-spaces/mamba）中用于处理变长序列的关键机制，类似Flash Attention的`varlen_fwd`/`varlen_bwd` API。它将多个不同长度的序列pack到单个flattened tensor中（batch_size=1），用cumulative sequence lengths数组（shape=[num_seqs+1]）标记每个序列的起始和结束位置。例如batch中有3个序列长度分别为5, 10, 3，则cu_seqlens=[0, 5, 15, 18]。Mamba的selective scan CUDA kernel在cu_seqlens定义的每个序列边界处自动重置hidden state，确保不同序列间的SSM state不交叉污染。相比传统的padding方式，cu_seqlens避免了浪费在padding token上的计算，在真实数据集上可获2-4x端到端加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
cu_seqlens在Mamba selective scan kernel中的使用：
```
// Mamba selective scan kernel with cu_seqlens support
Input: x ∈ R^{total_tokens×D}    // flattened batch
       cu_seqlens ∈ Z^{B+1}       // cumulative lengths

for batch_idx = 0 to B-1:
  seq_start = cu_seqlens[batch_idx]
  seq_end = cu_seqlens[batch_idx + 1]

  // Reset hidden state at each sequence boundary
  h = zeros(D, N_state)

  // Process only within this sequence
  for t = seq_start to seq_end-1:
    h = A[t] * h + B[t] * x[t]^T
    y[t] = C[t] @ h
```

在Attamba中的用法：Attamba利用cu_seqlens处理变长chunk——不同chunk可能大小不同（如Random chunking），cu_seqlens指定每个chunk的起止位置，SSM在每个chunk边界重置hidden state。Cyclic chunking中不同层的不同chunk边界也通过修改cu_seqlens偏移量实现。整个过程中不需要reshape或padding输入序列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba官方实现内建cu_seqlens支持（GitHub PR #244）。使用时传入cu_seqlens参数到Mamba block的forward函数。注意事项：(1) 仅支持训练，推理/生成场景尚未支持；(2) Bidirectional Mamba需要额外的reverse_cu_seqlens；(3) Mamba-2同样支持cu_seqlens用于序列并行。Attamba利用cu_seqlens实现了无需reshape的灵活chunking，是降低实现复杂度的关键设计选择。

涉及论文标题：
- Attamba__Attending_To_Multi-Token_States

---
