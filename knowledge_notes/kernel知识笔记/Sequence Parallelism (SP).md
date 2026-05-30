## Sequence Parallelism (SP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sequence Parallelism (SP, Korthikanti et al., 2023) 是 Megatron-LM 对 Tensor Parallelism 的扩展。在 TP 中，attention 层的 LayerNorm 和 Dropout 操作在每个 GPU 上对完整的 sequence 副本执行，浪费内存。SP 将 sequence 维度沿 TP group 切分，使 LayerNorm 和 Dropout 只操作部分 token（而非完整序列），减少激活内存。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# TP+SP 组合 (2-way)
# Transformer block forward:
GPU0 (seq half 0):   GPU1 (seq half 1):
  LN(X[0:N/2])          LN(X[N/2:N])          # SP: seq 切分, 各算一半
  Dropout(...)           Dropout(...)           # SP: seq 切分
  # gather full seq for attention (通信)
  X_full = all_gather(X_half)  ←→  all_gather
  # Attention with TP (算子切分)
  Z_half = TP_Attention(X_full)                 # TP: 算子切分
  # reduce-scatter for next SP
  Z_half = reduce_scatter(Z_full)
  LN(Z_half)             LN(Z_half)             # SP: seq 切分
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FOLDMOE 中 attention 层使用 TP+SP=8（intra-node）
- SP 仅作用于 token-wise 操作（LayerNorm、Dropout），不作用于 attention 计算和 MoE 计算
- 这意味着 SP 的 sequence 切分不影响 FOLDMOE 的 attention-MoE pipelining 数据完整性
- SP 减少激活内存，使 FOLDMOE 能在更长序列或更大模型上运行

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
