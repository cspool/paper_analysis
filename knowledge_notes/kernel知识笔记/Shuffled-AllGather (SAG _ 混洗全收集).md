## Shuffled-AllGather (SAG / 混洗全收集)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shuffled-AllGather (SAG) 是 Sem-MoE 中与 SRS 配对的融合通信原语。在 MoE expert computation 完成后，将 allgather + reverse shuffle 融合为单次操作。利用 SRS 保存的 shuffle_indices，通过 argsort 计算反向排列，经 GPU tensor indexing 恢复原始 token 顺序，消除标准 EP 的 all-to-all combine。

从kernel调度角度拆解术语：

```
# SAG Kernel
Input: Y_local per device, shuffle_indices (saved from SRS)

Step 1: Ring-based AllGather
  Y_shuffled = allgather(Y_local)  # in shuffled order

Step 2: Reverse shuffle
  reverse_indices = argsort(shuffle_indices)
  Y = Y_shuffled[reverse_indices]  # restored original order

Output: Y ∈ R^{B×H}  # next layer ready
```

SRS+SAG 组合确保 token shuffling 无损——计算位置变化不影响最终结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Triton 实现，与 SRS 共用调度表。Reverse shuffle 仅需 tensor indexing。Overhead ≈ 1%。配合 DeepEP 增强回退时的 all-to-all 性能。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
