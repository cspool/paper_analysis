## MoveCache Algorithm for Paged Block Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MoveCache 是 KV-Compress 提出的在 paged KV cache 中执行 block-level eviction 后重排物理 cache 的算法。在 paged attention 中，KVs 以固定大小 block（b=16）为单位存储。当某些 block 被选中 eviction 时，block 内的 KVs 需要在物理 cache 中重新排列，使得被 evicted 的 KVs 在物理上连续，从而整块释放。

核心问题：在 variable-head-rate eviction 下，跨 head 选择的 eviction candidates 分散在不同 block 中。简单将每个 block 中部分 KVs 标记为 evicted 无法释放任何 block（因为每个 block 仍包含至少一个 non-evicted KV）。MoveCache 通过反序遍历 eviction range，将 range 内的 non-evicted KVs 与 range 外的 evicted KVs 交换，使 eviction range 内所有 KVs 均为 evicted，可整块释放。

从kernel调度角度拆解术语：

**MoveCache 算法（Algorithm 1 伪代码）**：
```
输入：K_u, V_u ∈ R^{N×b×d}（physical unified cache）
      M ∈ R^{N×b}（eviction metrics, same layout）
      P ∈ R^{N×b}（logical indices, initially = token position）
      W ∈ {0,1}^{N×b}（eviction mask, 1=evict, 0=keep）
      E_s（eviction block count for sequence s）
      b（block size）

1:  eviction_range_start = end - E_s * b  # eviction range: last E_s blocks
2:  i = end - 1    # pointer: scan from end of eviction range backwards
3:  j = end - 1    # pointer: scan from end of cache backwards
4:  while i >= eviction_range_start:
5:      while W[i] == 0 and i >= eviction_range_start:
6:          i -= 1  # skip: KV in eviction range but NOT evicted
7:      while W[j] == 1 and j >= 0:
8:          j -= 1  # skip: KV outside eviction range but evicted
9:      # swap non-evicted KV (at logical position P[i]) with evicted KV (at P[j])
10:     swap K_u[P[i]], K_u[P[j]]
11:     swap V_u[P[i]], V_u[P[j]]
12:     swap M[P[i]], M[P[j]]
13:     swap P[i], P[j]  # update logical indices
14:     i -= 1
15:     j -= 1
16: # After loop: all KVs in eviction_range_start..end are W=1
17: # Free blocks in eviction range
```

**执行trace 示例（2 heads, b=2, evict 2 blocks, 简化）**：
```
Initial state (shown as [head: KV_metric]):
  Block 0: [h0: 0.8] [h0: 0.3]  → KVs with metrics 0.8 and 0.3
  Block 1: [h1: 0.9] [h0: 0.1]  → mixed heads
  Block 2: [h0: 0.2] [h1: 0.5]
  Block 3: [h1: 0.7] [h0: 0.0]  → last 2 blocks = eviction range

Sort by metric → mark lowest 4 KVs (E_s=2 blocks, 4 KVs) for eviction:
  Evicted: h0:0.0, h0:0.1, h0:0.2, h1:0.5
  Kept:    h0:0.3, h1:0.7, h0:0.8, h1:0.9

After MoveCache reordering:
  Eviction range (last 2 blocks): contains ONLY evicted KVs
  Non-eviction range (first 2 blocks): contains ONLY kept KVs
  → Free last 2 blocks
```

术语一般如何实现？如何使用？

MoveCache 在 GPU 上运行，通过 PyTorch 的 scatter/gather 或自定义 CUDA kernel 实现。KV-Compress 使用 PyTorch sort 和 indexing 操作完成重排。主要开销是 sort（额外内存 ~8× 排序 tensor 大小，在 1.7e8 元素后 runtime 线性增长）。

MoveCache 仅在压缩 iteration 时执行（prefill 后 + preemption 即将发生时），不是每个 forward pass 都运行。与 GPU block manager 协调——MoveCache 释放的 blocks 通过 GPU block manager 标记为 free。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
