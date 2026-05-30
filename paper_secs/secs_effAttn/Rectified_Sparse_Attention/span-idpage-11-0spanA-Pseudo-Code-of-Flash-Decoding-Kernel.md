# <span id="page-11-0"></span>A Pseudo Code of Flash Decoding Kernel

The proposed group block sparse attention (Section [2.1\)](#page-1-0) can be easily integrated into the Flash Decoding [\[6\]](#page-9-12) kernel implementation. The modified parts are highlighted as follows.

```
Algorithm 2 Flash Decoding with Block-Sparse Attention
Require: Queries Q, Keys K, Values V , block_indices
Ensure: Attention outputs Outpartial, logsumpartial, Out
 1: for Grid indexed by (num_splits, num_kv_heads, batch_size) do
 2: Load query vectors q in a GQA group
 3: Compute partial_block_indices with block_indices and num_splits
 4: Initialize accumulators: mi ← −∞, li ← 1.0, acc ← 0
 5: for block_id in partial_block_indices do
 6: Load keys k and values v from KV cache in block block_id
 7: Compute scaled attention scores qk ← (qk) × sm_scale
 8: Apply masking to invalid positions (qk ← −1e6)
 9: Compute and update mi
                               , li
                                 , acc
10: end for
11: Store partial logsum and attention outputs into logsumpartial, Outpartial
12: end for
13: Combine different splits Combine(logsumpartial, Outpartial, Out)
14: return Attention output tensor Out
```

