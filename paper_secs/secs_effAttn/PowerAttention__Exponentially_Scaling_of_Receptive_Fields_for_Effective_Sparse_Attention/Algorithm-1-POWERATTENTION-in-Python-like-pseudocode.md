# Algorithm 1 POWERATTENTION in Python-like pseudocode

```
# q_idx [M, 1]: The indices of the query
# kv_idx [1, N]: The indices of the key-value
# block_size (int): The size of the CUDA block
# window_size (int): The block size of the sliding window
# build sink token mask
mask_sink = kv_idx < block_size # [1, N]
# build sliding window mask
blk_qk = q_idx // block_size - kv_idx // block_size # [M, N]
mask_window = blk_qk < window_size # [M, N]
# build PowerAttention mask
mask_power = (blk_qk & (blk_qk - 1)) == 0 # [M, N]
# ensure causality
causal = q_idx >= kv_idx # [M, N]
# combine all masks
mask = causal & (mask_window | mask_power | mask_sink) # [M, N]
```

<span id="page-4-1"></span>Table 1. Perplexity of different static sparse attention methods on PG19. Each static sparse attention pattern achieves a sparsity ratio of 0.94.

|                  | PG19  |       |       |       |  |  |  |  |  |  |  |
|------------------|-------|-------|-------|-------|--|--|--|--|--|--|--|
| Method           | 4k    | 8k    | 16k   | 32k   |  |  |  |  |  |  |  |
| Full Attention   |       |       |       |       |  |  |  |  |  |  |  |
| Vanilla          | 9.77  | 9.72  | 9.60  | 9.42  |  |  |  |  |  |  |  |
| Sparse Attention |       |       |       |       |  |  |  |  |  |  |  |
| Sliding Window   | 9.94  | 9.99  | 9.99  | 9.97  |  |  |  |  |  |  |  |
| Dilated          | 10.74 | 10.72 | 10.64 | 10.58 |  |  |  |  |  |  |  |
| Stride Slash     | 10.01 | 10.11 | 10.07 | 10.03 |  |  |  |  |  |  |  |
| LongNet          | 10.14 | 10.28 | 10.30 | 10.31 |  |  |  |  |  |  |  |
| POWERATTENTION   | 10.03 | 10.08 | 10.05 | 10.00 |  |  |  |  |  |  |  |

