# B.3 FlashMLA Implementation

```
1 @tilelang.jit
2 def flash_attn(
3 Q: T.Tensor([batch, heads, dim], dtype),
4 Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
5 KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
6 K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
7 Output: T.Tensor([batch, heads, dim], dtype),
8 ):
9 with T.Kernel(batch, heads // min(block_H, kv_group_num), threads=256) as (bx, by):
10 Q_shared = T.alloc_shared([block_H, dim], dtype)
11 S_shared = T.alloc_shared([block_H, block_N], dtype)
12 Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
13 KV_shared = T.alloc_shared([block_N, dim], dtype)
14 K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
15 O_shared = T.alloc_shared([block_H, dim], dtype)
16 acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
17 acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
18 scores_max = T.alloc_fragment([block_H], accum_dtype)
19 scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
20 scores_scale = T.alloc_fragment([block_H], accum_dtype)
21 scores_sum = T.alloc_fragment([block_H], accum_dtype)
22 logsum = T.alloc_fragment([block_H], accum_dtype)
23
24 cur_kv_head = by // (kv_group_num // block_H)
25 T.use_swizzle(10)
26
27 T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
28 T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
29 T.fill(acc_o, 0)
30 T.fill(logsum, 0)
31 T.fill(scores_max, -T.infinity(accum_dtype))
32
33 loop_range = T.ceildiv(seqlen_kv, block_N)
34 for k in T.Pipelined(loop_range, num_stages=2):
35 T.copy(KV[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
36 T.copy(K_pe[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)
37 T.clear(acc_s)
38 T.gemm(
39 Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
40 T.gemm(
41 Q_pe_shared,
42 K_pe_shared,
43 acc_s,
44 transpose_B=True,
45 policy=T.GemmWarpPolicy.FullCol)
46 T.copy(scores_max, scores_max_prev)
47 T.fill(scores_max, -T.infinity(accum_dtype))
48 T.reduce_max(acc_s, scores_max, dim=1, clear=False)
49 for i in T.Parallel(block_H):
50 scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
51 for i, j in T.Parallel(block_H, block_N):
52 acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
53 T.reduce_sum(acc_s, scores_sum, dim=1)
54 T.copy(acc_s, S_shared)
55 for i in T.Parallel(block_H):
56 logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
57 for i, j in T.Parallel(block_H, dim):
58 acc_o[i, j] *= scores_scale[i]
59 T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
60 for i, j in T.Parallel(block_H, dim):
61 acc_o[i, j] /= logsum[i]
62 T.copy(acc_o, O_shared)
63 T.copy(O_shared, Output[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])
```

Fig. 18. Implementation of FlashMLA with TileLang.