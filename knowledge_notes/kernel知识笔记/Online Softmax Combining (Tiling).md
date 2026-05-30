## Online Softmax Combining (Tiling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online Softmax Combining 是一种将多个 partial attention 输出（各自计算了不同 KV subset 上的 softmax attention）合并为等价于完整 attention 输出的数值方法。核心原理：利用 online softmax 的 log-sum-exp (lse) 状态，对不同 attention 分片的 output 和 lse 进行 re-weight 和 re-scale 合并。

MoBA 中需要此技术的原因是：一个 query 可能同时关注当前 block（self-attention, causal=True）和多个历史 blocks（MoBA attention, causal=False），这两个 attention 计算分别在 FlashAttention varlen 中执行，产生两个 partial outputs O^s 和 O^m。需要通过 online softmax combining 将它们合并为数学上等价于对全部 (k+1)B 个 token 做一次统一 attention 的结果。

从kernel调度角度拆解术语：
```
输入：O^s, lse_s (self-attention output + log-sum-exp)
      O^m, lse_m (MoBA attention output + log-sum-exp)
输出：O (等价于 unified attention over all blocks)

# Step 1: Compute total lse
lse_total = max(lse_s, lse_m) + log(exp(lse_s - max) + exp(lse_m - max))

# Step 2: Re-weight and combine
w_s = exp(lse_s - lse_total)  # weight for self-attention
w_m = exp(lse_m - lse_total)  # weight for MoBA attention
O = w_s · O^s + w_m · O^m

# In practice: tiled implementation without explicit lse materialization
# Each tile computes partial lse, passes to next tile via online rescaling
```

与 FlashAttention 的 online softmax 关系：FlashAttention 在单个 kernel 内使用 online softmax 对不同 K/V tiles 进行累加；MoBA 将其扩展到跨 kernel 合并——self-attention kernel 和 MoBA attention kernel 各自返回 output + lse，然后在 combine kernel 中合并。

术语一般如何实现？如何使用？

在 MoBA 中实现为 `combine_with_online_softmax(O^s, O^m)` 函数，基于 Milakov et al. (2018) 和 Liu et al. (2023) 的 online normalizer calculation 方法。实现为一个轻量 CUDA kernel，接受两个 (output, lse) pairs，输出合并后的最终 attention output。该技术也广泛应用于 speculative decoding（合并 draft model 和 target model 的 attention output）和 sequence parallelism（合并不同 sequence chunk 的 partial attention）。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs
