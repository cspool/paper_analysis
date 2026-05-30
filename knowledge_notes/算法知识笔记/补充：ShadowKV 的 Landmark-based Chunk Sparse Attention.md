## 补充：ShadowKV 的 Landmark-based Chunk Sparse Attention

ShadowKV 为 Chunk Sparsity of Attention 增加了 Landmark-based 的 chunk 近似方法。与 InfiniteHiP 的 per-chunk 代表 token 选择不同，ShadowKV 使用 chunk 均值作为 compressed landmark，并通过 cosine similarity 检测 outlier chunk。

```
// ShadowKV Landmark 构建（pre-filling）
K_RoPE = RoPE(K)                     // post-RoPE key
C = Reduce(K_RoPE, chunk_size=c)     // chunk mean landmarks [h_kv, s/c, d]
S = CosineSimilarity(C, K_RoPE)      // 每 chunk 内 cosine similarity
I_outlier = ArgTopK(-Min(S, dim=-1), o) // 最差近似的 o 个 chunk
L = C \ Gather(C, I_outlier)         // 非 outlier landmarks 保留 GPU

// ShadowKV Landmark 解码查询
P = Q @ L^T                          // 近似 attention scores [h_q, 1, n_c]
S = Softmax(P / sqrt(d))
S_agg = max_kv_group(sum(S, dim=-2)) // GQA 聚合到 KV heads
I = ArgTopK(S_agg, k)                // 选择 top-k chunk
```

与 InfiniteHiP chunk sparsity 对比：ShadowKV 的 landmark 是固定均值（无需 per-query 重选代表），但通过 static outlier cache 弥补均值近似的误差。Outlier 仅占 0.2-0.3% 的 chunk，保留其完整 KV 对在 GPU 保证精度。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
