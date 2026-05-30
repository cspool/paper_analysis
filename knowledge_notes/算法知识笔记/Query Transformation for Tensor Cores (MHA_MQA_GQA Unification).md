## Query Transformation for Tensor Cores (MHA/MQA/GQA Unification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query Transformation 是 BitDecoding 中使 MHA/MQA/GQA 在 Tensor Cores 上高效执行的统一方法。Decode 阶段 Q length 仅为 1 token，M 维度极小（低 arithmetic intensity），直接 QK^T 会严重 underfill Tensor Cores。利用 GQA/MQA 的 KV head sharing 特性：GQA 下 gq = hq/hkv 个 query heads 共享同一组 KV head，将 [1, gq, hkv] 的 Q tensor reshape 为 [gq, hkv]（gq 个 query heads 被当作一个更大的 GEMM 块并行处理），饱满 Tensor Core mma fragment，提升 warp occupancy 和吞吐。

从算法pipeline角度拆解术语。

```
// 原始 decode Q layout（underfill TC）
Q: [batch=1, num_heads=hq, head_dim=d]
// QK^T: M=1×hq → small M → underfill TC tile T_m
// 对于 standard attention, batch=1, seq_q=1 → M=hq 但对于每个 KV head 仅 gq queries

// BitDecoding Query Transformation:
// 对于 GQA (gq = hq/hkv > 1):
Q_reshaped = Q.view(1, hkv, gq, d)  // 根据 KV head 分组
           = Q.view(hkv, gq, d)      // 每个 KV head 有 gq 个 queries
// 现在对于每组 KV head:
//   M_effective = gq (e.g., LLaMA-3.1-8B: gq=4, d=128 → GEMM [4,128]×[128,L])
//   Tile 填充率提升 gq× → arithmetic intensity 提升 → TC efficiency 提升

// Attention computation per KV head group:
for each KV head group i:
    Q_i = Q_reshaped[i]            // [gq, d]
    K_i = K_cache[i]               // [L, d]
    S_i = Q_i @ K_i^T / sqrt(d)    // [gq, L] — larger M → TC efficient
    A_i = softmax(S_i)
    O_i = A_i @ V_cache[i]         // [gq, d]
O = concat and reshape O_i back to [1, hq, d]
```

术语一般如何实现？如何使用？

实现在 BitDecoding 的 kernel launch 前。仅需一次 PyTorch view/reshape（零开销）。对于 MHA（gq=1），transformation 无效果；对 GQA（gq≈4-8）效果显著；对 MQA（gq=hq=32-64）效果最大。在 RTX 4090 上 BitDecoding GQA 3× speedup vs QServe 仅 1.4×——QServe CUDA Core-only 无法利用 GQA 带来的 compute intensity 提升。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
