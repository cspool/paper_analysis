## Query Transformation for Tensor Cores (MHA/MQA/GQA Unification)

术语是什么？通过联网搜索让回答具体和精准。

Query Transformation 是 BitDecoding 中使 MHA/MQA/GQA 在 Tensor Cores 上高效执行的统一方法。Decode 阶段 Q length 仅为 1 token，M 维度极小（低 arithmetic intensity），直接 QK^T 会严重 underfill Tensor Cores。BitDecoding 利用 GQA/MQA 的 KV head sharing 特性：GQA 下 gq = hq/hkv 个 query heads 共享同一组 K/V。Query Transformation 将 Q tensor 从 [1, (gq, hkv)] reshape 为 [gq, hkv]——在 batch 维度上将多个共享相同 KV head 的 query 合并为更大 GEMM block→TC fragment 被完整填充→warp occupancy 和 throughput 显著提升。MHA (gq=1) 下无 sharing 可合并，但 decode Q_len=1 时 M 维度仍小→BitDecoding 仍受益于 Wn↑ warp layout。MQA (hkv=1, gq=hq) 下 sharing 最大→收益最大。

从 kernel 调度角度拆解术语：

```
// === Query Transformation Example (GQA, gq=4, hkv=8) ===
// 原始: Q.shape = [1, 32]   (1 token, hq=32)
//       K.shape = [L, 8, d] (L tokens, hkv=8 heads)
// GQA: 每4个Q head共享1个KV head

// 原始decode实现: 
//   for each query head hq_i (0..31):
//       kv_head = hq_i / 4;  // 每4个Q head映射到同一KV head
//       score[hq_i] = Q[0, hq_i] × K[:, kv_head]^T  // 1×L GEMV
//   → 32个独立GEMV → TC严重underutilized (M=1)

// BitDecoding Query Transformation:
// 1. Reshape: Q: [1, 32] → [1, (4, 8)] → [4, 8]
//    - 4 = gq (grouped queries per KV head)
//    - 8 = hkv (number of KV heads)
// 2. 对每个KV head k (0..7):
//    Q_group = Q[:, k]  // shape [4, d_head] — 4个共享相同K的Q
//    K_kv   = K[:, k, :] // shape [L, d_head]
//    scores_k = Q_group × K_kv^T  // shape [4, L] — 4×L GEMM on TC
// 3. 结果: 8个[4, L] score矩阵

// 效果: M维度从1→4 → TC mma tile M=4更接近完整的M=16→更高occupancy
// MQA (hkv=1): Q reshape [1, hq] → [hq, 1] → M=hq完全利用TC
```

术语一般如何实现？如何使用？

Query Transformation 在 kernel launch 前通过 memory layout reshape 实现（zero data copy）。对于 MHA (gq=1)，no reshape needed——warp parallelism (Wn↑) 补偿。在 BitDecoding 的 query transformation module 中统一处理三种 attention 变体：通过 gq = hq/hkv 自动确定 reshape 逻辑。开源于 https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

