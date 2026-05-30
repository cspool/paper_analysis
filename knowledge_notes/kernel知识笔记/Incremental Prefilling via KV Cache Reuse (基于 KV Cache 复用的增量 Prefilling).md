## Incremental Prefilling via KV Cache Reuse (基于 KV Cache 复用的增量 Prefilling)

术语是什么？
一种 kernel 级优化技术，允许在已完成的 prefix prefill 基础上追加新 token chunk 而非重新计算全部。核心原理：Prefix 的 KV blocks 在首次 prefill 后驻留 HBM，后续追加 token 的 KV 计算仅需对新 token 执行 attention 投影（Q/K/V），prefix 部分的 K/V 直接从 HBM 读取。在 Faster-MoA 中，后继 agent 的前缀段先被完整 prefill → KV 驻留 HBM；随后前驱 agent 每产出一个 decode chunk，chunk 被追加到前缀之后 → 发出增量 /prefill_only 更新 → FlashAttention 仅计算新 chunk 的 KV 并追加到 KV cache → prefix 部分的 KV 从 HBM 读取（~100% cache hit rate）。

从kernel调度角度拆解：
增量 prefill 的注意力计算流程（以追加 chunk 为例）：

```
假设已有:
  prefix KV: K_{1..P}, V_{1..P} ∈ R^{P × d_head}   // P 个已 prefilled tokens 的 KV
  HBM 地址: kv_cache_addr_prefix

增量 prefill:

Step 1: 新 chunk input tokens → QKV projection
  x_new ∈ R^{C × d_model}                           // C 个新 token
  Q_new = x_new @ W_Q    ∈ R^{C × d_head}
  K_new = x_new @ W_K    ∈ R^{C × d_head}
  V_new = x_new @ W_V    ∈ R^{C × d_head}

Step 2: 从 HBM 加载 prefix KV (内存复用)
  K_prefix = load_HBM(kv_cache_addr_prefix)          // [P × d_head]
  V_prefix = load_HBM(kv_cache_addr_prefix + offset) // [P × d_head]

Step 3: 拼接
  K_full = concat([K_prefix, K_new])  // [(P+C) × d_head]
  V_full = concat([V_prefix, V_new])

Step 4: Attention 计算 (标准 FlashAttention, P+C < 全 prompt 长度)
  S = Q_new @ K_full^T  / sqrt(d_head)               // [C × (P+C)]
  A = softmax(S, dim=-1, causal_mask=True)            // causal mask 选填
  O = A @ V_full                                       // [C × d_head]

Step 5: 写入新增 KV 到 HBM
  store_HBM(kv_cache_addr_prefix + P*d_head*sizeof, K_new)
  store_HBM(kv_cache_addr_prefix + P*d_head*sizeof + offset, V_new)
  // 完整 KV cache 现在覆盖 [P+C] tokens
```

关键优化：(1) prefix KV 无需重计算，直接从 HBM 读取——O(P·d) 内存带宽 vs O(P·d^2) 重新计算的差距；(2) 新增计算量 ∝ C·(P+C)·d_head，其中 C≪P，即大部分 FLOPs 为 prefix 内存读取所替代；(3) 增量 prefill 是 memory-bandwidth-bound 而非 compute-bound，高效利用 HBM 带宽。

在 Faster-MoA 中的使用场景：Shell Router 每收到 APC 中一个 chunk，调用一次增量 prefill，chunk size 约 16-64 tokens。由于 prefix 部分已在 HBM 中且被上一次 prefill "预热"（近 100% cache hit），每次增量 prefill 延迟约 ~1-2ms（取决于 chunk size 和 HBM 带宽），远小于完整重新 prefill。

术语一般如何实现？如何使用？
- 基于 FlashAttention 或 PyTorch SDPA 实现：将 prefix KV 传入作为额外的 key_value 参数
- 在 SGLang 中通过 /prefill_only API + 已缓存 KV blocks 的 token 追加机制实现
- 要求 PE 维护 prefix KV blocks 在 HBM，不因 /prefill_only 无输出而被回收
- chunk size 权衡：大 chunk → 少请求数但单次延迟高；小 chunk → 多请求数但更高重叠度
- 也适用于其他需要流式/增量构建 prompt 的场景（如 multi-turn dialogue 的 context 积累）

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
