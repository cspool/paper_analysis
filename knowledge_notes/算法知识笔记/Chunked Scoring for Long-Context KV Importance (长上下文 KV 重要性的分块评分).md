## Chunked Scoring for Long-Context KV Importance (长上下文 KV 重要性的分块评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunked Scoring 是 KVzip 将上下文重建评分扩展到长上下文（>100K tokens）的关键技术。直接计算完整 context 的 cross-attention 矩阵需要 O(n_c²) 内存和计算，不可行。Chunked scoring 将 context 分为固定大小 chunk（m=2K tokens），逐 chunk 独立计算重要性得分，复杂度降至 O(m·n_c)，峰值内存恒定 O(m²)。

该技术的关键设计：(1) 每 chunk 独立 forward，仅 subsample KV_c 中该 chunk 对应的 keys，形成 m+n_in 长度的 attention 计算（而非全量 n_c+n_in）；(2) chunk 间通过"前一 chunk 最后 8 tokens"的衔接 prompt 保持上下文连续性；(3) 各 chunk 得分直接拼接聚合，无需跨 chunk 归一化。

从算法pipeline角度拆解术语：

**计算流程**：

```
固定: m = 2048, T = ceil(n_c / m)

for t = 1..T:
    // Key subsampling: 仅取出当前 chunk 对应的 KV_c keys
    K_sub = KV_c.keys[:, (t-1)*m : t*m]   // H × m × d
    
    // Forward: input length = n_prompt + m
    // FlashAttention: n_in × (m + n_in) attention
    Q, K_full = forward_layer(input, KV_c)
    A = FlashAttention(Q, cat([K_sub, K_input_keys]), V)
    
    // 取 query 维度 max
    S_chunk_t = max_{query_dim} A[:,:,:m]  // H × m

// 聚合: T 个 chunk 拼接为完整得分
S = concat([S_chunk_1, ..., S_chunk_T])    // L × H × n_c
```

**复杂度**：
- Per-chunk FLOPs: O(m²)，总 FLOPs: O(m·n_c)，线性于 n_c
- 总 overhead: O(n_c² + n_c·m/2)，约 2x 标准 prefill O(n_c²/2)
- 峰值内存: O(m²)，恒定（vs O(n_c²) 全量计算）

术语一般如何实现？如何使用？

Chunked scoring 通过标准 FlashAttention-2 实现，无需修改 attention kernel。chunk size m=2K 在计算效率与 token position index 限制间取得平衡，Section C.1 消融验证不同 chunk size 间性能差异 <2%。对于 context-independent eviction 模式，chunked scoring 仅在预计算阶段执行一次。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---
