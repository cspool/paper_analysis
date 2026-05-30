## Grouped-Query Attention (GQA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA, 分组查询注意力) 是介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间的一种 attention 变体。在 MHA 中，每个 query head 有独立的 KV head(GQA_ratio=1)；在 MQA 中，所有 query heads 共享单一 KV head（extreme sharing）。GQA 将 query heads 分组，同组内的多个 query heads 共享一个 KV head。GQA ratio (如 1, 4, 8, 16) 定义了 query heads 与 KV heads 的比例——ratio=1 等价于 MHA，ratio=num_heads 等价于 MQA。GQA 由 Ainslie et al. (2023) 提出，用于减少 KV cache 内存占用和 attention 计算量，同时保持比 MQA 更好的模型质量。在 LLM serving 中，GQA 对 attention kernel 性能有直接影响：高 GQA ratio 意味着更大的 Q 矩阵（更多 query heads 共享同一 K/V），decode 阶段每个 query head 的 K/V 共享度更高，缓解了 per-query GEMV 的 tensor core underutilization 问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GQA 在 attention 计算中的实现（以 ratio=4, num_q_heads=32, num_kv_heads=8 为例）：

```
// MHA (ratio=1): 32 Q heads, 32 KV heads
for head in 1..32:
    Q_h = X @ W_q[h]              // (seq_len, d)
    K_h = X @ W_k[h]              // (seq_len, d)
    V_h = X @ W_v[h]              // (seq_len, d)
    O_h = softmax(Q_h @ K_h^T / sqrt(d)) @ V_h
O = concat(O_1..O_32) @ W_o

// GQA (ratio=4): 32 Q heads, 8 KV heads
// KV heads shared: Q_1..Q_4 use KV_1; Q_5..Q_8 use KV_2; ...
for group in 1..8:
    KV_h = group
    for q_head in 4*(group-1)+1 .. 4*group:
        Q_h = X @ W_q[q_head]
        O_h = softmax(Q_h @ K_{KV_h}^T / sqrt(d)) @ V_{KV_h}
O = concat(all O_h) @ W_o

// Key difference in KV cache:
// MHA: KV cache size = 2 × 32 × L × d
// GQA-4: KV cache size = 2 × 8 × L × d     (4× smaller)
// GQA-16: KV cache size = 2 × 2 × L × d    (16× smaller, Llama-2-70B style)
```

GQA 对 attention kernel 性能的影响（decode 阶段）：
```
GQA ratio=1 (MHA):
  Q: (1, d) vector  → GEMV  → tensor core 利用率低

GQA ratio=4:
  Q: (4, d) matrix  → small GEMM  → tensor core partial utilization

GQA ratio=16:
  Q: (16, d) matrix → medium GEMM  → tensor core near-full utilization
  → FlashAttention/FlashInfer 在 GQA-16 时 decode 性能大幅改善
```

在 FastTree 的 kernel benchmark 评估中（Figure 9），GQA ratio=1 时 FastTree 对 FlashAttention/FlashInfer 的 speedup 最高（tensor core underutilization 最严重），GQA ratio=16 时 speedup 缩小但仍显著（因 FastTree 的 query aggregation further increases effective batch size beyond GQA grouping + KV reuse via shared memory）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA 在模型训练时通过修改 attention 的 KV projection 实现：将 K 和 V 的 weight 矩阵大小从 (d_model, num_heads × d_k) 改为 (d_model, num_kv_heads × d_k)。推理时，各 attention kernel 库（FlashAttention、FlashInfer、FastTree）通过 GQA ratio 参数确定 K/V head 与 Q head 的映射关系。FastTree 的 attention kernel 在 grouping plan 中不区分 head——context-queries grouping 在 batch（request）维度聚合 queries，而 GQA 在 head 内聚合。两者正交互补：GQA 增加每个 query 的有效 Q matrix size，FastTree 增加每个 group 的有效 Q matrix size（通过跨 request 的 query aggregation）。因此在 GQA-16 + FastTree grouping 下，tensor core 利用率达到最高。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
