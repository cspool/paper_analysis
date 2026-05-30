## Continual KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Continual KV Cache Compression 是在 LLM 推理的 decoding 阶段持续执行 KV cache 压缩的策略。传统方法仅在 prefill 完成后压缩一次，后续 decoding 生成的 KVs 不再压缩。Continual compression 在每个（或每隔若干）decoding step 后，累积新生成 token 的 attention 到已有 eviction metrics 中，当需要（如 preemption 即将发生）时基于更新后的 metrics 再次压缩。

数学上（KV-Compress Equation 20）：$M_{h_k,j}^{(cc)} = M_{h_k,j}^{(pool)} + \sum_{i=L_c}^{L_c+t} \sum_{h \in H_k} (A_{h,i,j})^2$，其中 $M^{(pool)}$ 为 prefill 阶段计算的初始 metric，$L_c$ 为 input context 长度，$t$ 为当前 decoding step。每次 decoding step 后，新 query 的 squared attention 被累积到对应 key 的 metric 中。

从算法pipeline角度拆解术语：

```
# Continual Compression 流程
输入：prefill 后的初始 metrics M_init, initial compressed KV cache
参数：compression rate r, block size b

for each decoding step t:
    # Step 1: 正常 decode
    Q = compute_query(x_t)
    A = attention(Q, K_cache, V_cache)  # 计算 attention
    next_token = sample(A @ V_cache)

    # Step 2: 累积新 attention 到 metrics
    for each key head h in H_kv:
        for each query head h_q in group H_k:
            for each cached key j:
                M[h, j] += (A[h_q, new_query, j])^2  # accumulate

    # Step 3: 检查是否触发重压缩
    if would_need_preemption() or compression_interval_reached():
        # 基于更新后 metrics 重新选择 eviction
        sort(M) → select E_s blocks to evict
        MoveCache(K, V, M) → free blocks
        # 重压缩可能 evict 不同 KVs（早期高 attention 后期低 attention 的 key 可能被新选中 evict）

    # Step 4: 追加当前 token KVs 到 cache
    append(K_new, V_new, to_cache)
```

术语一般如何实现？如何使用？

KV-Compress 中 continual compression 的触发条件是：(a) prefill 后立即压缩；(b) 当 preemption 即将发生时压缩（即 free blocks 不足时触发压缩以释放空间）。不使用固定间隔压缩（方案 1 和 2 被测试但不如方案 3+4 有效）。

适用场景：长文本生成（long output）或 high-concurrency serving 场景——decoding 过程中积累的 KVs 若不持续压缩，可能导致后期 KV cache 膨胀并触发 preemption。Continual compression 确保 KV cache 在 decoding 全程保持在压缩后的大小。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
