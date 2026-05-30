## H2O / Heavy Hitter Oracle (基于累积注意力分数的 KV Cache 动态逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

H2O（Heavy Hitter Oracle, Zhang et al., 2024f）是一种基于注意力分数的 KV cache 动态 eviction 算法。其核心思想是：在自回归生成过程中，追踪并累积每对 (query_head, key_token) 的注意力分数，少量 token（heavy hitters）会持续获得大部分注意力权重。通过保留这些 heavy hitters 的 KV cache 并 evict 其余不重要的 token，可以在几乎不损失精度的情况下大幅减少 KV cache 内存占用。H2O 在每个 decode step 更新 attention score 累积值（通过求和或指数衰减平均），然后用 top-K 选择保留最重要的 KV entries。与 StreamingLLM 不同，H2O 的 eviction 策略是动态的——heavy hitters 可能在生成过程中发生变化，H2O 会自适应调整保留哪些 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2O 动态 KV cache eviction 流程**：
```
# 参数：heavy_hitter_size=64, recent_size=448, total_cache=512

# Prefill 阶段：
K_cache = X @ W_K           # [L, heads, d_head]
V_cache = X @ W_V
Q = X @ W_Q

scores = Q @ K_cache^T / sqrt(d_head)    # [heads, L, L]
# 累积注意力分数（沿 query 维度求和，得到每 key_token 被关注的总分）
attn_accum = scores.sum(dim=-2)           # [heads, L]

# Decode 每步：
for each decode step t:
    # 1. 当前 token 的 Q, K, V
    q_t = x_t @ W_Q    # [heads, d_head]
    k_t = x_t @ W_K
    v_t = x_t @ W_V

    # 2. 与所有缓存的 K 计算注意力分数
    scores_t = q_t @ K_cache^T / sqrt(d_head)   # [heads, 1, len(K_cache)]

    # 3. 更新累积注意力分数（指数移动平均或直接累加）
    attn_accum += scores_t.squeeze(1)            # [heads, len(K_cache)]

    # 4. 选择 heavy hitters + recent tokens
    heavy_idx = topk(attn_accum, k=64)           # 累积分数最高的 64 个
    recent_idx = [len(K_cache)-448, ..., len(K_cache)-1]  # 最近 448 个
    keep_idx = union(heavy_idx, recent_idx)

    # 5. Evict 不重要的 KV cache
    K_cache = K_cache[keep_idx]
    V_cache = V_cache[keep_idx]
    attn_accum = attn_accum[keep_idx]             # 同步裁剪分数

    # 6. 追加新 KV
    K_cache = concat([K_cache, k_t])
    V_cache = concat([V_cache, v_t])

    # 7. 在保留的 KV 上计算 Attention
    output_t = softmax(q_t @ K_cache^T / sqrt(d_head)) @ V_cache
```

**Annotations**: Heavy hitter 选择用 `topk(attn_accum, k=64)` 动态确定每步最重要的 64 个历史 token。Recent window（448 token）保证近期上下文完整性。总 KV cache 大小 ≈ 512（64+448），相比全量 KV cache（可能数万 token）压缩显著。H2O 的关键开销：每步需计算一次完整 attention scores（以更新 attn_accum），这与 FlashAttention 的单 pass 设计冲突——FlashAttention 不保存中间 attention scores，导致 H2O 需要额外的 multi-pass attention 和内存访问。

术语一般如何实现？如何使用？

H2O 开源：https://github.com/FMInference/H2O。关键参数：heavy hitter oracle token size（默认 64）、recent size（默认 448）、total cache size（默认 512）。动态 eviction 每步计算 attention scores → 更新累积分数 → top-K 选择 → evict 不需要的 KV。论文 "Rethinking KV Cache Compression" 的评估显示：H2O 与 FlashAttention 不兼容（需要 multi-pass attention 获取 attention scores），因此在 LMDeploy（含 FlashAttention）上 prefill 吞吐仅 0.51-0.58× FP16 baseline（LLaMA-7B TP=1/2/4），decode 为 0.85-1.34×，依赖于 batch size 和 KV length。H2O 的动态 eviction 导致 KV cache length 不单调增长，与 PagedAttention 的 fixed-size page 管理冲突。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
