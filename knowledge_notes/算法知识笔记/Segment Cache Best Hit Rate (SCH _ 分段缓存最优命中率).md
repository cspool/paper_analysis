## Segment Cache Best Hit Rate (SCH / 分段缓存最优命中率)

术语是什么？
SCH 是衡量 MoE 模型在带 cache size 限制 (ρ = cache_size / active_experts) 的 expert offloading 场景下的理论最大 hit rate (ICLR 2026)。模拟 oracle segment cache：缓存大小为 ρ·k，驱逐未来 m 个 token 间激活次数最少的 expert。SCH 桥接 SRP 与实际 offloading：无容量限制时用 F1 (SRP)，有容量限制时用 hit rate (SCH)。实验表明 SCH 与 LRU/LFU hit rate 高度正相关 (m=64 时 r > 93%)，且在中等 ρ 下接近 Belady 最优 (ρ=2 时可达 90.55% of optimal)。

从算法pipeline角度拆解术语：
```
cache = empty, cache_size = ρ*k
for each segment [p, p+m):
    for token t in segment:
        demanded = top_k(router(t))
        missed = demanded \ cache
        if missed:
            # oracle: evict experts least activated in remaining segment
            future_counts = count_activations(t+1, p+m)
            evict = bottom_k(cache, future_counts, |missed|)
            cache = (cache \ evict) ∪ missed
        # record hit/miss per expert
SCH = hits / total_accesses
```

术语一般如何实现？如何使用？
与 SRP 使用相同路由决策数据。SCH 用于确定模型的最佳 GPU cache 大小——ρ=2 是大多数模型的 sweet spot（此后收益递减）。代码开源 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models
