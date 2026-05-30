## FLOP-Aware Eviction Policy

术语是什么？
FLOP-Aware Eviction是Marconi提出的prefix caching淘汰策略，在传统LRU recency基础上增加计算节省维度。核心公式：Utility = recency + α × flop_efficiency，其中flop_efficiency = Σ(all layers FLOPs saved) / Σ(all states memory bytes)。α由config_tuner根据workload命中率自动调优。关键洞察：SSM state固定大小但节省O(L) FLOPs，KV cache O(L)大小但节省O(L²) FLOPs——长前缀FLOP efficiency更高，应优先保留。

从系统架构角度拆解术语：
```
for each cached entry:
  total_flops = Σ(attention_flops + ssm_flops + mlp_flops for prefix_len)
  memory_bytes = Σ(kv_cache_bytes + ssm_state_bytes)  // SSM state固定
  flop_efficiency = total_flops / memory_bytes
  utility = recency_score + α × flop_efficiency_score
evict(argmin(utility))
```

术语一般如何实现？如何使用？
实现于Marconi的policy_exploration.py中可插拔eviction policy（V2版）。V1=SGLang+ LRU baseline。vs LRU: token hit rate提升19%–219%。SSM比例越高、context越长效果越好。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---
