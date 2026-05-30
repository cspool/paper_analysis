## KV Footprint (KV足迹) / Critical KV Footprint

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Footprint 是 PruLong 论文提出的衡量 KV cache 内存使用效率的统一指标，定义为所有推理时间步上未被逐出（un-evicted）的 KV entries 数量的时间积分（time-aggregated sum），归一化至完整 causal attention 对应值。与 KV cache size（瞬时指标）不同，KV Footprint 捕获 KV cache 在 pre-filling 和 decoding 全过程的累积内存使用，包括每个 KV entry 的生命周期。理想 KV eviction 方法应尽早逐出 KV 以最小化 KV Footprint。Critical KV Footprint 定义为保持 ≥90% full attention 性能的最小 KV Footprint。

从算法pipeline角度拆解术语。

**KV Footprint 计算**：
```
KV_footprint = 0
for t in prefill_steps + decode_steps:
    un_evicted[t] = count(active_KV[t]) + count(inactive_KV[t])  // 未被 evict 的 KV
    KV_footprint += un_evicted[t]
KV_footprint /= total_full_attention_entries  // 归一化百分比

// Critical KV Footprint
critical = min footprint subject to: score(footprint) >= 0.9 × score(full_attn)
```

**Footprint 示例（N=6 tokens, prefill 2 chunks, decode 4）**：
- Full causal attention：36 KV-query pairs → footprint 100%
- Step 3 后 evict 部分 KV：26/36 → 72.2%
- Step 1 后即刻 evict：更低 footprint

术语一般如何实现？如何使用？

KV Footprint 是分析性指标。论文附录验证了 KV Footprint 与真实硬件指标的相关性：PruLong 较低 KV footprint 对应较低 peak GPU memory（26.3 GiB vs PyramidKV 33.7 GiB）和较高 throughput。KV Footprint 作为理想化指标可忽略 CUDA kernel 差异、PyTorch GC 延迟等实现细节，实现跨方法公平比较。代码开源：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
