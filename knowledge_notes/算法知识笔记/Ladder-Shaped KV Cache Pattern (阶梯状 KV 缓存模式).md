## Ladder-Shaped KV Cache Pattern (阶梯状 KV 缓存模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Ladder-Shaped KV Cache Pattern 是 LaCache (Shi et al., ICML 2025) 提出的一种跨层异质 KV cache 存储模式。其核心思想是：不同于 StreamingLLM 在所有层缓存同一组 token 的 KV cache，Ladder Pattern 让不同层存储不同位置 token 的 KV cache——浅层保留早期 token 的 KV 状态，深层逐步将焦点转移到更近期的 token，形成阶梯状（ladder-shaped）的二维存储结构。

该 pattern 由两个关键超参数控制：
- **Span S**：同一 token 的 KV 状态被保留的连续层数。S 越大 → 每个 token 被更多层覆盖 → 信息保留下界越高 → 存储成本越大。
- **Overlap O**：每层保留的 token 数量。O 越大 → 每层保留更多 KV 状态 → 语义连续性越好 → 存储效率越低。

每层缓存范围的递推公式：第 l 层保留 [start_l, end_l) 范围的 token，其中 start_l = (l-1) × (S-O)，end_l = start_l + O。相邻层间有 O-(S-O) 的 token 重合。

该 pattern 的两个理论依据：(1) 均匀覆盖提升全部 token 的信息保留下界——最坏情况下重要 token 出现在覆盖最少的层，均匀分布最小化此风险；(2) 相邻 token 语义关联性高，ladder 的跨层平滑过渡实现旧 token 的 smooth fade-out 而非 abrupt eviction。经 1500+ 随机 pattern 的 PPL-cache size Pareto 验证，ladder pattern 位于最优边界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Ladder Pattern Eviction
for layer l in 1..L:
    start = (l - 1) * (S - O)
    end = start + O
    K_cache[l] = K_full[l, :, start:end, :]   # [H, O, d]
    V_cache[l] = V_full[l, :, start:end, :]   # [H, O, d]
```

关键维度：Full KV: L×H×T×d → Ladder compressed: L×H×O×d，压缩比 = T/O。

术语一般如何实现？如何使用？

Training-free，仅需在 prefill 后根据层索引计算保留范围并裁剪 KV cache。与 StreamingLLM 同属基于位置的静态 eviction，天然兼容 FlashAttention。LongBench 理解任务设 S ≈ num_layers × compression_ratio（均匀压缩），语言建模任务设 S = L/4（消融最优）。代码开源：https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---
