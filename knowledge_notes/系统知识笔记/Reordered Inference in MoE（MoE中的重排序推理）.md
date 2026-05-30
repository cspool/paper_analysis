## Reordered Inference in MoE（MoE中的重排序推理）

术语是什么？
Reordered Inference 是 ProMoE 提出的 cache-aware expert 计算顺序重排优化。传统 MoE 推理按 expert ID 顺序执行——可能导致已缓存 experts 被迫等待缺失 experts 的 prefetch（GPU 闲置），且缺失 expert 加载可能驱逐即将被访问的已缓存 experts（cache thrashing）。Reordered Inference 利用 MoE 中各 expert 输出通过加权求和合并、计算间无依赖的特性，按 cache/prefetch 状态重排：已缓存优先 → 正在 prefetch → 完全未开始，使缺失 expert 的 prefetch 与已缓存 experts 的计算形成 pipeline 重叠。

从系统架构角度拆解术语：
传统顺序：expert1(OK) → expert2(MISS! blocks, 可能 evict expert4/5) → expert4/5(wait!)
Reordered: expert1(OK) → expert4(OK) → expert5(OK) → expert2(prefetched during 4&5, no thrashing)
在 prefill 阶段效果最显著（几乎所有 experts 被激活时 thrashing 风险最高），贡献 2.39× speedup。

术语一般如何实现？如何使用：
实现为 PushPreciseExperts 中的 `desc_sort_by_ready_chunk(experts)` 排序（~50 行 C++）。prefetch worker thread 和推理线程按相同顺序处理同一 expert，建立"计算-传输 pipeline"。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
