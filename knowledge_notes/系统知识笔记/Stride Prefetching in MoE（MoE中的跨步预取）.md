## Stride Prefetching in MoE（MoE中的跨步预取）

术语是什么？
Stride Prefetching 是 ProMoE 提出的 expert prefetch 时序优化策略：将 predictor 的预测目标从"下一层"改为"下两层"（prediction distance +1），使预测执行与 prefetch 传输形成 pipeline 并行。在 layer-wise prefetching 中，预测（~200μs CPU）和 prefetch（cudaMemcpyAsync）是顺序执行的。Stride Prefetching 通过增加预测距离让当前层的预测和上一层的 prefetch 同时进行，最大化 CPU-GPU 带宽利用率，FetchRate 从 ~70% 提升至 ~95%。代价是 prediction distance 增加带来的 accuracy 轻微下降（~5%）。

从系统架构角度拆解术语：
Layer-wise: predict_{l+1} → prefetch_{l+1} → compute_{l+1}（串行，predict 时带宽闲置）
Stride: predict_{l+2} 与 prefetch_{l+1} 并行，compute_{l+1} 与 predict_{l+3} 并行（pipeline，带宽持续利用）

术语一般如何实现？如何使用：
在 ProMoE 中实现为：predictor 接收第 l-1 层 hidden state 时预测第 l+1 层 experts，CPU predictor 执行与 GPU prefetch worker thread 的 cudaMemcpyAsync 并行。stride distance=1 是 accuracy-FetchRate Pareto 最优。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
