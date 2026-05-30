## Chunked Prefetching for MoE Expert Loading（MoE专家加载的分块预取）

术语是什么？
Chunked Prefetching 是 ProMoE 提出的一种细粒度 expert 参数传输调度技术，用于解决 CUDA 异步拷贝（cudaMemcpyAsync）不可被中途抢占的问题。由于每个 MoE expert 包含三个线性层（gate_proj, up_proj, down_proj），ProMoE 将每个 expert 的参数按这三个自然边界拆分为 3 个 chunk，以 chunk 为最小单位提交 prefetch 任务。当高优先级（HIGH, precise）prefetch 任务到达时，worker thread 最多只需等待当前 chunk 完成（而非整个 expert），将高优先级任务的阻塞延迟降至原来的 1/3。

从kernel调度角度拆解术语：
Worker thread 循环从双优先级队列取 task（粒度=chunk），每个 task 包含 layer id、expert id、chunk id 和 priority。chunk=0 的任务触发 LRU cache replacement（为新 expert 分配空间）。每个 chunk 通过 cudaMemcpyAsync 传输，完成后更新 ready_chunk 计数器。Without chunking：LOW 任务 = 1 entire expert (~85MB) → HIGH 任务等待最多 ~3.7ms。With chunking（3 chunks）：LOW 任务 = 1 chunk (~28MB) → HIGH 任务等待最多 ~1.2ms，3× faster preemption。

术语一般如何实现？如何使用？
ProMoE 利用 MoE expert 天然的三层结构（gate_proj, up_proj, down_proj）作为 chunk 边界。ready_chunk 计数器跟踪每个 expert 的已完成 chunk 数。Inference 执行 expert FFN 前检查 ready_chunk 确保所需 chunk 已就绪。chunk 粒度权衡：更小 chunk 减少 preemption 延迟但增加任务队列管理开销；3 chunks/expert 在实践中提供良好平衡。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
