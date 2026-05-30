## Proactive Caching for MoE（MoE的主动式缓存）

术语是什么？
Proactive Caching 是 ProMoE 提出的面向 MoE-based LLM 推理的 expert 参数缓存策略，与传统的 Reactive Caching（被动响应 cache miss）根本不同。Proactive Caching 通过 learned predictor 提前预测未来将被访问的 experts，在推理过程中异步将预测的 expert 参数从 CPU memory 预取到 GPU cache 中，使大部分 expert 数据传输与 GPU 计算重叠，从而将数据传输移出推理关键路径。核心组件：Predictor（预测哪些 experts 将被需要）+ Prefetcher（协调 prefetch 与 inference 的执行）。

从系统架构角度拆解术语：
Proactive Caching 在 ProMoE 中的执行流程——双优先级任务队列（HIGH=precise, LOW=speculative），每层先由 predictor 发出 LOW priority speculative prefetch 任务（与 self-attention 并行），gate function 完成后 early preemption 清除同层 LOW 任务并发出 HIGH priority precise prefetch 任务，cached experts 优先执行，prefetching experts 的传输与计算 pipeline 重叠。对比 Reactive Caching 将 60-94% 的加载等待时间从关键路径移除。

术语一般如何实现？如何使用？
实现为 LLM serving 框架的扩展（ProMoE: 6,600 行 C++，集成到 transformers 和 llama.cpp）。需 offline 训练 per-layer MLP predictors（1-2 小时/模型）。Prefetcher 作为独立 CPU worker thread 运行，通过 cudaMemcpyAsync 与 GPU 计算 stream 并行。适用于 memory-constrained consumer-grade GPU 上运行大规模 MoE 模型。

涉及论文标题：
- ProMoE: Fast MoE-based LLM Serving using Proactive Caching
