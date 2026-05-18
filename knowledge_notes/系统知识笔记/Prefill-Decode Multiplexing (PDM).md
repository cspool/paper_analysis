## Prefill-Decode Multiplexing (PDM)

术语是什么？

Prefill-Decode Multiplexing (PDM) 是一种LLM推理服务架构范式，在**同一GPU内部**通过SM（Streaming Multiprocessor）空间分区，使Prefill和Decode阶段同时执行，而非将它们分配到不同GPU（PD-Disaggregation）或以时间片轮转（Chunked Prefill）。PDM在单个GPU实例内共享统一的KV Cache池，避免跨实例迁移开销，并动态调整SM资源分配以适配工作负载变化。

从系统架构角度拆解术语：

PDM将GPU的SM集合划分为两个分区：一部分SM专用于Decode（满足ITL SLO），剩余SM全部用于Prefill。系统以"Layer级"粒度执行Prefill，将长prompt拆分为多个小Prefill Block，在Decode迭代之间动态插入。关键流程为：

1. **SM资源划分**：通过NVIDIA GreenContext API，将GPU SM集合划分为Prefill分区和Decode分区，分区大小由SLO-aware Dispatcher动态调整。
2. **Decode优先保证**：每轮Decode迭代前，Dispatcher根据当前请求的ITL SLO，计算保证SLO所需的最少SM数量，分配给Decode。
3. **Prefill填空**：剩余SM全部用于Prefill，以Block粒度（如每Block处理N个token）插入空闲SM执行。
4. **Bubble消除**：当Decode因非确定性输出长度而产生GPU空闲时，小粒度Prefill Block能快速填充这些Bubble。
5. **KV Cache共享**：由于Prefill和Decode在同一进程内运行，共享统一KV Cache池，避免跨进程迁移开销。

术语一般如何实现？如何使用？

PDM在SGLang框架中基于NVIDIA GreenContext (CUDA 12.6+) 实现。具体实现方式：
- 使用`cuDevSmResourceSplit`将SM划分为两组，创建两个GreenContext分别绑定Prefill和Decode的CUDA Stream
- Decode Stream绑定SLO保证所需的最小SM集合，Prefill Stream绑定剩余SM
- 两个Stream异步并行执行，通过CUDA运行时自动调度
- 通过`Contention-tolerant Estimator`建模HBM带宽竞争对Prefill和Decode性能的影响，指导调度决策

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---
