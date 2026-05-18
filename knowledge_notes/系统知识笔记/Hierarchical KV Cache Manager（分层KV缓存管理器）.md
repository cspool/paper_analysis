## Hierarchical KV Cache Manager（分层KV缓存管理器）

术语是什么？

Hierarchical KV Cache Manager是TokenFlow提出的多级KV cache管理子系统，将GPU显存和CPU DRAM组织为两层缓存层次：GPU HBM作为L1“高速缓存”（低延迟、小容量，直接服务于LLM decoder的attention计算），CPU DRAM作为L2“备份存储”（高延迟、大容量，容纳被抢占或等待中请求的完整KV cache）。管理器通过三并行CUDA stream pipeline（compute/load/evict）和write-through策略，实现请求的快速抢占和恢复。

从系统架构角度拆解术语：

分层管理器的运转流程：
1. **Compute Stream**：执行LLM forward pass（attention + FFN），访问GPU HBM中的KV cache page。这是主工作流。
2. **Write Stream**：每次decode后，将新生成的KV chunk从GPU HBM异步D2H传输到CPU DRAM（write-through策略）。与compute stream通过CUDA events并发。
3. **Load Stream**：当被抢占请求恢复时，将其KV cache从CPU DRAM H2D加载回GPU HBM。Load操作按chunk粒度分批执行，优先加载即将被attention访问的recent chunk。
4. **Evict Stream**：当GPU HBM需要腾出空间给新请求时，释放已sync到CPU的KV cache block。Evict与load通过load-evict overlap重叠：preempted请求已同步chunk直接释放（evict），未同步剩余chunk与load操作在时间上重叠传输。
5. **动态Chunk Sizing**：不在每次传输中使用固定大小。根据预估的下一轮compute duration自适应选择传输chunk大小→最大化compute-I/O overlap→避免I/O stall或under-utilization。
6. **Batched Transfer**：多个请求的KV chunk合并为单次PCIe传输→减少DMA setup开销→提升PCIe带宽利用率。

术语一般如何实现？如何使用？

在TokenFlow中通过Python multithreading + PyTorch CUDA stream API实现：三类stream各运行独立线程，CUDA events在关键依赖点建立同步。系统约3000行Python代码。对比Andes的recompute-based preemption（抢占时丢弃KV cache、恢复时重新prefill计算）：TokenFlow的hierarchical KV cache避免了recompute的计算开销，但引入了持续的PCIe带宽占用。Trade-off由scheduler的buffer safety condition控制——仅在PCIe带宽可承受时维持write-through。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

---
