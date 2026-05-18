## Write-Through KV Cache（写通KV缓存）

术语是什么？

Write-Through KV Cache是TokenFlow提出的KV cache内存管理策略，将GPU显存视为CPU memory上大容量KV cache的**高速缓存层**。与普通write-back策略（仅在显存压力或请求完成时写回KV cache到host）不同，write-through在**每次decode iteration后**都将新生成的KV chunk异步同步到host memory。这使抢占时大部分KV cache已预先同步，剩余未同步chunk极小，上下文切换延迟大幅降低。

属于GPU作为cache、CPU作为backing store的两级KV cache架构。与cache领域的write-through概念一脉相承，专用于LLM serving的KV cache管理场景。

从系统架构角度拆解术语：

Write-Through KV Cache运转流程：
1. **Decode Iteration完成**：LLM Executor完成一次decode step，生成新的KV chunk（对应新生成的token在各层的K/V张量）。
2. **Write Buffer入队**：KV chunk进入write buffer（CPU端维护的待同步队列）。
3. **Duration预估**：在下一轮compute开始前，estimator根据当前batch size、序列长度预估下一轮decode compute时间。
4. **Chunk Selection**：根据预估的compute duration选择合适大小的KV chunk（动态chunk sizing），确保write stream能在compute完成前传输完毕（最大化compute-I/O overlap）。
5. **异步同步**：Write stream（CUDA stream）将选中的KV chunk从GPU显存D2H传输到host memory。Compute stream同时执行下一轮decode forward pass。
6. **抢占发生**：当scheduler决定抢占某请求时，由于write-through已将该请求的大部分KV cache同步到host，仅需快速evict最近一次decode iteration后尚未同步的增量chunk（通常极小）。
7. **对比write-back**：Write-back仅在真正抢占或请求完成时才同步全部KV cache→长I/O stall（+93%完成时间增加，TokenFlow消融实验）；write-through持续占用PCIe带宽但消除抢占时的同步瓶颈。

术语一般如何实现？如何使用？

TokenFlow在Hierarchical KV Cache Manager中通过CUDA stream pipeline实现：write stream（GPU→host传输）与compute stream（forward pass）通过CUDA events协调并发。动态chunk sizing避免固定大小导致的under-utilization（chunk太小）或I/O stall（chunk太大超过compute时间）。消融实验：去掉write-through后完成时间恶化（从66.00s到127.28s w/o offload），说明分层内存管理和write-through是TokenFlow性能收益的核心来源。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

---
