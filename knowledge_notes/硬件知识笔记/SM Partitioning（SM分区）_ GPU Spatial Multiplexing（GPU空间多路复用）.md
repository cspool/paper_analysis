## SM Partitioning（SM分区）/ GPU Spatial Multiplexing（GPU空间多路复用）

术语是什么？

SM Partitioning（也称GPU Spatial Multiplexing）是一种将GPU的流式多处理器(SM)按空间划分为多个独立子集的技术，每个子集可同时执行不同的计算任务。与时间多路复用(Time Multiplexing)不同，空间多路复用允许多个kernel或计算阶段在同一时刻真正并行执行。

从硬件架构角度拆解术语：

SM分区在GPU硬件层面的运转：

1. **GPU的SM组织结构**：以H200 (Arch 9.0)为例，共有132个SM，通过GPC (Graphics Processing Cluster) 组织。每个SM独立拥有寄存器文件、L0指令缓存、Warp Scheduler和Tensor Core。
2. **共享资源**：所有SM共享HBM（高带宽内存）控制器、L2 Cache和Crossbar互联网络。这意味着即使SM被物理分区，HBM访问带宽仍然是共享的。
3. **分区粒度约束**：不同GPU架构有不同的最小分区粒度（见GreenContext条目），由SM到内存控制器的物理拓扑决定。
4. **并行执行机制**：不同SM分区上的kernel通过CUDA Workqueue独立调度，GPU的Thread Block Scheduler将各分区的block映射到对应SM执行。
5. **带宽竞争**：当两个SM分区同时执行内存密集型任务（如Decode的KV Cache访问）时，HBM带宽成为瓶颈，需通过Contention-tolerant Estimator建模。

术语一般如何实现？如何使用？

主要实现方式：
- **NVIDIA GreenContext**（CUDA 12.4+）：进程内轻量SM分区，当前主流方案（MuxWise使用）。
- **libsmctrl + CUDA Stream Mask**：直接修改CUDA stream metadata实现SM分区，mask更新仅~4us，适合频繁动态重分区场景（Bullet使用）。
- **NVIDIA MIG (Multi-Instance GPU)**：硬件级分区，提供最强的隔离性但灵活性差（仅在A100/A30等数据中心GPU上支持）。
- **NVIDIA MPS (Multi-Process Service)**：进程间时间片共享，不完全的空间隔离。Bullet用MPS管理prefill/decode worker进程，配合libsmctrl在进程内做SM mask控制。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---

