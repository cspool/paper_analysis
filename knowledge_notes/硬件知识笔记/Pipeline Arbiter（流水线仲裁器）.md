## Pipeline Arbiter（流水线仲裁器）

术语是什么？通过联网搜索让回答具体和精准。

Pipeline Arbiter是RPU论文提出的硬件级同步机制，用于在Reasoning Core内协调decoupled的memory、compute和network三套DMA pipeline之间的数据流动和同步。它是一种嵌入在SRAM buffer entry粒度的轻量级software-managed但hardware-enforced同步机制。每个SRAM buffer entry包含一个2-bit valid counter，跟踪该entry上预期的异步consumer数量。Pipeline Arbiter通过valid counter的set/check/decrement操作，结合hardware-enforced arbitration（串行化multi-consumer访问），实现data-driven的pipeline同步，消除传统global barrier的stall overhead。

从硬件架构角度拆解：

Pipeline Arbiter的工作机制（per SRAM buffer entry）：
1. **Write端**：DMA operation在写入buffer entry时programmed with a valid count（预期consumer数）。例如Network DMA写activation时设置valid count=2，因为activation将被(1) compute pipeline消费和(2) 异步转发到neighboring cores。
2. **Read端**：Consumer使用check-valid标志来stall直到data ready。valid counter在consumer读完后可选择decrement（表示该consumer已消费完成）。
3. **Mutual exclusion**：每个buffer entry通过hardware-enforced arbitration mechanism串行化多个consumer的访问请求。一次只有一个DMA engine可以read/write/update该entry的valid counter。
4. **Software-configurable priority**：consumer访问优先级通过software-configurable policy排序，确保critical data path获得优先访问权。
5. **Blocking/non-blocking semantics**：DMA操作可以是blocking（check-valid stall until ready）或non-blocking（不检查valid直接写/读）。

这种设计使每个pipeline可以独立前进（memory pipeline在compute/network stall时继续预取数据），同时保证deadlock-free execution——因为所有同步通过data readiness而非global barrier协调。

术语一般如何实现？如何使用？

Pipeline Arbiter嵌入每Reasoning Core的SRAM buffer硬件中（memory buffer 512KB, network buffer 256KB, activation/compute buffer 32KB/TMAC）。Compiler在生成RPU ISA指令流时embed Pipeline Arbiter flags（valid count set/check values）到每条instruction中。运行时Pipeline Arbiter自动管理buffer entry粒度的producer-consumer同步，无需software polling。在RTL实现中，Arbiter控制DMA engine的stall信号（当check-valid条件不满足时hold DMA progression），直到对应buffer entry的valid count非零时才释放DMA继续执行。

涉及论文标题：
- RPU - A Reasoning Processing Unit

