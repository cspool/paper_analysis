## On-Module Dispatcher for PIM (pseudo-MMU)

术语是什么？通过联网搜索让回答具体和精准。

On-Module Dispatcher是PIMphony提出的轻量级pseudo-MMU（Memory Management Unit），位于PIM module内部（PIM HUB中），执行运行时虚拟地址到物理地址的翻译，使KV cache可以按需动态分配而非按最大context length静态预留。与UPMEM系统[14]中的MMU（主要用于管理DRAM和PIM core local SRAM间的DMA传输，限于固定data-transfer pipeline）不同，PIMphony的dispatcher直接在DRAM内部执行address translation和动态memory allocation，支持现代LLM inference系统所需的灵活内存管理。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。

Dispatcher内部结构：(1) Instruction Buffer——暂存来自compiler的PIM指令；(2) Configuration Buffer——存储per-request configuration（request ID、当前token index Tcur等）；(3) VA2PA Table——维护每个request的虚拟地址到物理KV cache chunk的映射。执行流程：新请求进入→host初始化request ID、Tcur和VA2PA table→dispatcher在decode每条指令时根据request ID查询VA2PA table，将Dyn-Modi计算出的virtual row/col翻译为已分配物理chunk的实际地址。若KV cache增长超过当前1MB chunk→host分配新chunk并更新VA2PA映射→dispatcher后续指令自动使用新mapping。请求结束→host释放chunks→容量可被后续请求复用。Dispatcher采用lazy chunk allocation策略，每个request最后一个chunk存在minor internal fragmentation，但消除了按Tmax整段预留的巨大浪费。硬件overhead：所有internal buffer（VA2PA + command + configuration）<200KB，远小于典型PIM HUB 512KB GPR容量；area overhead约4%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Dispatcher以专用硬件逻辑在PIM HUB内实现，包含CAM或SRAM-based lookup table（VA2PA table）、简单状态机（decode→translate→issue）和buffer管理逻辑。与通用CPU MMU（TLB + page table walker）不同，PIM dispatcher的mapping粒度粗（1MB chunk）、entry少（per-request一个基础mapping即可），无需complex TLB hierarchy或page fault handling。PIMphony在CENT和NeuPIMs simulator中建模dispatcher的cycle-level行为，包括VA2PA lookup延迟、chunk allocation/release的host-PIM通信开销。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

