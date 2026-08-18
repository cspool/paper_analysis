## sys-scoped 访存与 GPU 内存一致性模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA GPU 内存模型区分 weak/strong 访问与作用域（scope）：scope 决定同步可见范围（device-scope 仅同 GPU；system/sys-scope 跨 GPU+CPU）；sys-scoped 内存操作或 fence 显式执行跨 GPU 同步，其他访问不要求对其他 GPU 可见或有序。跨 GPU 正确同步必须使用 sys-scoped 操作。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CDFD 对 sys-scoped 写的处理：写目标页存在多副本 → 触发 fault → flush 该页所有在途访问 → 各副本合并为单一权威版本 → 此后所有访问重定向到持有合并副本的 GPU。语义仿 cudaMemAdviseSetReadMostly 的写失效路径（Web evidence: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ ），同时保证同一地址的顺序与一致性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
程序员用 __threadfence_system()、atomicAdd_system 等 sys-scoped 原语做跨 GPU 同步；UVM/多 GPU 运行时（GPS、CDFD）在 sys-scoped 写时执行副本合并。CDFD 沿用 [30] 的处理方式。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
