## Execution Engine (EE)

术语是什么？
Execution Engine (EE) 是 NVIDIA GPU 中执行 CUDA kernel 的硬件单元集合，包含一个或多个 SM。在本文的抽象模型中，EE 通过一个 FIFO EE queue 接收待调度的 kernel，然后将 kernel 的 thread block 分配到 SM 上执行。在 TX2 上，EE 包含 2 个 SM。

从硬件架构角度拆解术语：
论文假设 TX2 使用一个 per-address-space 的 FIFO EE queue。当 kernel 到达其 stream queue 头部时入队 EE queue（Rule G2）。EE queue 头部的 kernel 的 block 优先被分配到 SM（Rule X1：非抢占），只有头部 kernel 的所有 block 完成 dispatch（fully dispatched）后才从 EE queue 出队（Rule G3）。当启用 stream priority 时，论文假设存在两条 EE queue：priority-high queue 和 priority-low queue（Rule A1-A2），高优先级 queue 非空时低优先级 block 不可分配。EE 与 CE 并发工作——EE 执行 kernel 时 CE 可同时执行 copy 操作。

术语一般如何实现？如何使用？
在现代 NVIDIA GPU 中，EE 的物理实现是 SM 阵列。从 Kepler 架构引入 Hyper-Q 后，GPU 可同时管理多达 32 个硬件工作队列，允许多个 stream 的 kernel 真正并发执行。CUDA 程序员不直接与 EE queue 交互，而是通过 stream API 间接影响调度行为。了解 EE queue 的 FIFO 非抢占性质对实时系统设计至关重要——论文正是通过实验发现 TX2 的这种层次化 FIFO 调度特性。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
