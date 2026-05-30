## Scheduling Window (GPU Kernel)

术语是什么？
Scheduling Window（调度窗口）是 ACS 框架的核心数据结构，是一个固定大小（N=32）的滑动窗口，包含当前正在被评估依赖关系和调度状态的 GPU kernel。类似于 CPU 乱序执行中的指令窗口（instruction window / reservation station），调度窗口限定了同时被跟踪的 kernel 数量。窗口中的每个 kernel 维护三种状态：READY（所有依赖已满足，可随时发射）、PENDING（仍有未完成的 upstream kernel）、EXECUTING（正在 GPU 上执行）。窗口大小 N 的关键权衡：大窗口暴露更多 kernel 间并行性（Deep RL 仿真 N=32 比 N=16 性能高 4.5%），但增加依赖检查延迟（N=16, 6 segments: 410ns → N=32, 10 segments: 1640ns）和硬件面积（N=32: 1KB SRAM）。

从kernel调度角度拆解术语：
调度窗口的状态转移流程：
```
        ┌──────────────────────────────────────┐
        │            Scheduling Window          │
        │  ┌──────┐ ┌──────┐ ┌──────┐         │
        │  │k0 RDY│ │k1 PND│ │k2 PND│ ... k31 │ (N=32)
        │  │up:[] │ │up:[0]│ │up:[1]│         │
        │  └──┬───┘ └──┬───┘ └──┬───┘         │
        │     │launch   │        │              │
        └─────┼─────────┼────────┼──────────────┘
              │         │        │
              ▼         │        │
         GPU SM[0..27]  │        │
         (executing)    │        │
              │         │        │
         完成时通知 ◄────┘        │
         remove k0 from          │
         upstream of k1,k2,...   │
         k1→READY                │
         k2→PENDING (仍有upstream│
                   未完成)        │

状态转移规则:
- 插入窗口: up[]为空?→READY : →PENDING
- upstream kernel完成: 从所有slot的up[]中移除该ID
  - up[]变为空: PENDING→READY
- 被scheduler发射: READY→EXECUTING
- 执行完成: EXECUTING→移除出窗口, 新kernel从InputFIFO补入
```

术语一般如何实现？如何使用？
ACS-SW 中调度窗口由 CPU 端的 window module 线程维护（C++ 数据结构 + 互斥锁保护）。ACS-HW 中调度窗口由 GPU 命令处理器中的 SRAM 模块实现（全关联存储，每个 slot 一个 SRAM bank）。程序员通过 ACS 框架自动使用调度窗口，无需直接操作。论文通过实验确定 N=32 为合理的默认窗口大小（在并行性暴露和开销之间平衡）。窗口满时，后续 kernel 在输入 FIFO 中等待。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
