## Command Processor (GPU CP)

术语是什么？
Command Processor（CP，命令处理器）是 GPU 硬件中负责从命令队列中取 kernel launch 包、解码并分派 kernel 到计算单元执行的硬件模块。在现代 NVIDIA GPU 架构中，CPU 通过 PCIe 上的 MMIO 区域（设备映射内存）将 kernel launch 包写入用户模式命令队列（command queue），CP 按序从队列头部取包解码，将 kernel 分派到 GPU 的 kernel dispatch unit 进行 SM 级别的调度。每个命令队列由一个独立的 CP 服务，多队列可通过 HyperQ 等技术支持来自不同队列的 kernel 并发执行。CP 仅跟踪队列中的 kernel 顺序，不拥有 kernel 间数据依赖信息——同一队列中的 kernel 严格按序执行，不同队列的 kernel 被假设为独立。

从硬件架构角度拆解术语：
GPU 命令处理器的 kernel 调度流程：
```
CPU侧                             GPU侧
┌─────────────┐                 ┌──────────────────────┐
│ Application │                 │   GPU Device Memory  │
│   (Host)    │                 │                      │
│             │ write launch    │  ┌────────────────┐  │
│  CUDA       │──packet─────►   │  │ Command Queue  │  │
│  Runtime    │  (MMIO)         │  │ (Ring Buffer)  │  │
│             │                 │  │ [k0][k1][k2]...│  │
│             │                 │  └───────┬────────┘  │
│             │                 │          │ read       │
│             │                 │  ┌───────▼────────┐  │
│             │                 │  │Command Processor│  │
│             │                 │  │  Decode packet  │  │
│             │                 │  │  Check resources│  │
│             │                 │  │  Dispatch       │  │
│             │                 │  └───────┬────────┘  │
│             │                 │          │            │
│             │                 │  ┌───────▼────────┐  │
│             │                 │  │ Kernel Dispatch │  │
│             │                 │  │    Unit         │  │
│             │                 │  │ Assign TB→SM    │  │
│             │                 │  └───────┬────────┘  │
│             │                 │          │            │
│             │                 │  ┌───────▼────────┐  │
│             │                 │  │  SM[0..N-1]     │  │
│             │                 │  │  Execute TBs    │  │
│             │                 │  └────────────────┘  │
│             │                 └──────────────────────┘
```

在 ACS-HW 中，命令处理器被增强以支持乱序 kernel 调度：硬件调度窗口 SRAM 集成在 CP 中，CP 不仅按序取包，还能检测窗口内 kernel 的依赖状态，将 READY 的 kernel 乱序 dispatch。硬件 upstream load module 修正 CPU 端可能 stale 的 upstream list。kernel 完成后，CP 硬件自动更新调度窗口中所有 slot 的 upstream list（N-1 cycle），无需 CPU 同步。

术语一般如何实现？如何使用？
NVIDIA GPU 中 CP 是闭源固件实现（GPU System Processor / GSP），通过 MMIO 与 CPU 通信。AMD GPU 等价模块称为 Command Processor / ACE（Asynchronous Compute Engine）。CUDA 编程模型通过 `CUDA Stream` 抽象屏蔽了底层 CP 和命令队列。ACS-HW 提出的硬件修改在 CP 中增加 1KB SRAM（N=32 时）和 upstream load module 逻辑。标准用户无法直接配置或访问 CP——所有交互通过 CUDA API 间接进行。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
