## Hardware Scheduling Window (ACS-HW)

术语是什么？
Hardware Scheduling Window（ACS-HW 硬件调度窗口）是 ACS-HW 在 GPU 命令处理器中实现的硬件模块，用于在 GPU 侧管理 kernel 的依赖跟踪和乱序调度。它由两部分组成：(i) 调度窗口 SRAM（N 个 slot，每个含 kernel ID、N-1 个 upstream kernel ID、2-bit 状态），以全关联方式组织；(ii) upstream load module，负责修正 CPU 端可能 stale 的 upstream 列表（移除已完成的 kernel，阻塞过多新 kernel 插入以防遗漏长运行 kernel）。与纯软件方案 ACS-SW 不同，硬件调度窗口消除了 kernel launch 和同步所需的 CPU-GPU 通信延迟（5-20μs）。

从硬件架构角度拆解术语：
ACS-HW 硬件调度窗口的 SRAM 结构（N=32）：
```
┌───────────────────────────────────────────────────────┐
│            Hardware Scheduling Window SRAM (1KB)       │
│                                                       │
│  Slot 0: [kid:8bit][up0:8bit][up1:8bit]...[up30:8bit][st:2bit] │
│  Slot 1: [kid:8bit][up0:8bit][up1:8bit]...[up30:8bit][st:2bit] │
│  ...                                                   │
│  Slot 31:[kid:8bit][up0:8bit][up1:8bit]...[up30:8bit][st:2bit] │
│                                                       │
│  每个slot: 8 + (N-1)*8 + 2 = 250 bits ≈ 32 bytes     │
│  总计: 32 slots × 32 bytes = 1024 bytes (1KB)         │
│  组织: 每slot一个独立SRAM bank，全关联存储upstream IDs │
└───────────────────────────────────────────────────────┘

Upstream Load Module的工作流程:
┌──────────┐     ┌──────────────┐     ┌────────────────┐
│ CPU sends│     │ Upstream Load│     │ Scheduling     │
│ kernel + │────►│ Module       │────►│ Window SRAM    │
│ stale    │     │              │     │                │
│ upstream │     │ 1. Remove    │     │ Insert with    │
│ list     │     │    completed │     │ corrected      │
│          │     │    kernels   │     │ upstream list  │
│          │     │ 2. Track    │     │                │
│          │     │    oldest    │     │                │
│          │     │    scheduled │     │                │
│          │     │    kernel    │     │                │
│          │     │ 3. Block if │     │                │
│          │     │    >M newer │     │                │
│          │     │    kernels  │     │                │
└──────────┘     └──────────────┘     └────────────────┘
```

术语一般如何实现？如何使用？
ACS-HW 通过 Accel-Sim 模拟器进行建模评估。硬件修改在概念层面描述：SRAM 面积为 1KB（N=32），插入延迟约 32 cycles（~50ns），完成后更新延迟 N-1 cycles（~50ns），与 baseline kernel launch 开销（μs 级）相比可忽略。当前无物理实现（硅验证），属于架构提案。Accel-Sim 模拟器配置为 RTX 3070 参数（46 SM @ 1.4GHz），功耗由 AccelWattch 建模。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
