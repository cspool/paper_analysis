## Pushbuffer DMA（PBDMA）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pushbuffer DMA（PBDMA）是 NVIDIA GPU 硬件调度器中的一个硬件单元，负责将 GPU 命令从 CPU 端 pushbuffer（位于 CPU 主内存中，通过 user-writable 内存映射到用户空间）通过 DMA 传输到 GPU 端，然后解析命令并分发给对应的 GPU 引擎执行。PBDMA 是 GPU 调度管线中连接 runlist 和引擎的关键环节（Fig.4 Step ④）。PBDMA 的工作方式：GPU HW scheduler 通过 runlist 发现某个 channel 有 pending 命令后，指令 PBDMA 单元从该 channel 关联的 pushbuffer 中拉取命令数据，解析命令类型（compute kernel、memory copy、video encode 等），然后转发给对应的引擎（Compute Engine、Copy Engine、NVDEC、NVENC 等）进行进一步调度和执行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

PBDMA 在 GPU 调度管线中的核心流程：

```
① GPU HW scheduler 在 runlist 的当前 timeslice 中发现某 TSG 的某 channel 有 pending 命令
② HW scheduler 选择该 channel，触发关联的 PBDMA 单元
③ PBDMA 通过 PCIe DMA 从 CPU 主内存的 pushbuffer 读取命令数据
④ PBDMA 解析命令头，确定目标引擎类型（Compute/Copy/Video/NVENC/...）
⑤ PBDMA 将解析后的命令传递给对应 engine 的 engine-specific scheduler
   （如 Compute Engine 继续由 thread block scheduler 调度）
```

PBDMA 与 Runqueue 的关系（基于论文对单 runlist 上 compute 干扰 copy 的解释，Fig.9）：
- 每个 PBDMA 单元 snoop runlist 中不同的 runqueue
- 每个 channel 关联一个或多个 runqueue
- 每个 runqueue 限制其可运行的命令类型
- Compute-associated channel 可选使用 copy runqueue，而 copy-exclusive channel 仅使用 copy runqueue
- 这导致单 runlist 上 round-robin 仲裁所有 runqueue-using channel——compute channel 即使不做 copy 也会因为其关联的 copy runqueue 为 idle 而短暂中断 copy engine

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PBDMA 是 GPU 硬件中的固定功能单元，在 NVIDIA GPU 中作为 Host Interface 的一部分实现。PBDMA 的寄存器在 NVIDIA 开源 GPU 文档中有部分记录（manuals/ampere/ga100/dev_pbdma.ref.txt, line 3803 描述了 channel disable 行为；"Semaphore switch option" section, line 3797 描述了 runlist semaphore snooping）。开发者通常不直接与 PBDMA 交互——它完全由 GPU 硬件调度器内部使用。但了解 PBDMA 的行为对于实时系统 GPU 管理至关重要：例如 batch 多个小 kernel 可以减少 PBDMA 的 fetch 开销；避免在不必要时使用 copy engine（即使是隐式的）可减少 runqueue contention。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
