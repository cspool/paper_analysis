## GPU Runlist（GPU运行列表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Runlist 是 NVIDIA GPU 硬件调度器（Host Interface）中的核心调度数据结构。Runlist 是一个由 GPU 硬件直接管理的 channel 列表，GPU HW scheduler 通过 round-robin 方式在 runlist 内各 TSG（Time-Slice Group）间进行 timeslicing，每个 timeslice 内循环扫描当前活跃 TSG 中所有 channel 的 pushbuffer，寻找有待处理命令的 channel 进行调度。Runlist 的作用是将软件层的 CUDA context/stream 映射到硬件引擎的执行——Task 通过 CUDA 驱动初始化调度状态时将 channel（含 pushbuffer）封装为 TSG 并插入 runlist，GPU HW scheduler 定期扫描 runlist，选中 channel 后通过 PBDMA 将命令从 CPU pushbuffer 拉到 GPU，解析后分发给对应引擎执行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Runlist 是 GPU 硬件调度管线的中间层，连接 channel（面向 task/stream）和 engine（面向执行单元）：

```
CPU Userspace → Pushbuffer (channel) → Runlist → PBDMA → Engine Dispatch
```

关键硬件特性（基于 Bakita & Anderson, RTAS 2024 的实验发现）：

1. **Runlist 与 Engine 的多对多映射**：每个 runlist 可绑定多个 engine（R6），但每个 engine 只绑定一个 runlist（R7，由硬件 PTOP 寄存器约束）。例如 GTX 1060 的 Runlist 0 同时绑定 Graphics/Compute Engine 和两个 GRCE（Graphics Copy Engine），而 Runlist 5 仅绑定 Copy Engine 2（LCE2）。现代高端 GPU（如 RTX 6000 Ada）有 17 个 runlist。

2. **单 Runlist 互斥调度（R4）**：当仅一个 engine 关联到一个 runlist 时，每个 runlist 最多有一个 task 处于 active 状态。Compute task 在单 runlist 上以约 2ms 的 timeslice 交替执行（Fig.6）；copy task 以约 1ms 切换（Fig.7）。

3. **多 Runlist 独立调度（R5）**：多个 runlist 可并行活跃，各自独立调度。例如 GTX 1060 上 compute task（在 compute runlist）和 copy task（在 copy runlist）可同时执行且互不干扰（Fig.8）。但在 Jetson TX2 等嵌入式平台上，compute 和 copy 共享单 runlist 导致 copy engine 被 compute timeslicing 干扰（copy 中断间隔 = compute timeslice 1024µs，而非 copy timeslice 1049µs，Fig.9）。

4. **Runlist 内容检查**：可通过 nvdebug 的 /proc/gpuX/runlistY 接口查看 runlist 内容，包括 TSG 条目的 scale/timeout/length 和 channel 的 enabled/busy/status/PBDMA faulted/ENG faulted 等状态。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Runlist 的实现位于 GPU 硬件 Host Interface 中，通过 GPU MMIO 寄存器访问。NVIDIA 开源 GPU 文档（https://github.com/NVIDIA/open-gpu-doc）和开源内核驱动（https://github.com/NVIDIA/open-gpu-kernel-modules）中部分记录了 runlist 的寄存器布局。Runlist 条目存储在 GPU 物理内存中，需通过 GPU 页表（GMMU）进行虚拟→物理地址转换后访问。NVIDIA 专利（US 9,442,759）描述了 concurrent execution of independent streams in multi-channel time slice groups 的机制，其中 runlist 仲裁多个 TSG。在实际使用中，开发者通常不直接操作 runlist——而是通过 CUDA API 创建 context 和 stream，CUDA 驱动自动将 channel 和 TSG 插入适当的 runlist。nvdebug 工具通过绕过驱动直接读写 GPU MMIO 寄存器来暴露 runlist 状态，为实时系统开发者提供检查和验证 GPU 调度配置的能力。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
