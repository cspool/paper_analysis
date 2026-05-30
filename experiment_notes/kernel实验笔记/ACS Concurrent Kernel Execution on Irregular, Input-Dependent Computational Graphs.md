## ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ACS（Automatic Concurrent Scheduling）在运行时对顺序发射的 GPU kernel 进行乱序调度，通过在固定大小的调度窗口内执行依赖检查和状态跟踪来识别独立 kernel 并并发执行。实现分为两部分：(i) ACS-SW：纯软件实现，使用用户态运行时系统维护调度窗口，通过 CUDA stream 并发发射独立 kernel；(ii) ACS-HW：硬件-软件协同实现，在 GPU 硬件中实现调度窗口以消除 CPU-GPU 同步开销。
  实验比较了四种配置：(i) Baseline（单 CUDA stream 串行执行，cuDNN/JAX 实现）；(ii) ACS-SW（真实硬件上评估，仅 Deep RL 仿真 workload）；(iii) ACS-SW-Sim（模拟器上评估 ACS-SW，用于与 ACS-HW 对比）；(iv) ACS-HW（模拟器上评估硬件-软件协同机制）；(v) CUDAGraph（将核间依赖构建为 DAG 后提前发送给 GPU）。评估指标为运行时加速比和 GPU 达到的 occupancy。

- 后端平台是什么，配置是什么。
  - **真实硬件（ACS-SW）**：Intel Core i7-11700K CPU @ 3.6GHz，4-wide OOO dispatch，32-entry LSQ，L1D+L1I 32KB 4-way LRU，L2 256KB 8-way LRU，L3 1MB 16-way LRU，2-channel DDR4 DRAM（4GB / 12GB variant）；NVIDIA RTX 3060 GPU，28 SMs @ 1.3GHz，每 SM 2 个 scheduler，32768 registers，32KB shared memory，128KB L1D，12GB DDR4。
  - **GPU 模拟器（ACS-HW）**：Accel-Sim 模拟器，配置为 RTX 3070 参数：46 SMs @ 1.4GHz，每 SM 4 个 scheduler，32768 registers，32KB shared memory，128KB L1D，16GB DDR4。功耗建模使用 AccelWattch。调度窗口大小=32。

- 评估性能的软件/脚本是什么。修改了什么。
  - Deep RL 物理仿真：Brax 框架（JAX 实现），5 个 MuJoCo 环境（Ant, Grasp, Humanoid, Cheetah, Walker2d）。
  - 动态 DNN：InstaNAS-A（CIFAR10）、Dynamic Routing Dynamic-A 16-layer（Cityscapes）、Conditional Convolution 4-experts + EfficientNet-B4 backbone，PyTorch 实现，batch size=1。
  - 静态 DNN（NAS 优化 CNN）：NASNet、AmoebaNet、SqueezeNet、RandomWire，PyTorch 实现，batch size=1，CIFAR10。
  修改：通过 ACS_wrapper 为每个 kernel 定义 `__read_segments__` 和 `__write_segments__`（起始地址+大小列表），以及 `get_addresses()` 函数在 kernel launch 前解析虚拟地址。ACS-SW 运行时系统由 window module 线程和 scheduler module 线程实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文声明将提供 ACS-SW 开源实现（"We will provide an open-source software-only implementation of ACS"），但当前未找到公开开源链接（2026年5月检索 arXiv abs/2401.12377 及相关页面均未列出代码仓库 URL）。论文使用的评估框架均为已有开源项目（Brax、PyTorch、Accel-Sim）。

  评估原理：ACS 运行时系统在 CPU 端维护一个输入 FIFO 队列和调度窗口。应用线程调用 kernel 时，先通过 `get_addresses()` 解析 read/write segments（起始虚拟地址+大小），kernel 及其 segments 元数据进入输入 FIFO。Window module 线程将 kernel 插入调度窗口时，比较新 kernel 的 write segments 与窗口内所有 kernel 的 read+write segments 是否重叠，若有重叠则标记为 upstream kernel。Scheduler module 线程（可配置数量，每个对应一个 CUDA stream）轮询调度窗口，找到 upstream list 为空的 ready kernel，将其发射到自己的 CUDA stream 中，然后调用 cudaStreamSynchronize 等待完成。完成后通知 window module 更新所有 kernel 的 upstream list。

  ACS-HW 流程：软件端维护输入 FIFO 和 scheduled_list（允许 stale），硬件端（GPU 命令处理器内的调度窗口 SRAM）管理 kernel 依赖和状态（ready/pending/executing）。CPU 将 kernel 发送到 CUDA stream → GPU 硬件 upstream load module 修正 stale upstream list → 调度窗口跟踪各 kernel 的 upstream kernel ID（每个 slot 8-bit ID × (N-1) 个）→ kernel 完成时硬件自动更新所有 slot 的 upstream list → ready kernel 被硬件 dispatch 到 GPU 的 kernel dispatch unit 执行。

  硬件面积开销：调度窗口 N=32 时需要约 1KB SRAM；调度窗口 N=64 时插入延迟约 64 cycles（约 50-100ns）。依赖检查延迟约 410ns（窗口 16，6 个 RW-segment）到 1640ns（窗口 32，10 个 RW-segment）。
