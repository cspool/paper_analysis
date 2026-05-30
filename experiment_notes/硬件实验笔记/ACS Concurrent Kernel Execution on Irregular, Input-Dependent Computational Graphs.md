## ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

- 属于硬件架构的实现是什么？实验比较什么？
  ACS-HW 在 GPU 命令处理器（Command Processor）中实现了硬件调度窗口（Hardware Scheduling Window）和 upstream load module，以消除软件方案（ACS-SW）中因 CPU-GPU 通信导致的同步和 kernel launch 开销。硬件调度窗口包含固定数量 slot（N=32），每个 slot 存储 8-bit kernel ID 和 (N-1) 个 8-bit upstream kernel ID，以 SRAM 实现。Upstream load module 负责修正 CPU 端可能 stale 的 upstream list（移除已完成的 kernel，防止遗漏长运行 kernel）。
  实验比较 ACS-HW vs. Baseline（单 stream 串行）vs. ACS-SW-Sim（模拟器上软件方案）vs. CUDAGraph，评估指标为加速比和 GPU occupancy。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  Accel-Sim 模拟器（https://github.com/accel-sim/accel-sim），功耗建模使用 AccelWattch。论文未修改模拟器核心架构，而是在模拟器上建模 ACS-HW 的硬件行为（调度窗口的 SRAM 模块、upstream load module 的依赖修正逻辑），用软件模拟硬件功能。

- 模拟器模拟什么的性能，修改了什么。
  模拟 NVIDIA RTX 3070 级别 GPU 的性能（46 SMs @ 1.4GHz）。在 baseline 模拟器之上添加了 ACS-HW 的硬件调度行为建模：(i) GPU 命令处理器中的调度窗口 SRAM（N 个 slot，每个含 8-bit kernel ID + (N-1) 个 upstream kernel ID + 2-bit 状态）；(ii) upstream load module（检查 completed kernel list 修正 stale upstream list，跟踪 oldest scheduled kernel 防止 scheduled_list 遗漏）；(iii) 硬件自动从调度窗口 dispatch ready kernel 到 kernel dispatch unit。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  Accel-Sim 为学术界广泛使用的开源 GPU 模拟器。论文 ACS-SW 软件实现声明将开源，但当前（2026年5月检索）未找到公开代码仓库。Accel-Sim 的使用方式：用户配置 GPU 参数（SM 数量、频率、寄存器、shared memory、L1D、DRAM 等），将 CUDA 程序的 PTX/SASS trace 输入模拟器，模拟器逐 cycle 模拟 GPU 内部的 warp 调度、内存访问、缓存行为等，输出执行时间、IPC、occupancy 等性能指标。

  ACS-HW 硬件修改原理：应用通过 ACS_wrapper 标注每个 kernel 的 read/write segments → CPU runtime 将 kernel 元数据发送到 GPU 输入 FIFO（CUDA stream）→ 硬件 upstream load module 在插入前修正 upstream list → 调度窗口内每个 slot 存储 kernel ID 及 upstream kernel ID 列表（全关联 SRAM）→ kernel 完成时硬件扫描所有 slot（N-1 cycle）更新 upstream list → status bits 指示 ready/pending/executing → ready kernel 被硬件直接 dispatch 到 GPU kernel dispatch unit 执行。面积开销：N=32 时约 1KB SRAM（整个 GPU），N=64 时插入延迟约 64 cycles（~50-100ns），相对于 baseline kernel launch 开销（微秒级）可忽略。
