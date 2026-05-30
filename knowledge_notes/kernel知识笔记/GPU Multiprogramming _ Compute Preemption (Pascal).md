## GPU Multiprogramming / Compute Preemption (Pascal)

术语是什么？
GPU Multiprogramming（GPU 多道程序）指多个 CUDA 程序（来自不同 CPU process）的 kernel 在 GPU 上交替执行。在 Pascal 架构之前（如 TX1/Maxwell），多道程序通过 thread block 级别的轮转实现——不同 process 的 kernel 不能同时执行。Pascal 架构引入了 Compute Preemption（计算抢占），支持指令级抢占，允许 GPU 在执行中保存完整上下文（寄存器、shared memory、程序计数器等）到 GPU DRAM 并切换到另一 process 的 kernel。

从kernel调度角度拆解术语：
论文附录 A 揭示了 TX2（Pascal 架构）在多 process 场景下的调度行为：
- 不同 process 的 kernel 通过 preemption 实现 time-slicing 多路复用（而非真正并发）
- 现象：两个 process 各 launch 4096 线程的 kernel，GPU timeline 显示每个 kernel 看起来始终有 4096 线程在运行（但这超过了 TX2 的物理 4096 线程上限）——原因是 block 被抢占并恢复时，GPU 端记录的 start/end 时间戳实际包含了被抢占的时间段。
- 影响：多 process 场景下 block 执行时间可能翻倍（Fig. 10: Process 曲线的 worst-case block time 超过 Task 曲线的 2 倍）
- Priority 无效：stream priority 在多 process 场景下无效——只有 task 共享地址空间场景下 priority 才起作用。

```
Task 共享地址空间 (本文核心场景):
  Kernel K1 和 K2 可在不同 SM 上真正并发执行
  → 可预测的执行时间，FIFO 调度规则适用

Process 独立地址空间:
  Kernel K1 和 K2 通过 time-slicing 交替执行
  → block 被抢占导致执行时间膨胀，不可预测
  → 推荐使用 task 共享地址空间模型进行实时系统设计
```

术语一般如何实现？如何使用？
Compute Preemption 是 Pascal 架构引入的硬件特性，由 GPU driver 自动管理，不通过 CUDA API 暴露给程序员。它在以下场景被使用：(1) 多 process 的 time-slicing 调度；(2) 单 GPU 交互式 kernel 调试；(3) 防止长时间运行的 kernel 导致显示无响应（desktop GPU）。MPS (Multi-Process Service) 可以绕过 process 间 time-slicing，允许多 process 共享一个 CUDA context 实现真正并发。在 Jetson 嵌入式平台，推荐使用 task（共享地址空间）而非 process 来最大化 GPU 利用率和可预测性。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
