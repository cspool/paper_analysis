## GPU Timeslicing（NVIDIA GPU 硬件时间片轮转调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Timeslicing 是 NVIDIA GPU 硬件调度器（Host Interface）在 runlist 级别实现的时间片轮转（round-robin）调度机制。当多个 task（CUDA context/TSG）共享同一个 runlist 并使用同一个 engine 时，GPU HW scheduler 以固定时间片交替激活各个 TSG，使得多个 task 看似"同时"使用 GPU，但实际上在任何给定时刻只有一个 TSG 的 channel 正在被 dispatch。Timeslicing 在以下两种场景下发生：(i) 多 task 共享单 runlist 的同一 engine（R4 实验，Fig.6 compute task 互斥执行，约 2ms 时间片）；(ii) 单 runlist 上不同 engine 类型的 task 之间的干扰（R5 实验，Fig.9 Jetson TX2 上 copy task 被 compute timeslicing 以 1024µs 间隔中断）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

基于 Bakita & Anderson 实验测量的 Timeslicing 行为（microseconds 精度）：

```
GPU HW Scheduler Timeslicing (Runlist级别):

参数:
  - compute_timeslice ≈ 2ms (exec_logger实测)
  - copy_timeslice ≈ 1ms (copy_monitor实测)
  - 切换开销: ~145µs (time-slicing context switch, 来自prior work[8])

实际观测流程 (Fig.6, 两个exec_logger任务在GTX 1060 3GB单runlist上):

时间轴 (ms):  |--Logger1--|--Logger2--|--Logger1--|--Logger2--|...
持续时间:      |  ~2ms    |  ~2ms    |  ~2ms    |  ~2ms    |

切换频率: 右侧 ~20 timeslices / 80ms → 每任务约4ms周期内获得一个2ms时间片

单runlist跨引擎干扰 (Fig.9, Jetson TX2, exec_logger + copy_monitor):
  copy engine中断间隔 = 1024µs (compute timeslice, 而非copy timeslice 1049µs)
  → PBDMA在runlist的每个runqueue上独立round-robin
  → compute channel关联copy runqueue即使不执行copy也触发copy engine短暂中断
```

Timeslice 测量方法：
1. exec_logger: 持续执行compute kernel → 通过CUDA event记录每次kernel开始/结束时间戳 → 检测execution gap → gap pattern = timeslice切换
2. copy_monitor: 持续执行copy操作 → 记录每单位数据的copy完成时间 → 绘制progress-over-time曲线 → 拐点 = timeslice切换

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Timeslicing 由 GPU HW scheduler 全自动管理（无需软件干预），基于 TSG 的 scale 和 timeout 参数控制。在 NVIDIA Multi-Process Service (MPS) 启用时 timeslicing 行为不同——自 Volta 架构起，MPS 将各应用作为 subcontext 运行在 MPS-created context 内，Bakita & Anderson 指出其规则在 MPS 下可能仍适用（将所有 MPS-using task 视为一个 task），但未验证。实时系统开发者需注意：timeslicing 的间隔（compute ~2ms, copy ~1ms）决定了 GPU 任务的最大不可抢占执行窗口——这对实时响应时间分析有直接影响。在 Jetson 等单 runlist 平台上，timeslicing 的跨 engine 干扰效应可能严重延迟 copy 操作。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
