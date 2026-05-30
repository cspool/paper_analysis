## GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是通过黑盒实验和公开文档，逆向工程 NVIDIA TX2 GPU 的内部调度行为。具体包括：通过合成 benchmark（可配置 block 资源需求、kernel 持续时间、copy 操作）测量 GPU 上每个 thread block 的 start/end 时间（使用 globaltimer 寄存器），推导出 TX2 GPU 调度器在多 stream、多 task 场景下对 kernel dispatcher 和 copy engine 的调度规则（G1-G4, X1, R1-R3, C1-C4, N1-N2, A1-A2）。实验比较了：(1) task 共享地址空间 vs process 独立地址空间的调度行为差异；(2) NULL stream 对并发的阻塞影响；(3) stream priority 的抢占和饥饿行为；(4) 多 process 下 time-slicing/preemption 的开销。

- 后端平台是什么，配置是什么。
  NVIDIA Jetson TX2 嵌入式开发板。SoC 设计：四核 2.0GHz 64-bit ARMv8 A57 + 双核 2.0GHz superscalar ARMv8 Denver + 集成 Pascal GPU（2 个 SM，每个 SM 128 核 @ 1.3GHz，共享 512KB L2 cache）。6 核 CPU 与 GPU 共享 8GB 1.866GHz DRAM。GPU 有 1 个 Copy Engine (CE)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研合成 benchmark（synthetic workload），可配置：每个 block 的线程数、shared memory 用量、kernel 持续时间（通过 spin loop）、copy 操作大小。使用 CUDA 8.0.62 的 globaltimer 寄存器在 GPU 端记录每个 block 的 start/end 时间戳，CPU 端记录 kernel launch 时间。可视化工具绘制 GPU timeline（每 block 的起止时间、SM 分配、thread 数）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接：https://github.com/yalue/cuda_scheduling_examiner_mirror

  评估原理：
  1. 创建合成 CUDA kernel，每个 block 执行固定时长的 spin loop（如 1 秒），block 配置指定线程数（768/512/1024/256 等）和 shared memory（0/32KB）。
  2. 多个 kernel 按实验设计通过不同 stream（含 NULL stream、不同 priority）在不同 CPU task/process 上 launch。
  3. 每个 block 在 GPU 端通过 `clock64()` 或 `globaltimer` 记录自己的 start 和 end 时间戳，写入全局内存数组。
  4. CPU 端在 kernel 完成后读取时间戳数据，绘制 Gantt-chart 式 GPU timeline（x 轴为时间，y 轴为 SM，每个矩形为一个 block）。
  5. 从 timeline 中观察：kernel 何时 dispatch、block 在哪个 SM 执行、copy 操作与 kernel 执行的重叠关系、优先级抢占行为。
  6. 对比不同实验条件下的 timeline，推导调度规则。

  全过程：
  ```
  CPU task/process 调用 CUDA API launch kernel K(k) with N blocks, T threads/block, S shared_mem/block
    → CUDA runtime 将 GPU operation 入队到对应 stream queue (Rule G1)
    → 当 kernel 到达 stream queue 头部，入队到 EE queue (Rule G2)
    → GPU scheduler 检查 EE queue 头部 kernel 的 block 是否满足资源条件 (Rules R1-R3, X1)
    → 若有 SM 满足 threads ≤ 2048, shared_mem ≤ 64KB, registers ≤ 65536，则分配 block 到该 SM
    → Block 在 SM 上执行 spin loop → 记录 start/end 时间戳
    → 所有 block 完成后，kernel 从 EE queue 出队 (Rule G3)，从 stream queue 出队 (Rule G4)
    → CPU 端读取时间戳 → 生成 timeline 可视化
  ```
