## GPU Channel（NVIDIA GPU 调度通道）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPU Channel 是 NVIDIA GPU 硬件调度管线中的基本调度单元，封装了一个 pushbuffer（存放 GPU 命令的环形队列）及其簿记信息。Channel 是 GPU 端与软件端（CUDA stream）之间的桥梁：CPU 端将 GPU 操作（kernel launch、memory copy 等）的命令写入 pushbuffer，GPU 硬件调度器通过扫描 channel 的 pushbuffer 发现待处理命令。Channel 必须被插入 runlist 才能被调度（R3），且所有 GPU engine 操作（kernel launch、copy、device-mapped memory allocation）必须经过 channel（R1）。每个 CUDA context 默认创建有限数量的 channel——x86_64 上默认 8 个 compute channel（CUDA 12.2），嵌入式 Jetson 平台仅 2-4 个（R2）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Channel 在 GPU 调度管线中的角色（映射到 Fig.4 的高层调度管线 Step ②→③→④）：

```
Step ②: CUDA操作(如cudaLaunchKernel) → 命令写入对应stream的pushbuffer
Step ②→③: Pushbuffer被channel封装 → Channel通过TSG插入runlist
Step ③: GPU HW scheduler扫描runlist → 发现channel有pending命令
Step ④: PBDMA从channel的pushbuffer拉取命令 → 解析 → 分发到对应engine
```

Channel 数量限制的关键影响（R2 实验，Fig.5）：
- 当使用的 CUDA stream 数 > channel 数时，额外的 stream 产生 false dependency——其 head-of-stream kernel 需等待任意 channel 释放才能开始执行
- 例如 9 个 stream 在 8 channel 的 GTX 1060 上：Stream 1-8 立即开始执行，Stream 9 等到 t≈0.7s（某个 channel 释放）才开始
- Channel 分配非 FIFO（Corollary 2）：later-launched stream 可能先被分配刚释放的 channel
- 解决方法：通过环境变量 CUDA_DEVICE_MAX_CONNECTIONS 增加 channel 数

禁用 channel 的效果（R1 验证）：通过 nvdebug 的 disable_channel 接口禁用某 task 的所有 channel 后，该 task 的任何 kernel launch、memory copy 或 device-mapped memory allocation 都无法完成，直到 channel 重新启用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Channel 在 GPU 硬件中作为 PBDMA 可访问的数据结构实现，包含 pushbuffer 指针、channel 状态寄存器（enabled/next/busy/status/faulted 等）和 runqueue 关联信息。NVIDIA 开源 GPU 文档（manuals/ampere/ga100/dev_pbdma.ref.txt）记录了 PBDMA 的 channel 相关寄存器。在 CUDA 编程模型中，开发者通常不直接管理 channel——CUDA 驱动在 context 创建时自动分配 channel 并映射到 stream。但实时系统开发者需要意识到 channel 数量限制：默认数量可能不足以支持所需的最大并行 stream 数。nvdebug 工具可以查看和修改 channel 状态（enable/disable），为实时 GPU 管理提供 channel 级别的控制能力。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
