## Time-Slice Group（TSG，时间片组）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Time-Slice Group（TSG）是 NVIDIA GPU 硬件调度器中用于组织 task 调度状态的分组结构。一个 TSG 将 task 的所有 channel 与其 CUDA context 信息封装在一起，然后插入 runlist 使之变为 runnable。GPU HW scheduler 在 runlist 级别以 TSG 为粒度进行 round-robin timeslicing——每个 TSG 获得一个时间片（timeslice），在该时间片内 TSG 所属的所有 channel 被循环扫描以寻找待处理命令。TSG 的设计使得 GPU 可以在多个 task（context）之间公平分配 GPU 引擎时间，同时通过 scale/timeout 参数控制每个 TSG 的调度权重和时间片长度。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TSG 在 GPU 调度管线中的位置（映射到 Fig.4 的 Step ①→③）：

```
Step ①: Task初始化 → 创建pushbuffer → 封装为channel → 多channel封装为TSG → TSG插入runlist
Step ③: GPU HW scheduler对runlist中TSG进行round-robin → 活跃TSG获得timeslice
        → 扫描TSG内所有channel的pushbuffer → 发现pending命令 → dispatch
```

TSG 的关键属性：
- **Scale**：控制 TSG 的调度权重（nvdebug 输出中的 Scale 字段，如 Scale: 3）
- **Timeout**：时间片超时值（如 Timeout: 128）
- **Length**：TSG 内包含的 channel 数量（如 Length: 1）

基于 Bakita & Anderson 的 R4 规则：每个 runlist 最多有一个 TSG（即一个 task）处于 active 状态 per 关联 engine。这是通过 runlist 级别的 TSG round-robin 互斥调度实现的——同一时间只有一个 TSG 的 channel 被扫描和 dispatch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TSG 在 GPU 硬件中作为 runlist 条目存储在 GPU 物理内存中。每个 runlist 条目包含 TSG 标识符、scale、timeout 和 channel 列表指针。NVIDIA 专利 US 9,442,759（"Concurrent execution of independent streams in multi-channel time slice groups"）详细描述了 TSG 的实现机制。在 CUDA 使用中，TSG 由 CUDA 驱动自动创建和管理——每个 CUDA context 默认创建一个 TSG，所有该 context 的 channel 归入此 TSG。但 Bakita & Anderson 在 Jetson TX2 实验中发现，嵌入式平台上 CUDA 创建的默认 channel 数（2-4）和 TSG 配置可能与离散 GPU 不同，可能影响实时系统的并行度。nvdebug 工具可以检查和修改 TSG 及 runlist 配置。

涉及论文标题：
- Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management
