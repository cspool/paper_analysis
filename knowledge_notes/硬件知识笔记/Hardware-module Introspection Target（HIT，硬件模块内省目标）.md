## Hardware-module Introspection Target（HIT，硬件模块内省目标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HIT 是 IPU 架构中"被内省的硬件模块"的定义：一个 IPU 负责监控芯片上某个具体硬件组件（如核内子模块、L2 控制器、GPU SM 子模块、CPU 核等），该组件即为其 HIT，一对一关联（one IPU per HIT）。设计时工程师为每个 HIT 选择最多 32 个需要暴露的微架构内部信号（如 CPU 前端的 fetch-PC 与 miss 标志、cache 控制器的请求总线/逐出/一致性状态、GPU scoreboard 的功能单元活动标志），这些信号经 ABI Spec 文档化（信号语义+数据到达率），成为 introspection 程序可订阅的原始数据源。HIT 设计者不需要预判未来分析任务——只需按三轴方法论（datapath 值如地址/操作数字段、控制信号如 stall/flush/valid/分支结果、状态指示如队列占用/buffer 满/功能单元活动）覆盖主要流水线阶段与资源瓶颈点暴露内部状态，IPU 的可编程性就能从中组合出设计者从未预料的分析。P&R 时 IPU 被 flatten 进 HIT 层级（图 2(b)），信号走线短、开销可忽略；信号到达率与 IPU 处理速度不匹配时按 ABI Spec 暴露的数据率，用采样/聚合/事件稀疏性适配（IPU 不能 stall HIT）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HIT 在 IPU 架构中的运转流程：设计阶段——确定 HIT 类型（CPU 前端/cache 控制器/GPU SM scoreboard/L1D cache+prefetcher 等），按三轴方法选 32 个以内信号接入 IORegs（论文四大演示实际只用 2/17/3/6 个信号，均未到 32 上限），写 ABI Spec（含每信号位宽、寄存器编号、语义、数据率）；制造阶段——IPU flatten 进 HIT 层级做 P&R，远距离信号用标准缓冲技术（如 2mm² 方形 HIT 信号需走 ~2.8mm，高频设计加 1 级 FF 缓冲，且该信号缓冲要求所有 HIT→IPU 信号同步缓冲，可用 P&R 反馈挑选信号规避时序问题）；运行阶段——HIT 每周期把选中信号（含 valid 位指示新数据到达）推进 IORegs，IPU 按 ABI Spec 语义处理。具体例子（Capability 4）：HIT 为 L1D cache+prefetcher 复合体，暴露 6 个信号=132 bits：demand miss 指示（1 bit）、Gaze 预取器的 Accumulation Table（AT）hit 信号（1 bit）、Pattern History Table（PHT）的 missed_in_pt 标志（1 bit）、MSHR 预取 in-flight 状态（1 bit）、LSQ 来的 demand miss 虚拟地址与 PC（地址总线）；IPU 据此把每次 demand miss 分类到 4 类根因（cold region/no learned pattern/late prefetch/prefetch failure）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HIT 由芯片设计者在其 IP 设计中预留信号接口（与现有性能计数器的信号源同源，论文指出 GPU 的 NVIDIA Nsight Compute 计数信号与 SM 内部分数器证明这些信号在合理实现中现成可得、不具破坏性）；ABI Spec 是 HIT 信号的形式化文档（部分示例见表 I：itlb-miss→x0、icache-miss→x1、recycle→x9、fetch-pc-head→x10 等 64-bit 信号），closed 策略下可不对公众发布。使用方式：HIT 信号作为 introspection 程序的"原始数据源"，N 个 PMU 事件得 N 个固定答案，而 N 个 HIT 信号得组合爆炸的分析程序空间；设计者可造 IPU 变体（不同信号集）并随机选择集成，规模化后在不增开销的前提下提升分析灵活性；多路复用更多信号以突破 32 上限列为未来工作。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
