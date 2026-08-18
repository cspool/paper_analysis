## M100: An Orchestrated Dataflow Architecture Powering General AI Computing

- 属于编译框架的实现是什么？实验比较什么？
  - M100 AI compiler toolchain（与硬件 co-design 的自研编译器栈）：① space-time scheduler —— 把 NN 子图映射到 M100 NPU 硬件；必要时把大 tensor 沿多轴维度分解成 mini-tensor，按时间调度的阶段流经空间分配的 TPB 流水线（Fig.14：子图 4 个算子 OP1~OP4 分给 4 个 TPB，输入 tensor 维度分解后按时相流动）；② graph compiler —— 图优化（tensor fusion、死代码消除、代数化简、layout 变换）+ 动态 tensor 的动态内存分配；③ backend compiler —— C 扩展编译器，生成利用 M100 硬件能力的 intrinsic 指令（tensor 计算、数据搬运、同步）。运行时侧：NPU 固件（RISC-V 上）基于工具链生成的二进制用 JIT 编译动态生成优化 TPB 指令，实时计算 tensor shape 与存储地址，并把 TPB 指令下发给任务分配的 TPB 组；AI inference runtime 与 NPU driver 运行在 ARM Cortex-A78 上。
  - 实验比较：论文不独立比较编译器版本，而是通过端到端在 M100 实机上跑 UniAD / LLaMA2-7B / MindVLA 与 NVIDIA Thor-U 对比（结果见实验_硬件架构 条目），验证"编译器编排 dataflow"（松散有序指令流 + 数据就绪驱动执行）带来的高并行度、低同步开销与高硬件利用率；TCO 与可扩展性亦为主要动机。
  - 硬件平台：M100 SoC/NPU（TSMC N5A，14 TPB cluster；UniAD 用 8 cluster、LLM 用 12 cluster）vs NVIDIA Thor-U（TSMC N4），同功率预算。
  - 框架与修改：全部为自研新增（非修改开源框架）——M100 编译器工具链 + AI inference runtime（输入准备/模型加载/任务踢发/结果后处理/FuSa 错误监控）+ NPU driver（硬件抽象层）+ NPU firmware（RISC-V JIT）。编译产物为 TPB 指令流（数千 bit/条，含 tensor shape 与通信需求元数据），经 ICB 广播。
  - 开源情况：未开源，论文未给出仓库/链接；无法确认任何公开实现。
  - 使用例子（基于论文描述，编译器输入→输出全过程）：输入 = 训练好的 NN 图（如 UniAD 某子图）→ space-time scheduler 把子图空间映射到 4 个 TPB，输入 tensor 按多维分解成 mini-tensor 并沿时间阶段流动 → graph compiler 做算子融合/死代码消除/代数化简/layout 变换并为动态 tensor 分配内存 → backend compiler（C 扩展）为每个 tensor 算子生成 intrinsic 指令（含 tensor 计算、数据搬运、同步操作，附 shape/通信需求等元数据）→ NPU 固件 JIT 依据二进制动态生成最终 TPB 指令、解析 tensor shape 与地址 → 固件经 ICB（64 bits/cycle，destination mask 广播）下发给任务分配的 TPB 组 → TPB 指令队列在数据就绪与同步条件满足时派发（同功能单元保序、跨单元可乱序）。作用：让编译器承担传统 dataflow 架构的编排复杂度，使硬件保持简单（无 cache 一致性、无 register file、无全局执行顺序），同时保留数据流并行的高效率与对新模型的可适配性。
