## ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

- baseline方法是什么？
  **单 CUDA Stream 串行执行**：所有 GPU kernel 被发射到同一个 CUDA stream 中。CUDA runtime 保证同一 stream 上的 kernel 按发射顺序串行执行，consumer kernel 必须等待 producer kernel 完全结束后才能开始。这是 PyTorch、TensorFlow 等主流 DL 框架的默认行为。

  全栈执行例子（以 Deep RL Brax Ant 物理仿真的一次 training batch 数据生成，RTX 3060 28 SM）：
  - **模型推理算法层**：物理仿真计算刚体碰撞检测、关节力矩、接触力等的多个小 kernel（每个 kernel 通常 < 200 CTA）。Brax/JAX 实现。
  - **系统框架层**：Brax/JAX 将所有仿真 kernel 发射到单一 CUDA stream，无并发调度。程序员无法提前知道完整的计算图（每次仿真 input 不同导致不同的接触/碰撞计算路径）。
  - **编译框架层**：论文未明确说明编译框架层修改。使用标准 nvcc/JAX 编译器。
  - **kernel调度层**：大量小 kernel 在单一 stream 中串行执行。例如 Ant 环境一次 batch 生成需要数百个 kernel launch，每个 kernel 仅有少量 CTA（中位数 < 200 CTA），远不能填满 28 SM。GPU 实际达到的 occupancy 仅约 34%（Ant 环境，RTX 3060），即约 66% 的 SM 计算资源被浪费。每个 kernel 执行时间短，kernel launch 延迟（~5-20μs）相对于 kernel 执行时间不可忽略。
  - **硬件架构层**：标准 NVIDIA GPU 命令处理器按序从命令队列中取 kernel 发射。无 inter-kernel dependency 信息，同一队列内 kernel 严格串行。论文未明确说明硬件架构层自定义修改。

  Baseline 问题两重：(a) **Input-dependent 计算图**：kernel 间依赖关系随每次 input 变化，无法提前构建完整的 kernel 依赖 DAG。CUDA Graph 虽能消除 launch/sync 开销，但 DAG 构建耗时达执行时间的 47%（Brax），不适合每 input 重新构建。(b) **不规则依赖导致细粒度调度**：计算图不规则（非简单独立 partition），使用多 stream + cudaStreamWaitEvent 细粒度调度会产生大量 CPU-GPU 同步开销（每次同步 5-20μs）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **ACS（Automatic Concurrent Scheduling）——运行时乱序 kernel 调度**：在固定大小的调度窗口内（类似 CPU 乱序指令调度），对顺序发射的 kernel 进行运行时依赖检查和乱序并发调度。kernel 在调度窗口内被标记为 ready/pending/executing 三种状态，当 upstream kernel 全部完成时标记为 ready，由 scheduler 并发发射到多个 CUDA stream 或 hardware dispatch unit。

  全栈执行例子（同样 Brax Ant 仿真，RTX 3060 28 SM，ACS-SW 窗口大小 32）：
  - **模型推理算法层**：物理仿真计算逻辑不变。程序员通过 ACS_wrapper 为每个 kernel 标注 read_segments 和 write_segments（起始虚拟地址+大小）。对于常见 kernel（矩阵乘、卷积、加法等），segments 从函数原型直接可得；对于间接内存访问的 kernel，标注为访问全部 GPU memory（保守但保证正确）。
  - **系统框架层**：ACS-SW 作为用户态运行时系统替代 JAX/Brax 的单 stream 执行。应用线程调用 kernel 时，先调用 `get_addresses()` 将 kernel 参数解析为实际虚拟地址范围，然后将 kernel + segments metadata 送入输入 FIFO 队列。CPU 上的 window module 线程维护调度窗口和依赖关系，scheduler module 线程（可配置数量，每个绑定一个 CUDA stream）轮询窗口获取 ready kernel 并发发射。
  - **编译框架层**：论文未明确说明编译框架层修改。ACS_wrapper 的 `get_addresses()` 函数可由程序员手写或通过 GPUOcelot 等二进制分析工具自动提取。依赖检查算法为 O(segments²) 遍历 read+write segments 对检查地址范围重叠。
  - **kernel调度层**：调度窗口（大小 N=32）中，每个 kernel 维护一个 upstream kernel 列表（依赖的 kernel ID）。当 kernel A 的 write segments 与 kernel B 的 read/write segments 有重叠时，A 被加入 B 的 upstream list。kernel 完成时，window module 遍历窗口中所有 kernel 的 upstream list 移除已完成 kernel。Upstream list 变为空的 kernel 被标记为 ready，由 scheduler 发射到空闲的 CUDA stream。多个 CUDA stream 上的 kernel 在 GPU 上真正并发执行，原本少量 CTA 的小 kernel 现在可以并行填满 GPU SM。ACS-SW 达到平均 1.56×（最高 1.87×）加速，occupancy 从 34% 提升至接近满载。
  - **硬件架构层（ACS-HW）**：在 GPU 命令处理器中增加硬件调度窗口（1KB SRAM for N=32）：每个 slot 包含 8-bit kernel ID 及 (N-1) 个 8-bit upstream kernel ID（全关联存储），2-bit 状态（ready/pending/executing）。upstream load module 修正 CPU 端可能 stale 的 scheduled_list（移除已完成 kernel，阻塞超过 M 个新 kernel 插入以防遗漏）。kernel 完成时硬件在 N-1 cycle 内更新所有 slot 的 upstream list。Ready kernel 直接被 hardware dispatch unit 发射。消除 CPU-GPU 同步和 kernel launch 开销（原本 5-20μs/次）。ACS-HW 达到平均 1.79×（最高 2.19×）加速。端到端 Deep RL 训练加速 1.42×（ACS-HW）和 1.30×（ACS-SW）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: input-dependent 计算图导致无法提前静态调度** → 方案：运行时滑动窗口调度（类似乱序执行），每次仅检查和调度窗口内有限数量 kernel（N=32），延迟低（依赖检查 410ns~1640ns），不依赖完整 DAG 预构建。
  - **defect: 多 stream 细粒度调度的 CPU-GPU 同步开销大（5-20μs/次）** → 方案：ACS-SW 通过固定数量 scheduler 线程 + stream 复用减少同步次数；ACS-HW 将调度完全移到 GPU 硬件内，消除 CPU-GPU 往返。
  - **defect: persistent threads 无法支持异构 kernel** → 方案：ACS 通过 CUDA stream 发射原生 kernel，每个 kernel 保持自身的寄存器/shared memory 配置，无 PT 的同质性限制。
  - **defect: CUDA Dynamic Parallelism 仅支持父子依赖** → 方案：ACS 依赖检查支持任意 kernel 间的多对多依赖关系（通过 read/write segments 重叠检测）。
