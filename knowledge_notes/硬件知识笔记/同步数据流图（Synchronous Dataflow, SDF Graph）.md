## 同步数据流图（Synchronous Dataflow, SDF Graph）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SDF 图是一种数据流模型：图结构静态已知，节点（actor/任务）每次点火（firing）消耗/产生固定数量的 token（在仿真场景中是固定数量的输入值），因此可编译期静态调度、周期化无死锁执行。源自 Lee & Messerschmitt（Proc. IEEE 1987，"Synchronous Data Flow"），是 DSP 设计环境（Ptolemy、COSSAP）的经典建模基础，现代如 ARM CMSIS-DSP 的 SDFTools 用其生成静态调度与 FIFO 大小。区别于通用数据流图（DFG，见编译框架层条目，通常是编译 IR、动态/静态混合），SDF 强调每边固定 token 率与周期可重放。
- 在 Lotus（ISCA'26）中：循环级仿真把电路表示为 SDF 图——节点=组合逻辑（消费一个或多个输入值，全部到达即点火）、边=通信（wire 是 0 周期边、register 是 1 周期/跨周期边），图每个模拟周期被反复求值。示例 y[n]=a·x[n-1]+b·y[n-1] 的 2 级流水线被编码为 mult/add 两个任务 + Reg（跨周期）/Wire（同周期）边。Lotus 的 DSL 中 Reg<T>（跨周期边，带初值）与 Wire<T>（同周期边）直接对应 SDF 的跨周期/同周期边。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Lotus 中的运转：①编译器把 SDF 图映射到 tile（每个任务固定一个 tile，静态映射）；②任务单元在配置时装载图信息（每任务的 taskId、函数指针、输出 token 目的地、输入存储范围）；③每个模拟周期：所有任务的输入 token 汇聚 → 就绪任务派发 → 输出 token 沿边发送；④跨周期边（Reg）使输出在下一模拟周期成为下游输入，同周期边（Wire）在周期内传播——两版本（奇/偶周期）输入存储使连续周期重叠执行。图的静态性使 taskId 可直接索引存储、分层位图可实现低代价优先级队列、选择性执行可比较奇/偶周期输入。
- 与 DSP 中 SDF 的差异：DSP 的 SDF 用于静态调度（编译期确定点火顺序）；Lotus 的 SDF 图每个模拟周期完整重放一次，执行顺序由任务单元在运行时按依赖与优先级动态决定（因仿真的输入每周期都变、且要支持内存访问等动态延迟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Lotus 用 C++ 嵌入式 DSL 定义 SDF 图（genDfg() 中用 task()/Reg/Wire 构建），编译器执行图定义代码发出图的文本表示（复用 C++ 编译器做类型检查）；或改版 Verilator 从 Verilog 提取数据流图。Web 证据：SDF 静态调度理论见 Lee 的 IEEE TC 论文 "Static scheduling of synchronous data flow programs for digital signal processing"（https://www.osti.gov/biblio/7190136 ），现代实现见 ARM CMSIS-DSP SDFTools（https://raw.githubusercontent.com/ARM-software/CMSIS-DSP/refs/tags/v1.11.0/SDFTools/README.md ）。
- 使用场景：DSP 流处理（音频/软件无线电）、电路与系统级仿真的建模表示、数据流加速器编译前端；Lotus 把它扩展为跨芯片分布式执行的多 FPGA 仿真执行模型。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
