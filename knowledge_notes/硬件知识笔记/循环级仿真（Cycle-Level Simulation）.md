## 循环级仿真（Cycle-Level Simulation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 循环级仿真指把完整数字设计按"模拟周期"（simulated cycle）逐周期推进的仿真方式：每个模拟周期内评估组合逻辑与寄存器/内存状态的变化，周期边界处采样更新。它是 RTL 仿真（Verilator、VCS 等把 HDL 编译成 cycle-based 模型）的核心语义，也是周期精确微架构仿真器（如 timing-first 仿真）的基础。区别于事件驱动仿真（按信号翻转事件推进、支持亚周期时序）与事务级/指令级仿真（更高抽象、不逐周期）。
- 在 Lotus（ISCA'26）中：仿真目标是把待验证的数字设计编码为同步数据流图，图被反复求值——每个模拟周期消耗一组输入、产生一组输出。论文把"每个模拟周期内的求值"切分成成千上万个微小任务，分散到多 FPGA 上的数千核并行执行。衡量指标是仿真速度（KHz，即每秒推进多少模拟周期）：CPU 基线 46–493 KHz、emulator 基线 800 KHz、Lotus 485–2474 KHz（DSL benchmark）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Lotus 每个模拟周期）：①各 task unit 收到上一周期产生的 token（携带 cycleId/taskId）；②输入单元按 cycleId 写入对应版本（奇/偶周期双版本存储，允许连续周期重叠执行）；③就绪任务按优先级派发到 RISC-V 核执行任务函数；④输出单元按边产生输出 token；⑤跨周期边（Reg）的 token 在下一模拟周期才被消费，同周期边（Wire）的 token 立即（或同一周期内）被消费。核执行期间可访问 tile 内共享内存（模拟大内存/缓存），任务结束自失效 L1 保证一致性。
- 相比多核 CPU 的循环级仿真：CPU 上并行化需共享内存同步（Verilator 多线程同周期边通信开销大、扩展到几十线程即失效；RepCut 缓解）；Lotus 用专用 task unit 硬件解耦通信与计算，把同步成本从"每次通信都发生"降为"token 网络异步传递"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：软件端（Verilator 编译 Verilog→多线程 C++、Lotus DSL 编译器生成 task 程序）；硬件端（仿真器把 RTL 综合成门并空间映射到 FPGA；Lotus 用通用核时间映射执行任务）。Web 证据：Verilator 是 cycle-based 仿真器，只在时钟周期边界建模（https://www.veripool.org/verilator/ ）；硬件仿真器如 ZeBu/Palladium 以 MHz 级跑 cycle-based 验证（https://www.cadence.com/ 与 https://www.synopsys.com/ 产品页）。
- 使用场景：芯片 RTL 功能验证、SoC 软硬件协同验证、微架构评估；Lotus 还支持 RTL 之外的周期级建模（如 Multicore benchmark 用更高抽象建模 mesh 路由器时序与 buffer 占用）。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
