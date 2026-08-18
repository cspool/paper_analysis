## 选择性执行与保守同步（Selective Execution & Conservative Synchronization, CMB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 选择性执行利用设计活动因子（activity factor）低的特性跳过无效工作：很多硬件设计多数周期只有部分信号变化，若任务输入与上一周期相同则重复执行是浪费。实现分为保守同步与乐观同步两类：保守同步（Chandy-Misra-Bryant/CMB，即 Null Message 算法）检测输入未变 → 跳过任务 → 发 null token 告知消费者复用旧值；乐观同步（Time Warp，Jefferson 1985）假设缺失 token 来自被跳过的任务、直接执行，迟到 token 触发回滚重执行。
- 在 Lotus（ISCA'26）中：采用 CMB 保守同步实现选择性执行——任务单元输入单元比较非 null token 与旧值是否一致，一致则可跳过；null token 把旧输入值复制到当前版本；被跳过任务由调度器直接送输出单元发 null 消息。需要 4 级流水线全吞吐执行这些操作（分离奇/偶周期输入存储）。非投机实现逻辑仅占每 tile 的 1%，而 ASH 的 Time Warp 乐观同步在 FPGA 上约 2× 开销。限制：有副作用的任务不可跳过；收到非 null order token（读后写依赖）的任务即使输入一致也必须执行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一个可跳过的任务调用）：①输入单元收到该任务本轮全部输入 token；②逐个比较输入值与上一周期（另一版本存储）值；③全部一致且无副作用/无非 null order token → 不派发核，直接把该调用送入输出单元；④输出单元为每条输出边发 null token；⑤消费者输入单元收到 null token 时把旧输入值复制到当前版本（等同未变化）。核心是"每模拟周期比较奇偶周期输入存储 + 4 级流水"。
- 效果（论文 Fig.12）：对 Multicore（活动因子低，缓存缺失导致模拟核停顿、缓存/路由器组件活动低）收益最大——指令数减少 3.3×、加速 2.3×；对 NTT/MatMult 等每周期输入都变的流水线无收益（比较本身也耗资源但无效果）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：任务单元内的比较逻辑 + 双版本输入存储 + null token 传播路径；token/任务元数据扩展（副作用标记、order token 类型）。Web 证据：CMB 是并行离散事件仿真（PDES）的奠基性保守同步算法，靠"保证不再发送更小时间戳消息"的 null message 推进（RPI 课程笔记 http://www.cs.rpi.edu/~chrisc/rc-web/node4.html ；Conservative Synchronization Methods for Parallel DEVS and Cell-DEVS https://cell-devs-02.sce.carleton.ca/publications/2011/JW11a/summersim11.pdf ）。
- 使用：适用于活动因子低的大规模设计（多核、存储系统）；Lotus 用它减少无效执行，同时保持每个模拟周期的语义正确性（null token 传递"输入未变"信息）。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
