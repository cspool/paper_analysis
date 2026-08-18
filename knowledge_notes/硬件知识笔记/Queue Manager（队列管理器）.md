## Queue Manager（队列管理器）

术语解释
- Morpha Core 每个执行 stage 上的纯组合逻辑模块，独立维护其子队列的 head/tail/size/empty 状态，并负责把队列操作数直接供给 ALU，是"队列成为一等操作数"的硬件载体。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 每个执行 stage 有一个 Queue Manager，内部是本地队列表：以 queue ID 为索引，存储该 stage 上每个活动子队列的 head 地址、tail 地址与 size。指令携带 queue ID 流经流水线时：(1) pop/peek——Queue Manager 用 head 指针访问本 stage 的 scratchpad sub-bank，把元素直接喂给 ALU；(2) push——用 tail 指针把结果写入 sub-bank 并推进 tail；(3) 检测到子队列空则跳过本次 SIMD 运算（解决传统向量机的 lane 掩码问题）。此外 Queue Manager 承担分配器不做的事——追踪"哪个 slice 属于哪个队列"（经 slice 末字链表），并把队列统计（如长度）提供给 DMA 引擎做动态编程。共享队列（INIT_Q 第三字段 = TRUE）+ REMOTE_STORE/READ 指令使 QM 支持跨核队列访问，支撑 graph/tree morphas。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（pop 路径）：前端译码出 `ADD, Q.ID[#1].PUSH, Q.ID[#0].POP, 0x01` → 指令流水下传到 stage k → QM_k 查队列表得 Q#0 的 head 指针 → sub-bank_k 读出 head 元素 → ALU 加 0x01 → 结果写 Q#1 的 tail 指针位置并推进 tail → head 推进。分配/释放：INIT_Q 到达分配级时 8 棵 buddy tree 并行分配 slice，分配的 slice 索引沿流水寄存器传播，被各 stage QM 登记到同一 Q_ID 下。成本：纯组合逻辑，队列控制逻辑 + 硬件分配器合计仅占总能耗 5%；队列/vector 前端（QM、buddy tree、迭代表）合计仅 6% 面积。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现中，队列的 head/tail/满/空由软件维护（指针变量 + 检查指令序列），每轮循环都要重复执行；QM 把这些簿记沉到硬件、单周期完成。使用场景：任何以"动态尺寸集合 + 逐元素条件处理"为特征的负载（候选过滤、frontier 遍历、点云投影）。论文未明确说明 QM 之外是否有等价商用实现（Web 未找到）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
