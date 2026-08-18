## Morpha Core / Queue-Centric SIMD（队列中心 SIMD 执行）

术语解释
- Morphatron 的核心处理器：无寄存器文件的深流水 SIMD 管线，把 FIFO 队列升为一等 SIMD 操作数，用 interleave 划分子队列在保持 FIFO 语义的前提下并行执行。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 核心洞察：FIFO 语义只约束"顺序"而不约束"执行"——队列可并行化而不破坏顺序保证。做法：逻辑队列按 interleave 划分成 N 个子队列（11 元素分 4 路：⟨0,4,8⟩/⟨1,5,9⟩/⟨2,6,10⟩/⟨3,7⟩），每子队列映射到一个执行 stage，指令以 queue ID 引用操作数，pop/peek/push 直接在各 stage 的 scratchpad sub-bank 上完成，执行完队列在内存中自然保持原序。Morpha Core 每个执行 stage = ALU + 8 KB scratchpad sub-bank + Queue Manager（组合逻辑），前端为 3 级流水（取指/译码/内存分配），彻底消除寄存器文件与 cache 供给。对比传统向量化队列的两条路：(a) 多线程 MIMD——指令簿记、线程调度、控制复杂度高；(b) 向量处理器 SIMD——三笔开销：子队列满/空检查指令序列、向量寄存器文件搬运与地址计算、多子队列分支循环控制（本文消融：地址计算 28%、RF load/store 15%、循环管理 19% 平均运行时开销）。横向链 4 核成一行 = 32-wide SIMD 管线。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 以 `Q_LOOP_UNTIL_EMPTY, Q.ID[#0], 1: ADD, Q.ID[#1].PUSH, Q.ID[#0].POP, 0x01` 为例的运转流程：前端反复发出循环体指令（Q_LOOP_UNTIL_EMPTY 由硬件管理循环，程序计数器仅在所有 Queue Manager 报"空"后前进）→ ADD 指令流到每个执行 stage → 该 stage 的 Queue Manager 按 queue ID 索引本地队列表取 head 指针 → 从 scratchpad sub-bank 读操作数直接喂 ALU（无地址计算、无 RF 搬运）→ 结果按 tail 指针写回并推进 → 某 stage 子队列已空则自动 skip 该次计算。动态扩容：INIT_Q 在第三流水级触发 8 棵 buddy-tree 分配 256 B slice，slice 索引随指令下传登记进各 stage Queue Manager。控制流以数据可用性表达：Q_CONDITIONAL_PUSH 按条件选择性入队，替代 predication 掩码（SLAM 点云过滤 30% pass rate 时传统 predication 浪费 70% 周期，本设计 2.4× 加速）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：每 stage 独立管自己的子队列 → 子队列长度不均衡被透明吸收（最忙 lane 决定整体推进，1.6× 不均衡在 60% pass rate 时带来 26% 开销）；队列管理逻辑（Queue Manager）与数据存储（sub-bank）解耦 → sub-bank 可复用于 vector/systolic morphas。使用：覆盖 data-dependent semi-regular 访存与 coarse-grained/structured 并行（如 SLAM 过滤、MPC 候选过滤、A*/BFS frontier）。Web 证据：论文之外未见同名公开实现；概念上可与 decoupled access-execute 架构中的队列使用对比（后者用队列做访存/计算单元同步，不把队列当计算操作数）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
