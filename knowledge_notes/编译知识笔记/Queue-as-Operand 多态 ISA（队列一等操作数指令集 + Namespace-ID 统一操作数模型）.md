## Queue-as-Operand 多态 ISA（队列一等操作数指令集 + Namespace-ID 统一操作数模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morphatron 的 32-bit 指令集，通过三组机制把"加速器多态"暴露给编译器：(1) morpha 选择与重配指令——Polymorphic_Exec_Config 指定参与核的 XY 坐标与进入的 morpha，MORPHA_STATIC_CONFIG（数据流确定时编程 crossbar 方向位）与 MORPHA_DYN_CONFIG（graph/tree 时编程地址比较器的 [lower, upper] 地址范围与输出端口）；(2) 队列指令组——INIT_Q/FREE_Q（分配/释放，256 个架构队列）、POP_Q/PEEK_Q/PUSH_Q/IS_EMPTY_Q/SIZE_Q、Q_CONDITIONAL_PUSH、Q_LOOP_UNTIL_EMPTY（携带 INSTRUCTION_COUNT 告知硬件循环体大小），共享队列（INIT_Q 第三字段）+ REMOTE_STORE/REMOTE_READ 支持跨核队列通信；(3) Namespace-ID 统一操作数模型——计算指令不用寄存器，每个操作数以 (Namespace, ID) 对编码（Namespace 选 tensor/queue/scalar 底层结构，ID 定位实例，由 Queue Manager/迭代表解析），同一 ADD 等指令可跨 queue/vector/graph/tree morphas 使用而不暴露各形态的存储布局。另有 SET_ITER/SET_STRIDE/SET_BASE_ADDR 与 Loop Control 配置仿射迭代表、DMA 配置（LD_CONFIG_BASE/STRIDE/ITER）、collective 配置（BROADCAST/COLUMN_REDUCE）、SYNC（code_block_end 隔离 morpha 间同步屏障）。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程：regular 访存负载复用 in-house tensor compiler（prefetching、memory tiling、operator fusion 等标准张量优化）；queue-centric 负载先在软件层按 Morpha Core 的队列语义重写并做功能正确性验证，再由作者开发的编译器把软件 queue 函数直接映射为 Morphatron 指令。以论文给出的 SSSP processEdges 指令版为例：INIT_Q q0/q1（非共享）、q2（共享）→ 标量段 LD_Q 装载 → SYNC 切 SIMD 段 → `Q_LOOP_UNTIL_EMPTY, q0, 3` 包裹 POP_Q/ADD/PUSH_Q 循环体 → `Q_LOOP_UNTIL_EMPTY, q1, 1` + REMOTE_STORE 把更新写入远端共享队列 → SYNC exe_end。编译器还负责静态编程 DMA（vector/systolic 的 prefetch/double-buffering）与插入 barrier。论文明言"Compilation for accelerators remains an open problem"，并指出 STeP（把队列升为程序级一等构造、面向空间数据流加速器）与本文 queue-as-operand ISA 天然契合，是可期的更高层编程路径。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 与常规 load/store 向量 ISA 对比：传统向量机每个循环迭代要算地址（pointer bump）、把数据搬进向量寄存器文件、再用 sub+bnez 管循环——本文由迭代表硬件生成地址、scratchpad 直接供操作数、前端硬件管循环（消融：这三笔开销分别占 28%/15%/19% 平均运行时，本文一次性 per-kernel 配置开销最多 0.29%）。与 CGRA 对比：CGRA 把数据流图空间映射到 PE（布局布线 + 复杂互连），本文每个 morpha 是指令驱动的独立加速器、可用常规工具链编译。实现/使用：ISA 32-bit 编码含 Opcode 分组字段；编译器输出 morpha config 指令序列驱动运行时切换。编译器与工具链是否开源论文未明确说明（Web 未找到）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
