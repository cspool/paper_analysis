## ASU（Associative Sequencing Unit，关联定序单元）

术语解释
BAAP 对 DPU 解码逻辑的扩展：缓冲已发射指令并内置每条 AP 指令的优化微码序列（源自 Hyper-AP 的全栈优化），把表 IV 的宏指令展开成 CAM 搜索/更新信号序列逐周期驱动 AP 外围逻辑。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ASU 位于解码级与 AP 数据通路之间的定序器。BAAP 扩展 ISA 的 ap_* 指令本质是宏指令：AP 硬件唯一原生操作是 search/update（置 wordline、锁存 tag、按 tag 掩码位线写回），因此每条宏指令对应一个微码序列（ap_xor 4 步、ap_add 8n+2 步、ap_mul 4n²+4n 步、ap_regex <m−k 步）。ASU 存这些优化后的微码（优化方法源自 Hyper-AP [57] 的全栈搜索-更新序列压缩），操作数就绪后逐周期断言控制信号。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程（论文 §III-B1）：DPU 取指 ap_add → 解码 → ASU 缓冲并索引该指令微码（8n+2 条 search/update 步骤）→ 每周期发一条微操作到 32x36 子阵列链（搜索：置 WLL/WLR 与 key、锁存 tag；更新：按 tag 掩码位线写回）→ 序列完成才提交。作用：标量核无需逐周期干预 AP，长位串行算术期间 DPU 可处理控制流，DMA/VPU 并行工作；表 IV 的每指令 AP 周期数即 ASU 微码步数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现形式论文未给出门级描述（推测为解码器内微码 ROM + 指令缓冲状态机）。同类可参照：CAPE 的控制流水线、Duality Cache 的 SIMT 前端、EVE 把 RISC-V 向量扩展映射到存内微操作。使用价值：把可编程 ISA 与底层 CAM 操作解耦，编译器只看到表 IV 指令；与 VPU 配合实现"位串行计算 + DMA 访存"重叠。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
