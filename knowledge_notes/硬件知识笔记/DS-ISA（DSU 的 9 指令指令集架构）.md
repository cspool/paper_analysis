## DS-ISA（DSU 的 9 指令指令集架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 是第一个面向 Dynamical System Unit 的标准指令集架构：9 条极简指令，分三类逻辑类别——节点生命周期 N_LOAD/N_LOCK/N_EVOLVE/N_STORE（数据加载、锁存掩码、触发演化、读回状态），耦合生命周期 C_LOAD/C_LOCK/C_EVOLVE/C_STORE，加连接配置 CFG_CONN（用 Connection Mask 定义节点组间的耦合连通拓扑）。指令格式为定长 64-bit、三型（E/N/C-Type）：Opcode 1B | Imm_address 4B | Imm_time 或 Imm_NGID 或 Imm_CGID_col+row 2B | Reserved 1B。核心设计是"间接控制"：NLM/CLM/CM/GM 等 mask 尺寸随组规模/组数线性增长、无法嵌入指令，故指令只携带 32-bit 地址指向片上 SRAM（Mask Memory）中的 mask 数据；2B 立即数字段只装规模对数级增长的 NGID/CGID/演化时长。语义上属于 configurative（配置型）ISA：不规定算术步骤序列，而是配置拓扑、边界条件与演化时长，计算从集体物理演化中涌现——区别于 CPU/GPU 的 prescriptive（规定型）ISA（显式离散数字操作序列）、神经形态的事件驱动 spike ISA（Loihi/TrueNorth，仅在 spike 到达时计算）、以及 CIM 的 operation-centric ISA（显式暴露 MVM/卷积算术 kernel，如 MNEMOSENE/ReVAMP）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
每条指令在控制器微架构中的数据通路（Fig.10）：N_LOAD——NGID 送 NGID Select 激活对应节点组，地址从 Data Memory 取数，经 DAC 广播到 Input Bus 被选中节点捕获；N_STORE 反向——选中组驱动 Output Bus 经 ADC 写回内存。C_LOAD/C_STORE 复用同一数据通路，但用 CGID_col + CGID_row 分别激活列/行 Selection 精确定位耦合组（耦合组元素多于节点组，分多步编程）。N_LOCK——NLM Addr 从 Mask Memory 取 Node Lock Mask 放到 Node Lock Mask Bus，写入该组 NLM Registers（默认 Idle Registers 输出锁定信号防空闲演化，只有 _EVOLVE 才切换选择逻辑到 NLM Registers）。N_EVOLVE/C_EVOLVE——GM Addr 取 Group Mask 决定哪些组收演化信号，切换选择逻辑到标签寄存器，同时 Time Bus 写 Time Registers 启动倒计时；计时归零切回 Idle Registers 锁存演化结果。CFG_CONN/C_LOCK——CGID 定位耦合组，分别把 CM/CLM 写入 Col/Row CM/CLM Registers。C_EVOLVE 关键优化：同一 1D Group Mask 对称同时用于行/列组选择，激活对应交互子矩阵（物理语义：耦合是节点对的交互而非独立计算元件，选中一组节点演化时其两两耦合须同时参与），省去第二条大地址指针，并天然支持碎片化资源分配与磨损均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog 描述的 DS-ISA 数字控制器，Yosys 综合 + OpenROAD 物理设计，SKY130 130nm 工艺，200 MHz；评估 32 组 × 8/16/32 节点（最大 1K 节点 + 1M 耦合），全配置控制器 4.6W、79.58mm²。使用方式：主机按 load-lock-evolve-store 模型编排指令序列下发（例如 ML 推理 7 条指令）；多任务并行用 GM 分区（B2）、微调用 CLM 部分演化（C2）、多阶段工作流用 N_STORE/N_LOAD 显式复用节点中间态；控制器 scoreboard 按阻塞策略让独立指令重叠执行。论文未开源 RTL（联网搜索无公开仓库）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
