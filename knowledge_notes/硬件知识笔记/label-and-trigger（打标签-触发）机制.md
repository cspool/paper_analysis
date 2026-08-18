## label-and-trigger（打标签-触发）机制

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 管理同步集体演化的核心机制：把"配置"与"执行"分离——N_LOCK/C_LOCK 是 labeling（打标签）指令，只把锁存掩码（NLM/CLM）写入对应组的标签寄存器，不产生任何物理动作；N_EVOLVE/C_EVOLVE 是 trigger（触发）指令，把预设掩码一次性原子施加到 DSU，使所有未锁定组件从同一同步状态开始并行演化。若没有这一分离，先锁后演化逐组件串行进行，各组演化起点不一致，破坏"所有参与节点/耦合同步演化"的计算正确性要求。掩码写满后触发时选择逻辑从默认 Idle Registers（输出锁定信号）切换到标签寄存器（NLM/CLM），完成"空闲锁定 → 掩码控制演化 → 计时归零回到空闲锁定"的状态机。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件流程（一次 A1 推理的锁定与触发阶段）：① N_LOCK[NLM Addr, NGID]：NLM 从 Mask Memory 取出放上 Node Lock Mask Bus → NGID Select 选中输入节点组 → 掩码写入该组 NLM Registers（此刻 Idle Registers 仍活跃，全部节点保持锁定，无物理变化）；② C_LOCK 同理给对应耦合组写 CLM；③ N_EVOLVE[GM Addr, Time]：GM 从 Mask Memory 取出 → 选中组（输出组）的选择逻辑从 Idle Registers 切换到 NLM Registers → 输入节点仍锁、输出节点解放 → Time Bus 同步写 Time Registers 倒计时 → 全体输出节点从同一时刻开始连续时间演化。阻塞策略配合：_LOCK 对后续 _EVOLVE 施加黄阻塞（RAW），保证触发前标签已就位——这是该机制的指令调度保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上靠"标签寄存器 + 选择逻辑 + 空闲寄存器"三件套；数据通路与控制通路分离（Data Path 走 DAC/ADC、Control Path 走 mask 寄存器），使 LOCK 与数据/配置/其他 LOCK 指令可并行（绿放行）。使用方式：任何需要同步集体演化的 DSU 任务（推理的节点演化、训练的耦合演化）都必须"先 _LOCK 后 _EVOLVE"；多个组可通过 GM 一次触发或分次触发实现级联（A2 多层推理按层触发）。演化结束时选择逻辑回退 Idle Registers 自动锁定演化结果，天然免去显式"停止"指令。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
