## load-lock-evolve-store 执行模型

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 论文从各类 DSU 应用（ML 推理/训练、优化、DE 求解）中提炼的统一执行模型，把 DSU 编程归纳为五种基本行为：① Connectivity Configuration（激活/去激活耦合，分配资源）；② Data Loading（写节点初始值与耦合参数）；③ Component Clamping（锁定选定节点/耦合为边界条件）；④ System Evolution Management（触发并控制范围内所有未锁定组件的同步并行演化时长）；⑤ Results Retrieval（读回节点值作解或耦合值作训练参数）。命名 load-lock-evolve-store 来自其核心四阶段。它明确不是固定线性流水，而是可组合阶段：可排列成简单线性序列（ML 推理），也可组织成循环（训练 Evolve-Load 循环换新数据、优化 Evolve-Store-Load 循环做退火数据修改、DE 求解 Evolve-Load 循环更新时变边界条件）。论文用六个运行模式（A1 单向节点演化、A2 级联节点演化、B1 双向节点演化、B2 并行节点演化、C1 耦合演化、C2 部分耦合演化）证明该模型能覆盖全部已知 DSU 负载形态。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行模型落实到 DS-ISA 的指令流。ML 推理（线性序列）的例子：CFG_CONN（Connection Mask 定义输入组→输出组拓扑）→ C_LOAD/N_LOAD（DAC 写权重/输入）→ C_LOCK/N_LOCK（CLM/NLM 锁定输入与对应耦合）→ N_EVOLVE（Group Mask 选输出组、Time 给演化时长，同步触发）→ N_STORE（ADC 读回输出）。ML 训练（Evolve-Load 循环）：加载数据与初始权重 → 锁节点 → C_EVOLVE 让耦合在 EC-Loss 反馈下演化 → N_LOAD 载入新数据 → 循环至收敛 → C_STORE 保存训练后权重。优化（Evolve-Store-Load 循环）：B1 双向演化收敛 → N_STORE 读出解 → 修改数据（退火扰动）→ N_LOAD 重载 → 再演化。DE 求解（Evolve-Load 循环）：演化 → 更新时变边界条件重载 → 再演化。每个阶段在控制器中由一组指令 + 两级掩码实现，循环由主机程序重复发出指令构成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：由 9 指令 DS-ISA 的指令类别直接映射（节点生命周期 4 条 + 耦合生命周期 4 条 + 连接配置 1 条）；"label-and-trigger"机制负责同步性（_LOCK 打标签、_EVOLVE 原子触发）；两级掩码（inter-group GM、intra-group NLM/CLM/CM）负责选择性；Time Registers 倒计时负责演化时长。使用方式：主机（传统处理器）按应用循环结构编排指令序列即可编程 DSU，无需理解底层 selector 细节——这是与先前 BRIM/DS-GL/DS-TPU/DS-TIDE 逐应用 ad-hoc 控制方式的根本区别，也是其上编译器/IR/软件栈的基础。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
