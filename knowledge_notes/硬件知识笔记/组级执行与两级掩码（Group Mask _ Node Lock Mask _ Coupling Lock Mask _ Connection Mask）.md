## 组级执行与两级掩码（Group Mask / Node Lock Mask / Coupling Lock Mask / Connection Mask）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 控制器应对 DSU 三大挑战（节点/耦合规模巨大 → 控制复杂度；必须同步演化 → 正确性；多实例共存与细粒度配置 → 利用率）的分级控制方案。组级执行：借鉴 GPU 的 SIMD lockstep，把节点和耦合组织成组（node group / coupling group，以 NGID / CGID_col+row 寻址），一条指令同步作用于组内全部元素。两级掩码：① inter-group（组间）——Group Mask（GM）管理集体演化中哪些组参与；② intra-group（组内）——一维 Node Lock Mask（NLM）设置节点边界条件，二维 Coupling Lock Mask（CLM）与 Connection Mask（CM）按列/行分量精细控制耦合组的锁定与连通。掩码存在 Mask Memory（片上 SRAM），指令以 32-bit 地址间接引用（见 DS-ISA 条目）。C_EVOLVE 中同一 1D GM 对称应用于行与列组选择，激活对应交互子矩阵。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
掩码在控制器中的运转（Fig.10 + B2 并行多任务例子）：CFG_CONN[CM Addr, NGID] 把 Connection Mask 写入 Col/Row CM Registers——定义该耦合组内哪些格点连通（激活的耦合才参与交互），两块独立任务的拓扑可各自成区；N_LOCK[NLM Addr, NGID] 写 NLM Registers 锁住输入成员；N_EVOLVE[GM Addr, Time] 按 GM 同时向两个任务的输出组发演化信号——两个任务在同一 DSU 上并发演化（B2 模式）。微调场景（C2）：C_LOCK 用 CLM 只解锁待训练耦合子集，C_EVOLVE 只演化该子矩阵，其余耦合保持锁定。对称 GM 部署（C_EVOLVE）：选中节点组集合 → 同一 1D 掩码作行选与列选 → 激活这些节点组两两之间的耦合子矩阵，无需显式枚举耦合对，天然利用重复任务执行后散落的空闲节点/耦合"气泡"（抗碎片化），也可被 runtime 用于磨损均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每组配一组掩码寄存器（NLM/CLM/CM/Time Registers）+ 全局 Idle Registers 作默认锁定态；掩码数据在 SRAM 中由主机/编译器预生成；CGID 的 col/row 分量分开寻址列/行 Selection。使用方式：应用映射时先规划分组与掩码（输入组锁、输出组自由、连通子集由 CM 定义），再按 load-lock-evolve-store 编排指令；GM 也是多实例调度原语（不同组跑不同任务/不同层）。代价：掩码存储与广播带宽随组规模增长（评估中 stall 占比 <25%）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
