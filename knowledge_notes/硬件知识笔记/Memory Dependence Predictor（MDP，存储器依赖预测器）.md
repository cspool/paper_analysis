## Memory Dependence Predictor（MDP，存储器依赖预测器）

术语解释
乱序 CPU 中预测 store 与 load 之间数据依赖关系的微架构预测器：当 store 地址尚未解析时，MDP 预测后续 load 与该 store 是否地址重叠（数据依赖）。预测为独立（1→0）时 load 可乱序绕过执行，预测为依赖（→1）时 load 被阻塞直到 store 提交或地址解析。因 MDP 设计未公开文档化，是安全研究（Spectre-V4、MDP 侧信道）与微架构逆向的核心对象。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MDP 自 1990 年代提出（[9]）后衍生大量设计（[30][32][41][46][49][58][62] 等），已在 Intel、AMD、Arm、Apple 商用 CPU 上广泛部署。其工作流程（SSBench 论文 Fig.4）：① 一条触发预测的指令（load，有时连同 store 与上下文辅助信息）通过上下文特征（IP、数据地址、历史信息）选择预测表表项；② 查表项当前状态并产生预测（1=依赖/0=独立）；③ 实际执行后按结果（依赖/独立是否正确）更新表项状态。按两个维度可分类（SSBench 的 workflow-based taxonomy）：维度1 表项选择方式——L（仅用 load IP）、SL（store+load 双 IP）、HSL（额外含分支历史）、O（其他，如 Store Barrier Cache 仅 store IP、Branch MDP 仅分支指令）；维度2 状态机——S（有状态，动态置信度更新）、nS（无状态，命中即预测依赖直到驱逐/刷新）。SSBench 据此对 30 颗 CPU 实测：Intel 为 L-S（direct-mapped，256/512 项，index=load IP 低 8/9 位）；AMD Zen3 双 MDP（z3-mdp1 为 SL-S 与 PSFP 重叠、z3-mdp2 为 L-S 32 项 2 路 LRU），Zen4/Zen5 仅 L-S（32/64 项、FIFC/NLRU、物理 IP 索引 + 12-bit stride hash）；Arm A72/A73 为 L-nS（16 项 16 路 PLRU/FIFO）、A76 首次发现 L-S 状态机；Apple 为 SL-S（3-bit 计数器、hash 后虚拟 IP 作 tag，大小核表项数不同）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
硬件侧（以 Intel i9-13900K 为例）：store-load 对中 store 地址经乘法链延迟（地址生成被算术延迟放大）→ load（及 store IP）经 hash/index 位选 MDP 表项（低 8 位 load IP → 256 项 direct-mapped）→ 查表得预测：预测独立则 load 乱序执行（若地址实际重叠，store 地址解析后需 squash 并重执行，产生 rollback）；预测依赖则 load 阻塞（pipeline stall）。MDP 更新条件各厂商不同（SSBench 发现）：Intel 仅当 load 地址先于 store 解析时才更新（可被状态传播用于构造 μWM）；AMD 无前置 store 的 load 也能更新（扩大攻击面）；Apple 未提交的 store-load 对也能更新（支撑瞬态隐蔽信道）。安全影响：MDP 违反进程/特权隔离（Intel 的 SGX 边界、Apple 的用户/内核边界），可被用于 MDPeek、跨进程 Spectre-V4、MDP-Gates/MDP-CF/MDP-CC 等攻击。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用实现均未文档化，需逆向；SSBench（https://github.com/CPU-Security/SSBench，Apache-2.0）是首个自动化表征框架：Python 测试套件 + C/汇编微基准原语，`python3 main.py -c <core-id>` 自动探测 MDP 存在性/设计类型/状态机/hash 函数/表组织，输出 data/characterization.json。评估性能/安全影响：禁用 MDP（Intel/AMD 的 SSBD、Armv8.5+ 的 SSBS）会强制 load 等所有前序 store，SPEC2017 intrate 单核平均降速 10.7%（i9-13900K）、13.9%（Zen3）、4.3%（A76），且部分平台（如 Raspberry Pi 4B 的 A72）无 SSBS。软件缓解：constant-time 编程、敏感操作周围的内存屏障、OS 上下文切换时刷新 MDP 状态；硬件缓解方向：per-process 分区（PIDs）、随机化索引、跨上下文专用 MDP 历史缓冲。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
