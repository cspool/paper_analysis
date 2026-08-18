## MDP 定时侧信道（S/B/R 时序类型与自动识别）

术语解释
利用 MDP 对 store-load 依赖/独立预测产生的可测执行时延差异（bypass S / block B / rollback R）来探测 MDP 内部状态的一类微架构定时侧信道。SSBench 将其形式化为"定时侧信道驱动的自动识别"：固定 store/load IP 执行受控 store-load 对，聚类时序样本即可判定 MDP 存在性、设计类型（L/SL/HSL × S/nS）与激活条件。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
与分支预测器不同，MDP 的预测代价不对称：独立误预测导致流水线回滚（代价大），依赖误预测只是阻塞（代价小），因此状态机转移阈值与更新值不对称（如 AMD 状态机 ths>0、需 3 次误预测激活；Cortex-A76 状态机初始化 14、dependence 不更新、independence 递减）。SSBench 的识别算法：① 存在性——固定 IP 执行 100 个独立对（N_P）+100 个依赖对（D_P）取 200 个时序样本，DBSCAN（minPts=100, ε=2）聚类出 S/B/R，判定条件 max(T0)≤min(T2)（S/B 可分）且 min(T1)≥max(T2\T1)（B/R 可分），噪声（频率波动/上下文切换）呈稀疏长尾被过滤；② 设计类型——用 Pair_0^0/Pair_1^0/Pair_0^1 训练后测 T(N_P) 判定表项选择维度（L vs SL vs HSL），用 D_P 回滚后重复 N_P 是否 B→S 转移判定状态机维度（S vs nS）。消除其他预测器干扰：随机 base address（防地址预测器）、随机值（防值预测器）、预访存数据（防预取器）、分析 T2 中 B/S 混合以校正 PSFP/内存重命名。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
以 Apple M4 性能核为例（SSBench 实测输出）：store-load 对经 JIT 生成、固定 IP → 定时原语（Apple 用 S3_2_c15_c0_0 计数器，需内核修改）测量 → exist 测试得 S/B/R 区间（type_time_dict: S[[189,189]], B[[274,275]], R[[299,299]], b1=231,b2=287,b3=448）→ 状态机测试输出 store_sm=[0,-1,7,1,7,1,0,0]（3-bit 计数器：dependence+1/independence−1/threshold）→ hash 测试得 hash_func（虚拟 IP 位分组）→ org 测试得 90 项全相联 LRU。侧信道在此既是"探测手段"（恢复 MDP 内部状态）也是"泄漏源"（跨进程/特权可被攻击者观测，Table III：所有被测 CPU 进程隔离均为 X，即 MDP 表跨进程共享）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SSBench 开源仓库（https://github.com/CPU-Security/SSBench）中 exist.py/sm.py 等 Python 套件 + src/ 下 C/汇编微基准；定时原语按平台：Intel rdtscp、AMD rdpru、Arm PMCCNTR（tools/arm-pmu-enable 内核模块）、Apple S3_2_c15_c0_0（内核 patch，PacmanPatcher）、RISC-V rdcycle。使用：`python3 main.py -c <core-id>`（或 -a <arch>、-u 免 root）→ 各阶段输出到 data/characterization.json；每参数多次重复取众数 + 方差早停降噪。用途：自动 MDP 逆向（替代人工时序分析），支撑存在性/设计类型/状态机/hash/组织的全自动表征与安全测试（跨域共享、chaining、speculative 更新）。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
