## Ising Machine（Ising 机 / CMOS 环形振荡器 Ising 芯片 / Ising spin / machine capacity）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ising Machine 是基于物理原理的硬件求解器：把组合优化问题（COP，如 SAT）的全局最小值映射到物理系统的最低能量态。系统含相互作用的自旋（Ising spin，s_i∈{−1,+1}），能量为 $H(s) = -\sum_{i<j} J_{ij} s_i s_j - \sum_i h_i s_i$（J_ij 耦合、h_i 局域场）；问题变量映射为自旋、约束映射为自旋间交互，系统被扰动后自然弛豫到低能态，即近似解。技术路线差异决定两个关键能力：machine capacity（最大自旋数）与 connectivity（自旋间最大连接度）。技术实现包括超导量子比特（量子退火）、Rydberg 原子、经典 CMOS 耦合振荡器等。SATIC 的测试床是 45 个全互连（all-to-all）自旋的 CMOS Ising 芯片：每个自旋 = 一个室温 CMOS 环形振荡器（COBI 系列，文献[20]），系数范围 [−14,+14]；45-spin all-to-all 等价于 1000+ spin 的受限邻居连接芯片。芯片装在有 FPGA 的板卡上经 PCIe 与主机通信（8 卡并发 repeats）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（SATIC 的一次子问题求解）：
```
# 主机（Intel Xeon Gold 6240R，Python 3.8）编译：CNF → VIG → 子问题 → QUBO（≤45 变量）
# Machine Embedding：Adaptive Spin Merging（动态合并未用 spin 扩系数范围）、Dynamic Upscaling（缩放系数）
# 下发：QUBO 系数（J,h）→ FPGA 板卡 → 芯片 45 个环形振荡器自旋
# 演化：环形振荡器自旋间通过耦合网络交互，系统退火/弛豫到低能态（≤200μs/次，4.8μJ/次）
# 回读：自旋取值 → 主机回收 S_sub 更新全局解向量 → CheckSolution
# 并行：PCIe 多路复用 8 张卡 → 并发 repeats（论文全程 >20 亿次硬件访问）
```
硬件约束影响编译：容量 45 spin → 子问题 ≤45 变量；系数范围 [−14,+14] → 超出需缩放/合并；有限精度 → ILP 大系数被截断（Flat ILP、Adaptive Spin Merging 应对）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CMOS 环形振荡器阵列（振荡相位编码自旋、注入锁定实现耦合）、FPGA 控制 + PCIe 主机接口；同族芯片见文献[20]（COBI）、DROID（文献[23] 离散时间仿真工具）。使用：结合 SATIC 类编译器解决 SAT/Max-Cut 等 COP；相对软件求解器（D-Wave Tabu：20ms/3.3J 每迭代）快约 2 个数量级、能耗低约 5 个数量级（200μs/4.8μJ）；对比 Cilasun et al. 的 49-spin 芯片（系数范围相同、spin 多 8%）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
