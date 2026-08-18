## 工作量级并行（WLP，Workload-Level Parallelism）与 HILP 三阶段 workload 建模

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
工作量级并行（WLP）指异构 SoC 上同时运行的多个独立工作负载（应用）之间的并行性，区别于应用内部线程级/指令级并行。HILP（Rogers, Eeckhout, Jahre, HPCA 2025）[55] 是首个把 WLP 完整计入 SoC 早期设计空间探索的方法：把"在异构 SoC 上调度多阶段应用的 workload"形式化为 Job-Shop Scheduling Problem（JSSP），用整数线性规划（ILP）求解；并把每个工作负载分解为三个阶段——setup（设置，串行于 CPU）、compute（计算，可并行、可跑 CPU 或 GPU/DSA）、teardown（收尾，串行于 CPU）。同一 workload 的 setup→compute→teardown 有依赖（compute 需等 setup、teardown 需等 compute），不同 workload 间可重叠，形成调度优化问题。RHODES（ISCA'26）复用 HILP 公开提供的 Rodinia benchmark profiling 数据（每 workload 三阶段执行时间与 TDP），但目的不同：HILP 建模性能/WLP 本身，RHODES 把三阶段分解用作鲁棒碳优化中运营碳（C_op 按 setup/compute/teardown 分相）与执行时间约束的输入。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 的 WLP 案例（§V-B）中，系统调度建模为：二进制变量 x 表示 compute 阶段跑 GPU（x=1）还是 CPU（x=0）；辅助变量 w_i=c_i·x 线性化 CPU/GPU 活跃组合；约束 (c_i−w_i) 激活 CPU compute 相（x=0 时）、Σg_j=x 保证 x=1 时恰选一个 GPU 配置。执行时间约束（Eq.6）：(T_c,s+T_c,td)^T·c + T_c,k^T·(c−w) + T_g,k^T·g ≤ T_max。WLP 场景下多个 workload 经 HILP 的 JSSP/ILP 调度确定各 workload 的阶段在 CPU/GPU/DSA 上的重叠执行计划，RHODES 把整个计划的总执行时间与分相运营碳纳入碳约束（用 U_budget 不确定性集防止 EPW/GPW/MPW 跨组件同时取最坏值）。效果：含 WLP 的鲁棒设计在 Monte Carlo 模拟 tC 的 0.951–1.147× 内，而 nominal 优化低估 1.17–1.55×；RHODES 与 HILP/Neoscope 的早期 SoC DSE 生态互补（HILP 出性能 profiling、RHODES 出碳鲁棒优化、Neoscope 出 workload churn 弹性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HILP 以 ILP 解 JSSP 调度 workload（开源 PDF：https://users.elis.ugent.be/~leeckhou/papers/hpca2025-hilp.pdf，联网未确认独立公开 GitHub 仓库），输出 SoC 配置下 workload 的总执行时间与阶段分解；RHODES 在 Julia 中把该三阶段模型编入鲁棒 MILP（Gurobi 求解，WLP 案例在 Intel Xeon E5-2680 28 线程运行）。使用：设计者输入 workload 集（Rodinia 负载，如 BFS/pathfinder/myocyte/heartwall/lavaMD/nn）与 SoC 配置空间（CPU 核、GPU SM×频率、DSA、HBM），得到"含 WLP 调度 + 碳不确定性鲁棒"的最优 SoC 配置。局限性：HILP 的性能模型基于早期 profiling（非周期级模拟），碳结果依赖 profiling 数据与实际运行的一致性（garbage in, garbage out，论文明确声明）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
