## GPC / TPC / SM（GPU 层次结构：Graphics Processing Cluster / Texture Processing Cluster / Streaming Multiprocessor）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVIDIA GPU 的物理组织层次：整颗 GPU 由若干 Graphics Processing Cluster（GPC）组成，各 GPC 共享 L2 cache；每个 GPC 内含若干 Texture Processing Cluster（TPC）；每个 TPC 含 2 个 Streaming Multiprocessor（SM）；SM 是 GPU 上调度/执行 thread block 的基本计算单元（含 SIMT 核、Tensor Core、共享内存等）。Blackwell B200：9 GPC、74 TPC、148 SM。传统上整颗 GPU 的所有 SM 运行在同一时钟频率（单一 DVFS 域）。该层次结构定义了空间 DVFS 的频率域候选粒度：per-GPC、per-TPC、per-SM（论文硬件分析在 per-GPC 到 per-SM 之间扫描，per-SM=148 域为最细）。
- 与编程模型的关系：GPU kernel 是"计算网格（grid）→ 许多 thread block（block）→ SIMD 线程"的结构；thread block 被调度到 SM 上执行，一个 block 独占一个 SM 直到完成。kernel 的 block 数、每 block 线程数（occupancy）决定每个 SM 要串行执行多少波（waves），这是 PowerWeave latency predictor 用 wave 比例泛化 kernel 延迟的基础（l = waves × l_old/waves_old）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PowerWeave 中，GPC/TPC/SM 层次是"空间域划分"与"资源分配"的坐标：LithOS/PowerWeave 以 TPC 为分配粒度（多租户租户 1/2/3 分别分 18/19/37 个 TPC ≈ 1/4/1/4/1/2 卡）；空间 DVFS 每域覆盖一组 TPC/SM，每域独立频率。运转流程示例（disaggregated prefill，B200 74 TPC）：74 个 TPC 一分为二，37 TPC 给 prefill 实例、37 TPC 给 decode 实例（各构成一个频率域）→ prefill 域的 SM 跑 compute-bound GEMM kernel 时域频率高、decode 域的 SM 跑 memory-bound 内核时域频率低 → 每 kernel 的 waves 数 = launched blocks / (blocks-per-SM × 分配 SM 数) 用于延迟预测 → 能量按"未分配 TPC 份额"比例扣除 idle 功耗（idle ≈140W）。
- 频率切换硬件现实：Blackwell 频率切换延迟 ≈10–100µs（比 Hopper 的 ≈10–100ms 快约 1000 倍），使 100µs 级的按域/按阶段频率切换成为可能，这是空间 DVFS 在 Blackwell 上可行的关键硬件前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：软件以 TPC 粒度分配 GPU 资源——LithOS 做 TPC 级空间调度（基于 MPS）、PowerWeave 在 TPC 分配基础上叠加每域频率控制；NVIDIA MIG 是硬件级空间分区（与 TPC 分配正交），AMD MI300X/MI355X 的 SPX/DPX/CPX 提供跨 XCD 的 MIG 等价隔离、ROCm CU masking 提供 MPS 式流到计算单元的细粒度分配。论文实验以 TPC 数描述各租户/agent 的资源（agentic pipeline：Agent1 10 TPC、Agent2 27 TPC、Agent3 37 TPC）。GPU 空闲功耗 ≈140W（B200）用于能量测量校正。
- RHODES 的 SM 建模视角（ISCA'26，设计早期碳感知 DSE）：把 GPU 抽象为"SM 数量 × 时钟频率"的配置变量 g_m^f（m∈1..128、f∈210–765 MHz 11 档），每个 SM 面积按 NVIDIA A100 估算（826 mm² / 108 SM = 7.65 mm²@7nm EUV）；功耗按 TDP 估计（active 与 idle 向量 P_g/P_g,idle），执行时间按工作量 compute 阶段在 GPU 上运行建模（T_g,k^T·g），并参与面积（A_g^T·g）、功耗（P_c,idle^T·c+P_g^T·g≤P_max）、tC（(FPW+GPW+MPW)⊙A_g）约束。SM 配置选择是鲁棒 MILP 的二进制决策变量，目标是选出对碳不确定性集内所有实现都可行的 SM 数/频率组合。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
