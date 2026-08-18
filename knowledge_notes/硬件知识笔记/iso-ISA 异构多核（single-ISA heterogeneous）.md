## iso-ISA 异构多核（single-ISA heterogeneous）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
iso-ISA（single-ISA）异构指系统内不同核心/芯片共享同一指令集架构，但微架构配置各异（频率、发射宽度、缓存、向量单元等），从而在不牺牲可移植性的前提下实现性能/功耗/资源专精。PhaseWeave 的四类 chiplet（compute/fast-memory/near-network/low-power）全部实现 x86 ISA：线程可在 chiplet 间迁移而无需重编译或二进制翻译；每个 chiplet 的微架构按 phase 类别调优。区别于 big.LITTLE 式异构（两类核沿 compute-energy 一维轴，OS 启发式/程序员标注、task 级粗粒度），PhaseWeave 是四类 chiplet 的多维资源异构（compute/memory/network/power），且由硬件预测器驱动细粒度（100µs epoch）迁移。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件组织中，iso-ISA 是"异构硬件 + 同构软件接口"的桥：OS 把 chiplet 看成 CPU 簇（类似 NUMA 节点、不连续 CPU ID 区间），调度器照常工作；硬件预测器经 MSR/MMIO 给出 placement recommendation，OS 用原生 affinity 机制（sched_setaffinity 类）迁移线程。运转例子：AdSim 请求的 GEMM phase 在 compute chiplet 的 3.0GHz/6-wide/512-ROB 核执行（有 SIMD/大 ROB/TAGE 预测器、无 NIC）；同一线程下一 epoch 预测为 DeepCopy 后经标准 context switch 迁到 fast-memory chiplet 的 2.5GHz/4-wide 核（L2 2MB、5MB LLC slice、25.6GB/s、无向量/NIC）；网络密集 phase 迁到 die 内集成 NIC 的 near-network chiplet；空闲 phase 迁到 2.0GHz/2-wide 无向量单元的 low-power chiplet 省功耗。迁移平均 23.8µs（中位 9.5µs）<< phase 时长（数百 µs），开销可忽略。未来扩展：可混入不同 ISA/加速器，此时不做线程迁移而向 runtime/编译器发 offload 提示（论文 Section III-F）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：chiplet 共享 ISA 核设计（不同微架构的同一 ISA 实现），OS 暴露为异构 CPU 簇；迁移=常规跨核任务迁移（页表根更新、TLB 一致性走既有机制），不需要新调度器数据结构。使用/价值：(1) 免重编译的透明迁移是细粒度 phase 调度的前提；(2) "每核专精一类"使 iso-area 下核数从 28 增至 38（per-core 面积降 27-39%、功耗降 34-56%），排队显著缓解；(3) 与 single-GPU-like 编程模型（多 die GPU 统一编程）同构的思路：硬件异构被软件抽象。对比：big.LITTLE 基线（bigL-OoO、bigL-Opt 等，核比 1:1/1:2/1:4）在 iso-area 下即便手工标注最优核类型也低于 PhaseWeave 1.3× 吞吐，证明多维异构 + 细粒度检测的价值。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
