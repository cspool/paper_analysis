## 收益感知迁移调度（benefit-aware thread migration）与 Load State

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 的线程迁移决策算法（Algorithm 1）：不无条件迁到预测的"最优" chiplet，而是显式权衡预测收益与迁移/排队代价。对线程 t、预测 phase p、当前 chiplet c，到候选 chiplet c' 的迁移效用为 U(t,c')=S(p,c')-S(p,c)-λ·Q(c')-C_switch：S(p,c) 是 phase p 在 chiplet c 的离线预期稳态性能（如 IPC），Q(c') 是 c' 当前 runqueue 长度（由硬件 Load State 模块报告），λ 把排队线程折算成预期延迟惩罚，C_switch 是单次上下文切换代价；若 c'=c 则 U=0。调度器选 c*=argmax U(c')，仅当 U(c*)>θ 且线程在现 chiplet 驻留 ≥T_min 才迁移（T_min 防预测噪声导致的振荡）。这是把"负载感知"并入迁移决策的运行时调度策略，属于异构后端上的运行时任务调度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
调度循环（每 100µs epoch 边界触发，不在线程关键路径上）：

```
for each thread t at epoch boundary:
    p <- predicted phase for t            # 硬件 RF 预测器输出
    c <- current chiplet of t
    for each chiplet c' in C:
        S_c' <- expected perf of phase p on c'   # 离线 phase 特征表
        S_c  <- expected perf of phase p on c
        Q_c' <- runqueue length reported by Load State of c'
        U(t,c') <- S_c' - S_c - lambda*Q_c' - C_switch
    c* <- argmax_{c'} U(t,c')
    if U(t,c*) > theta and residency_time(t) > T_min:
        Migrate t from c to c*             # 标准 context switch 入队目标 chiplet runqueue
```

例子：某线程在 compute chiplet 跑完 GEMM phase，预测下一 phase 为 DeepCopy；若 fast-memory chiplet 空闲（Q≈0），U≈(fast-mem 相对 compute 的 DeepCopy 加速) - C_switch > θ → 迁移；若 fast-memory chiplet runqueue 很长（λ·Q 大），U 变负 → 留在 compute chiplet 或迁往次优但空闲的 chiplet（论文统计 18.5% 的迁移因最优核满载而落到次优核）。硬件支持：每 chiplet 有软件可写 task-count 寄存器（OS 在 runqueue 变化时更新），Load State 模块读取后导出给调度器，全程不占线程关键路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器在 OS 侧运行（PhaseWeave 不改 OS 调度器数据结构——chiplet 暴露为 CPU 簇，硬件经 MSR/MMIO 给 placement recommendation，OS 周期读并设 affinity），迁移=常规跨核任务迁移（页表根/TLB 走既有机制）；Load State/task-count 为硬件支持。使用与收益：对照实验显示无条件迁移（PhaseWeave-NoMigrationAlg）只达 1.26× 吞吐（专精+多核收益），加收益感知算法后 1.56×（算法额外贡献 1.24×）；53.2% 的 epoch 迁移≥1 线程；真实系统迁移开销平均 23.8µs、中位 9.5µs，远小于 phase 时长。通用启示：负载感知迁移是异构资源池（chiplet/GPU/加速器）调度的通用模式——把"排队惩罚"与"架构优势"统一进效用函数，避免异构系统被"最优目标过载"反噬。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
