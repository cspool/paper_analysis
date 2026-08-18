## 分层静态-动态协同调度（Hierarchical Scheduling / Static-Dynamic Co-design）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
分层静态-动态协同调度是把"离线全局规划"与"在线局部自适应"组合成闭环的调度范式，核心前提是：稀疏工作负载的 intra-tile（tile 内数据流）与 inter-tile（tile 形状/占用/遍历顺序）参数互相耦合，任何一层单独优化都会次优。Harmonia 将其实现为三层：(a) 静态分析层（Static Analytical Layer）——只用粗粒度描述符（矩阵形状 M,K,N、全局密度 ρA/ρB、数据类型）与硬件参数（PE 阵列大小 P、SRAM 容量 S_SRAM）离线枚举候选块形状 (T_M,T_K,T_N)，以操作强度 OI=OPs/Bytes 为目标、受 SRAM 可行性约束 s_val(E[nnz_A]+E[nnz_B])+s_psum·T_M·T_N ≤ β·S_SRAM（β=0.8）过滤，输出基线块形状、块间遍历顺序与 SRAM 分区 S_A/S_B/S_C，提供一个安全（1.0×）性能下限并缩小在线搜索空间；(b) 动态 Profiling 层（Dynamic Profiling Layer）——块进入 SRAM 后按 tile 精确 nnz 采样（tile 密度 ρtile=nnz/(T_M·T_K)、行/列方差、非零聚类），细化 tile 形状并选择 InP/Row/OutP 数据流；(c) 动态 Tuning 层（Dynamic Tuning Layer）——消费硬件反馈计数器（SRAM pressure、psum spill、MRN merge-depth、PE stall），用成本模型 Gain>α·Cost 决定是否在 tile 边界切换数据流/微重切块，滞回（T=2~4 周期）防振荡、最坏情况回退 1 次重构延迟（50 cycles）。三层共同把"同构硬件"变成"逻辑异构"执行引擎。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
调度流水（伪代码，Harmonia 处理一次 SpMSpM 的全过程）：
```
# 静态层（离线，每矩阵一次）
T_set = gen_candidates(M,K,N)            # 按可整除性与 PE 对齐剪枝
for (TM,TK,TN) in T_set:
    if feasible(TM,TK,TN, rhoA,rhoB, S_SRAM):   # 约束(2)
        OI[T] = est_OPs(T) / est_Bytes(T)
(TM*,TK*,TN*) = argmax OI; order = pick_traversal(M,K,N)
# 动态 Profiling 层（每块一次）
for block in blocks:
    rho = block.nnz / (TM*TK);  var = row_variance(A_block)
    if rho < thr_low:  expand(block)           # 低密度扩张提复用
    if rho > thr_high: shrink(block)           # 高密度收缩防 psum spill
    df = select_dataflow(rho, var, cluster)    # InP/Row/OutP
# 动态 Tuning 层（每 tile 执行中，反馈驱动）
while executing(tile):
    if anomaly(merge_depth, spill, stall):
        if gain(candidate_df) > alpha*cost(reconfig): switch_dataflow(tile)
        elif gain(micro_tile) > alpha*cost(micro_tile): micro_retile(tile)
```
关键设计：静态层决策独立于逐 tile nnz（保证 buffer 可行性）；切换严格在 tile 边界发生（全局矩阵是一串独立 tile），pipeline flush + DN/MRN 重编程 + buffer reset 共 20–50 cycles，总 stall <1%。结果：16 个 SpMSpM workload 平均 1.75×（orani678 3.46×，接近 2.03× oracle 上界），端到端 DNN 1.87×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对比既有单层调度：Misam 用决策树在 tile 内选数据流但忽略 tile 间依赖与共享 SRAM 压力；HYTE 用静态分析+运行时细化调整全局 tile 边界但假定固定 tile 内数据流；Vesper 用统一解析模型离线搜全局最优但假设均匀稀疏、无运行时反馈。Harmonia 的实现要点：(1) 静态层用"保守可行性"而非逐 tile 精确建模，避免离线采样过拟合；(2) 在线层每块只做一次轻量采样（A 矩阵驻留部分），开销可忽略；(3) 反馈路径与执行数据通路完全解耦（每 PE 行 128 个计数器占 <0.5% PE 阵列面积），保证 profiling 不扰动时序；(4) 成本模型用 α 调节激进程度（不规则负载 α 大、规则负载 α 小）。论文未提供开源实现。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
