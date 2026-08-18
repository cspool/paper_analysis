## 机会回填（Opportunistic Backfilling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
机会回填是 Triage 回收空闲解码器吞吐的机制：紧急计划的并行度（峰值解码器需求 M_peak）通常小于总解码器数 M，剩余解码器在紧急执行期间空闲；回填用这部分预算派发非关键、因果无关的任务。公式（论文）：M_usable(t)=max(0, min(M−M_peak−B_bf(t), F(t)−E(t)))，其中 B_bf(t)=正在运行的回填任务数、F(t)=物理空闲解码器数、E(t)=同 pass 已派发的紧急任务数。回填任务用稳态启发式挑选，与紧急计划因果无关，因此不会干扰关键路径——在紧急模式保持延迟最优的同时维持吞吐。类比 HPC 批调度器的 backfill 思想（填充分区调度的空隙），但在 FTQC 解码器池上以"因果无关"保证不侵犯紧急计划。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
回填在紧急模式的每次派发 pass 中执行：
```
# 紧急模式一个 pass
计划 P（预测性着色）→ M_peak = max 同时派发数
B_bf ← 进行中的回填任务
for t in 计划时间线:
    M_usable(t) = max(0, min(M − M_peak − B_bf(t), F(t) − E(t)))
    D_bf = SelectNonCritical(M_usable(t))     # 与紧急计划因果无关的 PENDING slice
    dispatch(D_bf, 稳态启发式)                  # 不抢占紧急任务
```
效果（论文 Fig.11）：无回填时 Triage 解码器利用率在紧急期间出现空闲，回填后利用率显著提升。这是 Triage 与"紧急模式只管关键路径"的朴素实现的区别——吞吐最大化而不牺牲延迟最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：由紧急计划先算 M_peak，再按公式逐 pass 算可用回填预算；回填任务选取复用稳态加权启发式。使用场景：紧急模式期间利用空闲解码器推进非关键 slice，降低整体 backlog 与 idle 层数。论文未开源实现。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
