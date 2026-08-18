## Triage 双模解码调度器（稳态模式 + 紧急模式 + Triage Trigger）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Triage 是 FTQC 经典控制栈的解码器调度中间件（ISCA 2026，arXiv:2605.04459，HKUST-GZ Wang Research Group），双模架构：①稳态模式（steady mode）——轻量启发式保证平均吞吐：统一优先级函数 P(V)=w_u·Urgency(V)+w_c·Cost-Efficiency(V)，Urgency=1/Deadline（接近关键截止时间者优先）、Cost-Efficiency=1/(Degree+1)（计算便宜者优先），w_u=w_c=0.5；探索策略含 FIFO（最老优先清积压）、EDF（最小 deadline）、MDF（最小 degree）。②紧急模式（emergency mode）——priority-aware look-ahead：Triage Trigger 在任一 PENDING slice 的 deadline 达到 τ_emergency 时触发，用"预测性因果锥着色"（Algorithm 1）预先计算关键操作因果锥的解码计划并按时间表派发。配套机制：ScopeCap（因果锥>100 回退稳态）、增量重规划（新因果锥超现有 scope 30% 且距上次重规划 ≥2 层才触发，防 thrashing）、机会回填（回收紧急计划之外的解码器吞吐）。逻辑链：FTQC 解码需求在 T 门同步点前有尖峰，纯启发式缺乏前瞻无法保证低延迟 Pauli frame 更新（idle 层仍多），纯前瞻式暴力并行在资源受限时不可行——双模在平均负载与临界时刻之间自适应。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Triage 在经典控制流水线中的完整运转流程：
```
# 离线：编译器 → 静态分析器
高层程序 → Litinski 风格编译器 → LLI 流（multi-patch measurement/rotation/idle）
→ 静态分析器单趟 → Timeline（每 unit: layer/坐标/操作/6-bit 邻居 mask/deadline/因果锥引用）

# 在线：每个 syndrome 到达与解码完成事件
1) 新 PENDING slice 入 Timeline；若关键操作因果锥未解码完 → 插 idle layer（t≥ℓ 层右移）
2) 稳态：按 P(V) 选 ≤M_available 个无冲突 slice → 分派
3) Trigger 检查：有 PENDING deadline ≤ τ_emergency → 切紧急模式
   紧急：BFS 算因果锥（≤ScopeCap）→ Algorithm 1 生成 (t_sim, s) 计划 → 缓存 → 按表派发
4) 空闲解码器 → 机会回填非关键任务（M_usable(t)=max(0,min(M−M_peak−B_bf(t),F(t)−E(t))))
5) 每事件重复；idle 层超 10× 原层数 → backlog 失败终止
# 输出：idle 层数 → 总层数 → T_total=N_layers×d×T_meas；Stim 蒙特卡洛聚合 LER
```
调度开销：Triage 稳态每层中位亚毫秒；紧急计划 O(n log n)（大 scope 多毫秒尾延迟，对离子阱/中性原子慢周期平台可接受，超导 ~20μs 层周期需编译/硬件辅助实现）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python 3.9 离散事件模拟器（Intel i9-14900K + 188GB RAM）；解码延迟用 pymatching 幂律模型 t_dec=A·volume^α（α=1.17），随机抖动用校准 log-normal。baseline：serial sliding window、time-parallel window、SWIPER（10% misprediction、10% speculation time）。评估：15 个 QASMBench benchmark × 2 资源场景（并行富余 count=2×#LQs/speed=0.9；延迟富余 count=#LQs/speed=1.8）；指标=插入 idle 层数 + LER。结果：平均 LER 比时间并行 baseline 降 52.6%；慢解码器区（τ_dec>τ_gen）仍有效；对超参数（w_u、τ_emergency∈[2,8]）鲁棒。论文代码未开源（arXiv:2605.04459）。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
