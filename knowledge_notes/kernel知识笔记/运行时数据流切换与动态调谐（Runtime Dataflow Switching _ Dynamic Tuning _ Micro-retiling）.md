## 运行时数据流切换与动态调谐（Runtime Dataflow Switching / Dynamic Tuning / Micro-retiling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
运行时数据流切换是 Harmonia 动态 Tuning 层的核心机制：在执行一个 tile 的过程中，根据硬件反馈计数器观测到的异常（psum 溢出、merge 深度超预期、PE stall、SRAM pressure），在 tile 边界把 intra-tile 数据流在 InP/Row/OutP 之间切换，或在块内做微重切块（micro-retiling），使同构 PE 阵列呈现"逻辑异构"（每个 tile 用最适合当前稀疏模式的数据流）。切换规则（论文给出的显式映射）：InP→Row 当 A 行极不平衡、InP→OutP 当稀疏近均匀；Row→OutP 当 merge 深度低但 B 行重载压 SRAM、Row→InP 当 tile 稠密；OutP→Row 当 spill 源于局部性、OutP→InP 当 spill 源于高密度。每次切换的硬件开销 = pipeline flush + DN 路由表/MRN 模式（merge-before-store vs column-accumulate）重编程 + AGU/buffer 控制器策略重置 = 20–50 cycles。只有 Gain=T_before−T_after > α·Cost 才触发，滞回机制（异常计数器连续 T=2~4 周期超阈值）吸收瞬时抖动，切换失败时最多损失 1 次重构延迟（50 cycles）后回退到静态基线。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 动态 Tuning 反馈环（每 tile）
loop over cycles:
    spill  = read_counter(psum_spill)
    depth  = read_counter(mrn_merge_depth)
    stall  = read_counter(pe_stall)
    if (spill > TH_spill or depth > TH_depth) and anomaly_hold >= T:
        cand = candidate_switches[dataflow]        # 表驱动切换候选
        if best_gain(cand) > alpha * reconfig_cost:   # 式(3)
            flush_pipeline(); reprogram_DN_MRN(cand); reset_buffers()
            # 20-50 cycles
        elif micro_tile_gain > alpha * micro_cost:
            shrink_K(tile)   # 降 merge 深度 / 收缩 M,N 限 DN/buffer 负载
```
具体例子（email.mtx）：静态层选 OutP，执行中发现部分 tile 密度高导致 psum 溢出本地 buffer、merge 深度超预期 → 反馈计数器连续触发 → 评估切换成本（~40 cycles）后在该 tile 边界切到 Row（选择性路由 B 行片段、merge 深度更浅更可预测），后续 tile 恢复 OutP；整体 16 个 workload 平均 1.75× 加速、总 stall <1%。bcsstk10 场景：静态模型预测 OutP 最优但实际 OutP 延迟为 Row 的 1.6×，动态 Tuning 通过反馈纠正了这一离线误判。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现层面依赖三项轻量硬件支撑（合计 3.3% 面积、<0.5% PE 阵列面积）：Feedback Counters（每 PE 行一组，经轻量 metadata crossbar 汇聚到 Tiling Controller，反馈路径与主数据通路解耦）、Reconfiguration Engine（重编程 DN 路由表、MRN 模式、AGU 与 buffer 策略）、Tiling Controller（执行切换决策与成本模型）。使用上：作为调度器最末级，只纠正 tile 级偏差、不改变全局计划；适用于稀疏模式在运行时剧烈变化的负载（LLM attention/MLP 投影的 token 级稀疏抖动、CNN 剪枝后的 channel 级波动、高度不规则矩阵如 orani678 的聚类长尾）。论文未提供开源实现。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
