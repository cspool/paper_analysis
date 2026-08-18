## Pre-Binning（预分档）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
正式分 bin 前的软件阶段：把修复后的芯片按硬件指标预分组，并用轻量采样为每组生成 bin 级优化目标（MLCC_exp^target / Φ^target），供后续映射与调度使用。流程：芯片按硬件指标 F（冗余设计适应度，Sec.V-C）排序 → 分 B+1 个分位组（初步 bin 估计）→ 每组取 top 5–15% 芯片、用缩减种群/代数的轻量映射 + 调度各跑一遍 → 组内平均性能定义为该 bin 的目标。作用 = 把"性能收敛"变成可执行目标：同 bin 芯片追求同一目标（消除"强者更强、弱者更弱"的方差放大），又通过升级机制保留晋升空间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
目标生成伪代码：
```
chips.sort(by=F)                       # 按硬件指标 F 排序
groups = quantile_split(chips, B+1)    # B+1 个分位组
for g in groups:
    samples = top(g, 5%..15%)          # 组内轻量采样
    for chip in samples:
        run_reduced_SEGA_mapping(chip)   # 缩减种群/代数
        run_reduced_NSGAIII_sched(chip)
    MLCC_target[g]  = mean(samples.MLCC_exp)
    Phi_target[g]   = mean(samples.Phi)
```
之后全量优化阶段，每芯片以所属 bin 目标为中心优化、达标后升级到高一 bin（≤2 次）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
轻量采样避免全量仿真的成本（每 bin 只跑 5–15% 芯片的缩减版优化）；目标以 contention 指标（而非最终 latency）表达，与映射/调度代价函数同构（MLCC_exp ↔ LCC、Φ ↔ φ^k）；B+1 组而非 B 组预留边界裕量。pre-binning 是 ConBin 软件栈三阶段（pre-binning → 映射 → 调度）的第一环，为后两级提供统一优化方向。Web 证据：未发现同名独立方法（论文自述新机制），其思想与"分层/分批生成优化目标"的产线测试流程（speed binning 前测、分档测试）同构。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
