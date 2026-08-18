## TAGE-SC-L 分支预测器（含 MPKI 指标）

术语解释
TAGE-SC-L 是 André Seznec 在 CBP-5（2016）提交的分支预测器：TAGE（部分 tagged 的几何历史长度预测）+ 循环预测器（L）+ 统计校正器（SC），是本文 64KB 基线预测器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) TAGE 即 TAgged GEometric history length：默认的 tagless bimodal 预测器 + 多个"部分 tagged"预测组件，各组件用不同长度的全局分支历史索引，历史长度呈几何级数增长（如 4,8,16,...,640）；预测由历史最长的命中组件给出（tag 匹配），否则回退到默认预测器——长历史捕捉远距离相关，短历史在冷启动时提供覆盖；(2) L 是 loop predictor，专门处理长循环分支（行程计数规律），TAGE 的几何历史难以捕获超长循环；(3) SC 是统计校正器（statistical corrector），利用局部历史、全局历史与 IMLI 计数器对主预测做二次校正；(4) 本文以 64KB 配置作为 baseline 预测器（64KB TAGE-SC-L），并对比直接放大到 192KB 的版本。MPKI（Mispredictions Per Kilo-Instruction）= 误预测次数 / 千条指令，衡量预测器精度：$$\mathrm{MPKI} = \frac{\text{mispredictions}}{\text{retired instructions}} \times 1000$$。本文按 MPKI_base − MPKI_SBRB 排序 benchmark 展示 SBRB 的精度收益，且基线 MPKI 越高的 benchmark 从 SBRB/SYRANT 获益越大（因为它们在"更老误预测分支的阴影"里消除了额外的误预测）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TAGE-SC-L 位于 fetch 单元，为每个取到的分支给出方向/目标预测，是整条推测流水线的驱动者；其输出在本文中是 SBRB 的"被覆盖对象"。运转流程例子：取到 br3 → TAGE-SC-L 用几何历史索引命中某 tagged 组件出方向 → 同时 SBRB 按 key(br3) 查询 → 若 SBRB 命中且 BTB 置信 >3，用 SBRB outcome 替换 TAGE-SC-L 预测 → 最终预测送入流水线；若 br3 后续执行证明预测错，则该次误预测会计入 MPKI。关键实验事实：192KB TAGE-SC-L 只对少数 benchmark 明显有效（445.gobmk +6.43%、602.gcc +5.18%），且 192KB 预测器预测延迟高到不可接受——说明"堆大预测器"不是解决 CIDI 分支重复误预测的出路，SBRB 的 11KB 开销与之相比性价比极高。敏感性实验：把 64KB 换成 192KB TAGE-SC-L 后 SBRB 仍有 4.36% 平均收益（vs 4.43%），说明 SBRB 与预测器规模近乎正交。Bumper 论文的新用法：baseline 采用 64KB TAGE-SC（Seznec CBP2025 版，192KB 配置获 CBP2025 第二名 3.363 MPKI，含 untagged base + 28 个 2K-entry tagged 表、HCpred 备选预测与概率分配过滤等优化）+ iTAGE（即 ITTAGE 间接目标预测器，处理间接分支），配合 1K-entry L1-BTB / 16K-entry L2-BTB；Bumper 的关键观察是：BTB 容量 miss 时 TAGE 根本不产生该分支的预测，故基于 TAGE 置信度的预取过滤（UDP）节流不了由 BTB miss 引发的错误路径预取——这正是 L2C 污染的主来源。MPKI 在 Bumper 中的两种用法：(1) BPU MPKI = 前端重定向次数 / 千条提交指令，平均 8.0，并分解为 BTB miss 引起的部分（BTB MPKI，主成分）与方向/目标误预测部分（Direction/Target MPKI）；(2) L2C MPKI = L2C miss 数 / 千条指令，按 IFU（指令，平均 8.4）与 LSU（数据，平均 5.5）请求来源分解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TAGE 组件的实现：每个 tagged 组件是一个表，条目含 tag + 预测计数器，用 PC 与相应长度全局历史的 hash 索引；命中需 tag 匹配，未命中该组件不提供预测；备用机制处理新分配条目。SC/L 按 CBP-5 提交方案实现。MPKI 的使用方式：用于 benchmark 选择（本文选 ref input 中加权平均 MPKI 最高的 SimPoint）与结果分析（MPKI 曲线对比 baseline vs SBRB）。本文的 TAGE-SC-L 参数：64KB 预算、配合 8K 项 4-way BTB、64 项 RAS。

RUNLTS 补充视角（ISCA'26）：RUNLTS 在 192 KiB 的 TAGE-SC-L 基础上重设计了 TAGE 部分——① TAGE entry 结构：tag ~10-bit + 3-bit 饱和预测计数器（范围 -4..3，MSB 为方向、其余位为置信度，taken 增/not-taken 减、饱和）+ 1-2-bit u 计数器（当该 entry 覆盖短历史 entry 且预测正确时递增，与短历史同向且正确时清零）；只有 tag 匹配中历史最长的 entry 被更新；新 entry 仅在短历史预测失败时分配到更长历史表。② 新的历史长度集合（针对几何级数相邻短历史重叠、最长表闲置、各表负载不均衡的问题）：短历史用二阶等差（相邻差递增，如 6 之后至少 12）、中历史保持几何（切换点取几何增长率超过二阶等差处）、长历史用"比率成等差"（假设最大有效历史长度的对数近似均匀分布，使每张长历史表服务的期望分支数均衡）；该规则由模拟退火最优集的结构规律提炼，可迁移到 GEHL（192 KiB GEHL 3.338→3.316 MPKI）与 BATAGE-SC。③ thrashing 检测与分配节流：保留原始 TAGE counter 格式，把稀有状态组合 ctr∈{0,-1}+u=1（约 0.02% 条目，强制清零无精度损失）复用作 newly-allocated 标记，跟踪 newly_useful（标记条目首次正确预测并清标记）与 newly_decayed（标记条目被逐出前从未预测），按 newly_decayed ≤ 2×newly_useful→每次 misprediction 分配 4 个 entry、>4×→2 个、中间→3 个（192 KiB 经验映射；64 KiB 下为 3/1/2，counter 宽度取 log2(表条目数)）。④ SC 精修：IMLI 组件 gain 2→3、WT 更新步长×3、UT 条目 8→256；新增 call-stack-based history GEHL 组件（sC 6.053 KiB，即使无函数调用区域也因"同时训练所有历史长度"而互补 TAGE 的"只更新最长匹配"学习行为）。RUNLTS 移除 loop predictor（作者称大容量预测器中 loop predictor 收益有限），新增 RBias 组件（见 RBias 条目）。最终 RUNLTS-Log/-Seq 在 192 KiB 预算下（总量 191.760/191.921 KiB）相比 TAGE-SC-L 降低 MPKI 5.0%（CBP2025 673 条 trace）/5.25%（gem5 SPEC CPU 2017）。MPKI 在本论文中的用法：S-curve 按每 trace 的 MPKI reduction 排序（Figure 10），平均 reduction 0.137、中位数 0.048；机制分解按 RBias/IMLI/call-stack/历史长度+节流逐项统计 MPKI reduction（Figure 11/12，asinh 尺度）。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability（基线使用 64KB TAGE 预测器 + 2K-set 4-way BTB、15-cycle 误预测惩罚，见其 Table I）
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
