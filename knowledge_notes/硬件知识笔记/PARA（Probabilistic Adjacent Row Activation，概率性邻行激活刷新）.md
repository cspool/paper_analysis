## PARA（Probabilistic Adjacent Row Activation，概率性邻行激活刷新）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Kim et al. 在 RowHammer 原始论文（ISCA 2014）中提出的轻量 RowHammer 缓解：内存控制器每次关闭行后，以极低概率 p（典型 0.005）激活（即刷新）该行的一个相邻行。理论保证：p=0.005 时一年内攻击成功概率约 9.4×10⁻¹⁴，保护强度可调 p；实测平均减速仅 0.20%（最大 0.75%，29 个 benchmark）。优点：无状态（不需要每行计数器）、控制器易实现、代价低；缺点：需要物理行布局知识（邻行映射，通常需逆向）且为概率性保证。Web 来源：RowHammer: A Retrospective（arXiv:1904.09724）、Flipping Bits in Memory Without Accessing Them（ISCA 2014）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 内存控制器命令路径：收到 ACT 请求 → 行地址查物理映射得邻行 → 以与配置 N_RH 相关的概率抛硬币 → 命中则在命令流插入邻行的 ACT/PRE（或 REF）→ 继续正常服务原请求。DejaVu（ISCA 2026）在 Ramulator 2.0 中按此实现 PARA，并把 PARA 与 PRAC 作为"阈值依赖型缓解"的代表评估 DejaVu 影响：真实 ACmin 因写历史低于测得值 → 缓解阈值需要 guardband；N_RH=64 时 −20% 使 PARA 平均性能开销 6.3%、+20% 使性能提升 7.8%（60 个四核 mix 几何平均，相对无缓解基线）。概率型刷新只保护攻击行少数邻行、无计数状态，因此是"低开销 vs 概率性残留风险"权衡的典型代表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Ramulator 2.0 控制器插件（update(DRAM_CMD, ADDR) 回调，见 Ramulator 2.0 条目），或真实控制器固件/硬件概率逻辑；DejaVu 的 perf_eval 目录 + Docker 镜像 richardluo831/ramulator2 可复现阈值扫描实验。使用：作为 RowHammer 缓解的基线对照（与 TRR/PRAC/Graphene/Hydra 等对比）、阈值-性能权衡研究（DejaVu §10）。局限：只刷新少数邻行，无法覆盖 ColumnDisturb 列级破坏面；概率保证在极严格安全需求下不足。

Sigries 视角（ISCA'26）：PARA/PRA 被归类为"行采样（row-sampling）"类防御——每次行激活以概率 p（p≪1）采样并把该行当作 aggressor 处理（执行刷 victim 行等缓解动作），足够高的采样率概率性保证 aggressor 行无法逃脱采样。Sigries 指出其关键缺陷：**无法区分系统是否受攻击**，因此带宽/延迟开销恒定（即使在常见无攻击情形），DREAM-R 利用 DRFM 特性降低开销但仍恒定——这正是 Sigries 只在 heavy mode（检测到疑似攻击后）才切换到行采样的动机；采样率 p 的取值依据作者先前工作（Saroiu & Wolman, DRAMSec 2022）按目标阈值与逃逸概率推导。Sigries 评估中 PARA/PRA 对全部 commodity workload 都发 DRFM 消耗带宽（Graphene/Sigries/PRAC 则零开销）。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
