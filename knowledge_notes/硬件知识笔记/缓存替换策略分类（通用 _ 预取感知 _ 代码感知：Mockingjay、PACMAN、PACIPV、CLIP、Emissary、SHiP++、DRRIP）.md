## 缓存替换策略分类（通用 / 预取感知 / 代码感知：Mockingjay、PACMAN、PACIPV、CLIP、Emissary、SHiP++、DRRIP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
论文按驱动因素把缓存替换策略分为三类（II 节）：①通用（general-purpose）——用块 recency（无 miss 历史）或与过去访问相关、预测复用距离的特征驱动替换；SOTA 为 Mockingjay（用长 PC 历史中的模式精确预测复用距离），DRRIP/SHiP++ 亦属此类（SHiP 用历史签名预测）。②预取感知（prefetch-aware）——区分 demand 与 prefetch 请求做替换决策：PACMAN 动态调整插入/提升策略以缓解错误预取的负面影响；PACIPV 基于离线探索不同 demand/prefetch 行的插入与提升 RRPV 组合。③代码感知（code-aware）——面向代码足迹大的 server 负载，优先代码行：CLIP（基于 RRIP，提高 L2C 中代码行优先级，代价是更多数据 miss）、Emissary（防止 L2C 中最关键的代码块被逐出）。IP-CaT 的 TIPRP 是"预取代码行感知"的新类别（专管 L1I 预取行），在 105 个 workload 上全面超越上述策略（如 EPI 下超 Mockingjay/PACIPV/SHiP++/SRRIP/DRRIP 8.0%/3.0%/5.3%/4.4%/3.6%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
各策略在缓存层次中的定位（表 II）：baseline 为 L2C LRU + LLC SHiP；CLIP/EMISSARY/SRRIP 应用于 L2C，PACIPV/PACMAN/DRRIP/SHiP++/Mockingjay 应用于 LLC，CHiRP 应用于 sTLB，Morrigan 为指令 TLB 预取器。评估方式：把每个策略放入 ChampSim 对应缓存替换模块，配合同一 L1I 预取器（EPI/Barça/FNL+MMA）跑 105 个 server workload，统计 geomean speedup、L2C/LLC MPKI 与 miss 延迟。论文还对比"LLC 原设计的策略移到 L2C"的变体（Mockingjay/PACIPV/SHiP++/SRRIP/DRRIP 应用到 L2C），IP-CaT 仍最优（EPI 下超 8.0%/3.0%/5.3%/4.4%/3.6%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：均为 ChampSim 中的替换策略模块（RRIP/SHIP++/DRRIP/Mockingjay/PACMAN/PACIPV 有公开实现，CLIP/Emissary 见各自论文/artifact）。使用方式：作为缓存替换策略 baseline 与新的预取/替换方案对比；评估时需注意策略应用层级（L2C vs LLC）与预取器配置一致。局限：这些策略均未专门针对 L1I 预取行管理（通用/LLC 优化导向），IP-CaT 填补了"L1I 预取代码行 L2C 管理"的空白。

R-Max 补充视角（ISCA'26，MIN+预取扩展）：R-Max 把 Bélády's MIN 从"miss 驱动的替换"扩展为"预取时机+替换联合决策"（C.-Bel-adys-Optimal-Cache-Replacement-Algorithm-.md 与 C.-R-Max.md）：理想预取器的目标是让每个 set 都按下次使用优先级装满块——先按访问顺序预填空 way，某块被访问后比较其下次访问时间与"不在 set 中但未来将被访问的块"的时间，若后者更早则预取后者替换最远未来使用的块。替换与预取在 R-Max 中统一由 dead block counter 驱动（计数器归零即发预取替换）。无预取 MIN 单独在 L2 增益很小（~5.5%），说明 L2 的替换策略改进空间有限、主要空间在预取；而大多数替换策略工作聚焦 LLC（更大关联度与过滤作用），R-Max 结果与之一致。
涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
