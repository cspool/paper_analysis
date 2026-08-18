## Bingo / eBingo（空间足迹预取 baseline）

术语解释
Bingo 是评估中最强的单触发空间足迹预取 baseline 之一，通过更丰富的匹配机制强化单点触发（specific 历史、fallback 到 coarse 历史、多候选聚合）；eBingo 是 STEP 论文的 ISO-storage 增强版（Bingo + 与 STEP 相同的 dense-PC 流检测）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bingo（Bakhshalipour et al., HPCA 2019）不依赖单一事件键，而是强化单触发预取：可用更 specific 的历史时用 specific，不可用时 fallback 到 coarse 历史，并在触发点聚合多个匹配候选。STEP 论文把 Bingo 增强为 eBingo（加 Gaze 式流检测），作为"强化固定触发 baseline"；eBingo 存储远大于 STEP（Bingo 类地址/页键设计 >100 KB），需 >100 KB 元数据才能逼近 STEP 10.5 KB 工作点（存储敏感性实验 Fig.19）。本地证据：主文件（score 1253）、`A.-Simulator...`（score 560）、`I.-Storage-Sensitivity...`（score 476）、`F.-Storage-Overhead.md`（score 385）。
- 评估结果：单核 SPEC06 eBingo 1.47× vs STEP 1.49×，SPEC17 1.34× vs 1.40×，CloudSuite 并列 1.07×，总均值 1.26× vs STEP 1.28×；多核同构 STEP 全程领先，异构 8 核 eBingo 打平 STEP。mcf-484 案例中 eBingo 因 PC+Address rich 匹配在首触发点即可区分多数有用模式而胜 STEP（代价更高污染）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转：单固定触发点（首访问）用 PC+Address 查历史，specific 命中即下发；否则 fallback 到 coarse 历史再试；多候选时聚合；全部在 ChampSim L2C 层实现、与 STEP 相同 MSHR 与预取队列预算。作为 baseline 的硬件视角：同一定时决策（固定触发点）下靠"匹配更丰富"而非"时机可变"提升质量，体现与 STEP 的组织原则差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 ChampSim prefetcher/ 目录实现 Bingo 的 FT/AT/PHT + specific/coarse 两级匹配 + 候选聚合，并加 DPCT 流检测构成 eBingo；与 STEP 在相同模拟环境公平对比（ISO-storage 配置）。使用场景：作为空间足迹预取研究的强 baseline，特别用于验证"多触发时机"是否比"更强匹配"更优。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
