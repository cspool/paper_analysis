## 分发网络与归约网络（Distribution Network, DN / Merge-Reduction Network, MRN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DN（Distribution Network，分发网络）与 MRN（Merge-Reduction Network，归约网络）是多功能稀疏加速器 PE 阵列的两个可重构路由部件，源自 Flexagon 的架构词汇，被 Trapezoid 采纳为"只重配置两个部件"的折中设计，Harmonia 沿用为硬件基板：DN 负责把输入操作数从 SRAM/上游按数据流模板广播或选择性路由到各 PE 行（Benes 网络可无冲突地实现任意置换/广播）；MRN 负责把各 PE 产生的部分积（partial sums）按模板归约——merge-before-store（先归约再写缓冲，InP/Row 用）或 column-accumulate（按输出列累加，OutP 用）——并把结果送回缓冲/SRAM。DN 与 MRN 的可编程路由/归约模式就是"数据流模板"的物理载体：不改 PE 乘加数据通路，只改 DN 路由表与 MRN 归约模式即可在 InP/Row/OutP 间切换。Harmonia 单 PE 行：DN 为 16-to-16 Benes+控制（7,001 μm²/8.7%）、MRN 为 radix-16 FP32 加法树（23,234 μm²/28.8%），在 32 行加速器中合计构成计算部分（2 PEs×32 Rows，5.16 mm²/68.7%）的核心。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程例子（Row 数据流下一个 tile）：SRAM 把 A 行送入各 PE 行的驻留缓冲，DN 按该 A 行 nnz 索引集合配置路由表，只把 B 中对应行片段送到相关 PE 并缓冲共享 → 各 PE 做乘加产生部分积 → 本行 MRN 以行顺序归约（merge-before-store）把部分积压缩后写回 on-row BUF → 溢出时经 spill 计数器反馈给 Tiling Controller。切换 OutP 时：DN 重配置为广播 (A_*k,B_k*)，MRN 改 column-accumulate，buffer 策略重设。反馈路径与主数据通路完全解耦（每 PE 行维护 128 个计数器跟踪 MRN merge 事件，占 <0.5% PE 阵列面积，经轻量 metadata crossbar 汇聚），保证 DN/MRN 时序不受 profiling 扰动。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Benes 网络（DN，可编程置换/广播，O(log N) 级）、radix-N 加法树（MRN，按 N 路输入流水归约）；面积/能耗用 RTL（TSMC 28nm Synopsys DC 综合）+ CACTI 7（SRAM）建模。使用：作为数据流模板的执行载体，DN/MRN 模式在 tile 边界由 Reconfiguration Engine 重编程（20–50 cycles）；MRN merge 深度是运行时反馈信号（merge-depth monitor），用于判断数据流是否失配并触发切换。Harmonia 论文未开源，DN/MRN 概念最早出自 Flexagon（ISCA'20）。相关评估：Trapezoid 以此二部件实现 InP/Row 双模板近常效，Harmonia 扩展为三模板 + 运行时切换。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
