## Gaze（延迟触发的低存储空间足迹预取器）

术语解释
低存储空间足迹预取 baseline（4.46 KB）：把预取下发推迟到第二访问点，利用同一区域内 offset 之间的短时相关（internal temporal correlations）提高预测质量；STEP 在 Gaze 式低存储设定上实例化其多触发机制。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gaze（Chen et al., HPCA 2025）针对"首访问触发上下文不足"的问题，把下发推迟到更晚触发点，从而用更简单状态获得更准决策——代表"晚触发"这一单点设计方向。硬件：8-way 64-entry FT、8-way 64-entry AT（存 64-bit footprint + 近期访问 offsets）、256-entry 4-way PHT（64-bit footprints）、32-entry PB，总 4.46 KB；并含 dense-PC 流检测（DPCT）。STEP 论文为公平比较把 Gaze 配置为仅 L2 填充（消除多级填充策略优势），并验证开启其原始多级预取不改变定性结论。本地证据：主文件（score 1272）、`A.-Simulator...`（score 594）、`C.-L1-level...`（score 497）、`F.-Storage-Overhead.md`（score 381）。
- 评估结果：Gaze 总均值 1.24×（L2）vs STEP 1.28×；L1 级 1.25× vs STEP 1.28×。与 STEP 更可比（设计哲学相近：offset 事件表示、低存储），是隔离"分阶段触发决策"收益的干净参照点。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转：固定第二访问点触发——首访问只记录，第二访问累积两个 offset 后再查 PHT 下发；相比 FOE 首触发，Gaze 错过页上下文在第二访问前消失的短命机会（mcf-192），但避免了首触发的歧义（mcf-484 类场景由 STEP 的 TOE 提供同样的后期消歧能力）。作为 STEP 的"晚触发单点"对照：STEP 在该点之上增加 FOE 早机会与 TOE 更精确消歧，形成三点序列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ChampSim prefetcher/ 目录实现其 FT/AT/PHT/PB/DPCT；评估用 SPEC CPU2006/2017/CloudSuite 130 trace。使用场景：作为低存储空间足迹预取器的代表 baseline，衡量"把触发时机变为运行时决策变量"的增量收益。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
