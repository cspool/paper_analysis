## SeGraM（序列到图比对加速器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SeGraM（Scalable sequence-to-graph alignment）是 ISCA 2022 的通用基因组"序列-图/序列-序列"映射硬件加速器（CMU-SAFARI，基于 GenASM bitvector 近似字符串匹配框架）。核心算法 BitAlign——首个 bitvector 序列-图比对算法：利用位并行、支持图的非相邻字符（hop/边，区别于序列-序列只邻接字符）、预处理拓扑排序快速解数据依赖、每节点只存 k+1 个 bitvector（R[d]），traceback 时按需再生 3(k+1) 个/边，内存占用至少省 3×；并含 MinSeed（首个 minimizer 种子加速器）。SeGraM 支持短/长 reads 的 S2G 与 S2S 映射（S2G 比 GraphAligner 快 5.9×/106×（长/短 reads）、功耗低 4.1×/3.0×；S2G 对齐比 PaSGAL 快 41×–539×）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 GRAINS 的 alignment-based read mapping 工作流中，SeGraM 作为后端的对齐引擎：① GRAINS（或 Fulgor/MetaGraph）先做 k-mer 集合匹配，识别出候选图区域（candidate graph regions）；② 候选区域与 reads 喂给 SeGraM，SeGraM 用 BitAlign 对这些区域做编辑距离阈值 k 内的 bitvector 动态规划比对，得到 read 与图的精确差异（变异位点）；③ 结果整合回分析流程。评估：SeGraM+GRAINS 平均比 SeGraM+Fulgor、SeGraM+MetaGraph 快 6.2×/9.0×（用 SeGraM 报告的对齐吞吐），证明 GRAINS 作为 SCC 前端与对齐加速器正交互补，可直接集成。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：github.com/CMU-SAFARI/SeGraM（ISCA 2022 论文的软件实现）。使用流程：图预处理（拓扑排序）→ 输入 query read 与编辑距离阈值 k → 输出对齐结果；硬件为 HBM2E 等平台的加速器架构。GRAINS 论文把它作为 alignment-based mapping 的评估集成对象（参考文献 [231]），并指出 GRAINS 可与这些"缓解计算/主存瓶颈"的硬件工具灵活集成以进一步缓解其 I/O 开销。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
