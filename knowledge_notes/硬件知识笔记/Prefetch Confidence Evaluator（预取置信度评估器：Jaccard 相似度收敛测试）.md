## Prefetch Confidence Evaluator（预取置信度评估器：Jaccard 相似度收敛测试）

术语解释
STEP 引入的轻量硬件组件：在每个时序触发点，对 PHT 返回的最近 N=3 个匹配历史足迹做两两 Jaccard 相似度测试，全部相似度 > 阈值 T=0.75 才判定"收敛"并下发预取（取交集），否则推迟到下一触发点累积证据。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间足迹匹配不是单一置信状态：同一事件键可能匹配多个候选足迹，置信度取决于多个观察点的多个候选足迹是否一致。评估器把"是否够成熟值得发预取"量化成集合收敛测试：取 PHT 返回的最近 N 个匹配足迹（默认 N=3），以最新足迹为基准与其余 N-1 个计算 Jaccard 相似度 $J(A,B) = |A \cap B| / |A \cup B|$，全部超过阈值 T（默认 0.75）则集合收敛，按匹配足迹交集下发（优先精度而非覆盖）；否则不发，等待下一触发点。限定最近 N 项使硬件小、避免依赖过时阶段的旧模式。本地证据：`paper_secs/.../B.-Prefetch-Confidence-Evaluation.md`（score 1895）、`D.-Hardware-Organization-and-Dataflow.md`（score 2164）。Jaccard 作为集合相似度量在 vault 中另有 LLM 注意力语境佐证：`knowledge_notes/算法知识笔记/Cross-Head Spatial Locality*.md`（score 141，KV head 对 Jaccard ~0.4-0.8）、`Intra-Group Layer KV Cache Indices Sharing.md`（score 84）。
- 与经典 stream/stride 预取器的置信度机制（靠重复稳定模式观察门控激进性）精神相通，但扩展到时序连接多个触发点对空间足迹预取不直接——故用"多候选足迹跨观察点一致性"代替单一置信状态。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程（以 FOE 为例）：FT 把 offsets + PC + event ID 转发给 PHT → PHT 按 FO+PC 取最近 N=3 个匹配条目 → 若仅 1 个匹配，检查 maturity 标志（不成熟则抑制下发、推迟）→ 否则把匹配足迹送入 Evaluator → Evaluator 算两两 Jaccard，全 >0.75 则把交集 footprint 推入 PB 并发预取，同时通知 FT 该页已由 FOE 下发；否则不动作等 SOE。SOE 同理（查 FO+SO 匹配），TOE 用满 tag 高特异性、命中即下发。
- 硬件代价：仅需 N 个最近足迹的小寄存器窗口 + 组合逻辑做 Jaccard 比较（每对一次与/或/计数），N=3/T=0.75 下开销可忽略，与 PHT 共享数据通路。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PHT 维护每键最近 N 个足迹（替换最近最少用条目），Evaluator 读取窗口内足迹按 Jaccard 阈值判定；单匹配冷启动场景用 per-entry 1-bit maturity 标志（新插入条目不成熟，仅在 hashed PC 与同位置被逐条目一致即"重现过"时标记成熟）防误信。使用场景：任何空间足迹预取器的触发决策点；论文消融（STEP-D1/D2/D3）显示该机制使 FOE 贡献早期机会、TOE 恢复精度、SOE 平均冗余（最终配置禁用 SOE 下发但保留为一般框架）。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
