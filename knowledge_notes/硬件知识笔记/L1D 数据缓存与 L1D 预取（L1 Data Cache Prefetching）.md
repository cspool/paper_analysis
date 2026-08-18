## L1D 数据缓存与 L1D 预取（L1 Data Cache Prefetching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- L1D（一级数据缓存）是处理器离核心最近的私有数据缓存，容量仅几十 KB（Moirai 配置 48KB 12-way、5-cycle 延迟，Intel Alder Lake 为 32KB）。它是预取的最强位置：观察完整未过滤请求流、最低注入延迟、可访问 PC/核心上下文；但也是最资源受限位置。Moirai 的核心命题是"L1D 预取悖论"：记忆式预取器够小但不强（对复杂模式脆弱），泛化式 ML 预取器强大但太大太慢只能放 L2/LLC。论文由此提出架构问题：在几 KB 预算内，是把预算投给更大的记忆表（增量收益），还是投给一个激进高效的泛化模型（机会成本收益）——Moirai 选择后者。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- L1D 预取数据路径（Moirai）：LSU 产生 speculative VA → 与 L1D tag/dTLB 并行送预取器 → 预取器 3 周期内产出预取 VA（CaPNet 前向，2.5-4GHz）→ PRQ → 预取沿 L1D→L2→LLC→DRAM 取数填充 L1D（命中 L2/LLC 则无片外流量，全部 miss 才 DRAM 访问）→ L1D 按 LRU 替换。每个 L1D 预取填充引发的 eviction 分 harmless（不再访问）/harmful（真正污染）两类：Moirai 的 eviction 绝大多数 harmless（与 IPCP 相当），DRAM 访问率低于 IPCP，证明不加剧片外带宽压力。
- 为什么 L1D 位置关键：L1D 过滤掉绝大多数请求，L2/LLC 预取器只见 decimated 非连续流（失去完整时序上下文）；物理距离使预取注入延迟化，准确预测也失效。Moirai 在 L1D 提供双重收益：未过滤的语义序列可见性 + 及时隐藏延迟的注入。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：L1D 预取器作为独立硬件模块挂在 LSU/L1D 旁（Moirai 的 Input Processing Unit + CaPNet Engine + Adaptive Control Unit，780 Bytes），ChampSim 中在 prefetcher 模块实现（moirai.cc/h）；预取 VA 经 PRQ 注入。ChampSim 中 L1D 配置 48KB 12-way 5-cycle LRU；L1D 预取常用 block+1 等硬连线偏移兜底（Moirai 每次 demand miss 硬发 block+1，零存储）。使用场景：通用数据预取（SPEC/图负载），Moirai 对 omnetpp 11.53%、GAP bfs 70.65% speedup。局限：L1D 预算严苛，ML 预取器需极端压缩（二值化）才能落地。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
