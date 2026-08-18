## Dense PC Table（DPCT）与 dense-PC 流式模式检测

术语解释
轻量流式访问检测器：用一张小表记录近期产生密集连续访问的 PC（dense PC），识别顺序流式访问模式并独立处理，避免其占用空间足迹 PHT 的历史容量。STEP 复用 Gaze 的该机制。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 流式访问（顺序读连续 cache line）在许多 workload 中常见，可被更简单的专用机制高效处理；若让主 PHT 学习这类模式，会浪费历史容量并干扰非流式足迹学习。dense-PC 检测：DPCT（Dense PC Table）记录近期密集 PC，命中则按流式方式连续预取。STEP 采用与 Gaze 相同的轻量 dense-PC 检测器（DPCT），与 eBingo 共享同一机制；该组件与 STEP 核心贡献（分阶段触发决策）正交。本地证据：`E.-Streaming-Pattern-Prefetching.md`（score 406）、`D.-Hardware-Organization-and-Dataflow.md`（Table I 中 DPCT 0.015 KB）。
- Web 佐证：Gaze（Chen et al., HPCA 2025）用 dense-PC 流检测处理流式模式；eBingo 即 Bingo 加该检测的 ISO-storage 增强版。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件配置（Table I）：DPCT 8-entry，每项 Hashed PC 12 bit + LRU 3 bit，共 15 bits/entry，总 0.015 KB，几乎零成本。运转：访问流中若某 PC 在短窗内产生密集连续地址，DPCT 记录之；后续该 PC 再次出现即触发连续预取（流式），与 PHT 足迹学习并行、互不干扰。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DPCT 组相联小表 + 密集度判定逻辑；作为预取器 add-on 组件与主 PHT 并列。使用场景：任何空间足迹预取器都可叠加流式检测以覆盖顺序访问；论文消融（Fig.14）把 STREAM 触发点单独统计，显示其贡献"easy-win"的流式预取。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
