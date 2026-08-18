## Event Key（事件键）

术语解释
空间足迹预取器记录历史足迹时伴随的标识符：未来访问靠匹配相同事件键来复用历史足迹。事件键沿频谱分布——高频简单键（页 offset）匹配多但信息少，低频特定键（全地址/页号）匹配少但更精确。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 足迹只在事件键重现时可复用，因此键的设计决定 accuracy-coverage-storage 三角。键空间小（如 4 KB 页 64 个 offset）→ 存储小、匹配频率高、但每次匹配携带信息少、精度低；键空间大（全地址/页号）→ 存储多、匹配少、但更特定、通常更准。STEP 用 offset 作事件键（三个触发点 FOE/SOE/TOE 各提供递增的 offset 信息），并给 FOE 附加 hashed PC（12 bit，不作 tag）消歧调用上下文；相比 eBingo 的 PC+Address 键，offset 键在页上下文快速变化的场景（mcf-192）保留更多可复用历史，但早期歧义需 TOE 解决（mcf-484）。本地证据：主文件（score 1118，event key 多处）、`III.-THE-DESIGN...`（score 871）。
- Web 佐证：SMS 用 PC 或 offset 触发，Gaze 用 offset 间短时相关，Planaria 用页邻近性跨页迁移足迹——都是事件键设计的不同取舍。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件落地：PHT 以第一 offset 作索引、第 2/3 offset 作 tag（SOE 截断上 6 位、TOE 全 tag）；FT/AT 条目各存 hashed PC（12 bit）与 offset 字段；FOE 查表用 FO+PC 组合键。匹配流程：事件到达 → 按事件类型组装键 → PHT 查找 → 命中则取历史足迹交 Evaluator 判定 → 决定下发或推迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：键作为 PHT tag 的组成部分（12b），配合 LRU 与 maturity 位管理；hashed PC 存一次每 PHT 条目与 FT/AT 条目，成本极小。使用场景：所有空间足迹预取器的核心设计自由度；STEP 说明事件键设计与触发时机设计是两个正交维度，可独立改进。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
