## SetConnect / Propagate Scheduling Primitives (WELDER)

术语是什么？
SetConnect和Propagate是WELDER的两个核心tile-graph调度原语。**SetConnect(Edge *edge, MemLevel level)**: 在edge上设置数据复用的目标memory level，决定两个operator-tile之间中间数据在哪层复用。**Propagate(TileGraph g, Map<Axis, Dim> config)**: 给定output tile轴尺寸，通过tensor expression链式推断所有tile shape。两者构成完整tile-graph更新接口：SetConnect修改edge（决定哪里复用），Propagate修改node（决定如何切分）。

从编译框架角度拆解术语：
Graph Connecting调度：枚举每条edge的所有memory level → SetConnect(edge, level) → ExtractSubgraph(connected nodes where connect_level > 0) → SubGraphTiling(subgraph, Propagate) → Profile得到latency → 选择最优level。ExtractSubgraph利用connect_level属性识别连通sub-graph。

术语一般如何实现？如何使用？
SetConnect更新edge的connect_level属性。Propagate通过逆拓扑序+仿射变换:tensor expression实现shape inference。对多输出node需对齐两次propagation结果（若tile shape不一致则不连接该output node）。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---
