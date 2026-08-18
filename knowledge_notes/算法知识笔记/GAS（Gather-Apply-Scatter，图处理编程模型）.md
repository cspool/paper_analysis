## GAS（Gather-Apply-Scatter，图处理编程模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GAS 是图处理的高层编程抽象（源自 PowerGraph）：每次迭代分为 Gather（从邻居聚合消息）→ Apply（用聚合结果更新顶点状态）→ Scatter（把更新传播到邻居），迭代至收敛。有 vertex-centric 与 edge-centric 两个变体；edge-centric 变体以流式顺序访存适合 HBM/FPGA 加速（ACTS、GraphLily、ForeGraph、Swift 等广泛采用，Web 佐证：Swift 的 decoupled-asynchronous GAS 在 8-FPGA 上 12.8× 优于 ForeGraph）。Graph.hls 论文把 GAS 作为 DSL 的 baseline 表达力基准：GAS 是 Graph.hls Frontend 的特例（Scatter≈iteration_input+map 边流、Gather≈reduce(可交换可结合 lambda)、Apply≈归约后 map 读 self 属性），且 DSL 是其超集。
- 关键局限（Graph.hls 动机）：GAS 的 undifferentiated Gather 无法表达"排除目标邻居"的选择性聚合——Belief Propagation 需聚合所有入边邻居"除目标邻居外"的消息，GAS 无法表达；Graph.hls DSL 用 filter 先排除目标边再 reduce 解决。

从算法pipeline角度拆解术语，比如术语如何在算法pipeline中发挥作用，给出术语在算法pipeline中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 算法pipeline 运转流程（PageRank 的 GAS 映射，Graph.hls 论文 Figure 5a）：
```
# 每轮迭代（一次 GAS pass）
iteration_input(G.EDGES)                       # 生成边流（结构数据）
map:   val = e.src.rank / self.out_deg          # Gather：沿出边读源顶点属性
reduce(e.dst, val, lambda a,b: a+b)             # Gather：按键聚合邻居贡献
map(self): self.rank = 0.15 + 0.85 * reduced    # Apply：更新顶点状态
# Scatter 由下一轮 iteration_input 承载（主机迭代调用直至收敛）
```
- pipeline 特征：GAS 天然是"边流 → 聚合 → 状态更新"的三段数据流水；edge-centric 变体下内存访问流式、可乱序处理边（Graph.hls 假设流无序以最大化带宽）；Apply 依赖 reduce 结果构成迭代间依赖，故硬件 kernel 只实现一个 pass、主机负责跨迭代循环与收敛判断（ε 阈值属 L1 参数）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件框架（PowerGraph、Ligra、GraphIt）与 FPGA 加速器（ACTS/GraphLily/ForeGraph/ThunderGP/ReGraph/Swift）均围绕 GAS 构建；FPGA 侧 edge-centric 变体以流式流水 + 片上聚合缓冲（URAM reduce buffer）实现。Graph.hls DSL 把 GAS 编译为空间数据流 DAG（iteration_input/map/filter/reduce/return 五类节点）→ GH-Architect 生成 HLS 硬件。
- 使用：作为"描述任意图迭代算法"的通用模板——需可交换可结合的聚合（GAS 的 reduce 假设）与单 pass 表达；Dijkstra 等顺序依赖算法需改写为 Bellman-Ford 式并发松弛才可高效空间并行。跨论文复用：把 GAS 作为图加速器 DSL 的表达力基准（GAS 可表达 ⊆ DSL 可表达），并据"GAS 无法表达什么"（选择性排除、不规则数据流）定位 DSL 扩展点。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
