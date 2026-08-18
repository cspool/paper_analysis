## Graph.hls DSL（Graph.hls Frontend：流式 map/reduce/filter 数据流图处理 DSL，GAS 超集）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Graph.hls DSL（论文称 Graph.hls Frontend）是面向 FPGA 图处理加速器的领域专用语言，用流式函数原语描述图算法硬件数据通路。核心 API（Table I）：`iteration_input(src)`——从图结构数据（如 G.EDGES）生成初始边流（Stream<edge>）；`map([stream], lambda)`——对流的每个元素应用变换（Stream<inferred>）；`filter([stream], lambda)`——过滤 lambda 返回 false 的元素（Stream<origin>）；`reduce(key, val, lambda)`——按 key 用归约函数聚合（Stream<key,val>），要求 lambda 可交换可结合；`return <obj> as <target>`——把输出流写入指定顶点属性（结果节点属性 target，如 result_node_prop）。上下文访问器：`self.<prop>` 读当前顶点属性、`e.src`/`e.dst`/`e.<prop>` 读源/目的/边属性。算法体写在 Iteration block 内（一次完整数据通路 = 一个处理 pass），Node/Edge 属性在 schema 中声明，架构参数可选写在 HierarchicalParam/HlsConfig 块（L2/L3 缺省由 GH-Architect 推断，L1 总是自动推断）。
- 论文刻意声明 DSL 是 GAS 模型的**超集**：GAS 可映射为特例——Scatter ≈ iteration_input + map(边流)、Gather ≈ reduce(可交换可结合 lambda)、Apply ≈ 归约后 map 读 self 属性。超出 GAS 的表达力来自原语组合：Belief Propagation 需聚合所有入边邻居"除目标邻居外"的消息（GAS 的 undifferentiated Gather 无法表达选择性排除），DSL 中以 filter 先排除目标边、再 reduce、再 map 算出出边消息。DSL 依赖三条算法假设：数据流无序（允许硬件乱序处理边、最大化带宽利用率）、reduce lambda 可交换可结合（支持多分区空间聚合）、行为可完全表达在单迭代边界内（主机迭代调用统一 kernel 直至收敛；Dijkstra 这类顺序依赖算法建议用 Bellman-Ford 式并发松弛等价形式）。
- Web evidence（GitHub pku-lemonade/Graph.hls）：支持算法 SSSP/PageRank/ArticleRank/CC/WCC/BFS，DSL 数值类型含 int<N>/fixed<N,F>/float<N>，仓库含 50+ 拓扑变体 DSL 与 docs/ 文档。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中的位置：DSL 是前端输入，GH-Architect 解析它 → 降为 Graph.hls-IR（空间数据流 DAG，5 类节点即 DSL 原语）→ 参数解析 → 生成 Vitis HLS C++ 工程。运转流程例（Figure 5a PageRank）：`iteration_input(G.EDGES)`（第 13 行）生成边流 → `map(lambda e: e.src.rank/self.out_deg)`（14–15 行，并发读上下文属性）→ `reduce(e.dst, val, lambda a,b: a+b)`（16–17 行聚合）→ `return rank as result_node_prop`（19 行，指定硬件输出目标）。
- 设计意图：把"算法逻辑"与"硬件物理变换"解耦（对比 Green-Marl/GraphIt 面向 CPU/GPU 的指令式抽象无法映射到空间 FPGA pipeline）；让领域专家无需硬件知识即可声明算法，硬件参数交由编译器推断。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Graph.hls DSL 前端由 Rust 实现（chumsky 解析器），`--emit-hls <dsl> [dest]` 从 DSL 文件生成 HLS 工程、`--simulate-json <dsl> <graph.json> [max_iters]` 运行 GH-Scope IR 模拟。仓库提供全部 6 个评估算法（PR/SSSP/Weighted SSSP/CC/AR/WCC）的 DSL 规格。
- 使用：用户写 DSL（Iteration block + 属性 schema + 可选参数块）→ cargo 编译的 Graph.hls 二进制消费 DSL → GH-Architect 推断 L1 参数（分区比、buffer 大小由目标图数据集与 FPGA 平台自动推断）→ 输出可综合 HLS 工程（~100ms）。意义：把 ReGraph 中"位宽修改 200+ 行跨 10+ 文件"的侵入式重构降为"单行参数 + 自动传播"，并让 Belief Propagation 这类不规则数据流可表达。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
