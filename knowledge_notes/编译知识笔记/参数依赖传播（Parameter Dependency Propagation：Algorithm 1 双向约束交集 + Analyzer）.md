## 参数依赖传播（Parameter Dependency Propagation：Algorithm 1 双向约束交集 + Analyzer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 参数依赖传播是 GH-Architect 解析 L1/L2 参数的确定性算法（Algorithm 1）：给定依赖图 G、用户规格 U、固定 L3 策略 s_{L3}，产出合法参数组合 C。初始化：每个节点可用集 A(v)=全值域；应用 U 约束对应节点；从 s_{L3} 设根约束；迭代至固定点：每轮按拓扑序遍历，A'(v) = A(v) ∩ ⋂_{u∈parents(v)} Analyzer_u(A(u), v)（父节点约束经 analyzer 过滤后交集），空集则无解返回；再按逆拓扑序遍历，A'(v) = A(v) ∩ ⋂_{w∈children(v)} BAnalyzer_w(A(w), v)（子节点反约束）；最终返回 CartesianProduct({A(v)})。
- 关键设计：把"完整约束传播需两遍 BFS（root→leaf + leaf→root）"改进为"正反两遍在每轮内交错"，减少达到固定点所需遍历次数；全程确定性、无启发式。每个参数节点一个 Analyzer，承担约束过滤器（向子节点传值前剪掉不兼容值）与最终选择器（传播完成后选实例化值）双重角色；Default Analyzer 仅透传+任选，Specialized Analyzer 利用硬件规则与图结构统计做最优选择。
- 为什么需要双向：图参数有双向跨层依赖（III-A4）——L3→L2→L1（父约束子：ε 精度→位宽）与 L1→L2（子约束父：位宽→分区粒度、内存布局→buffer 大小）。PageRank 例证明单向不够：forward 定"物理可行"（URAM 放得下 32-bit 且 16-bit 也放得下）、backward 定"算法正确"（16-bit 定点把 ~10⁻³ 贡献舍入为 0 致假收敛 → 剪掉 16-bit），两遍交集才得唯一解 32-bit。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中：是设计空间求解器，与 L3 启发式互补——L3 是拓扑相关搜索、L1/L2 是确定性约束求解，二者构成 GH-Architect 的完整 DSE。运转流程例（SSSP 8-bit 自动推导）：L3 固定后，forward 传播——每条 72-bit URAM 行按位宽容纳 |72/bitwidth| 个值：8-bit 时 8 值/行 → 每 pipeline URAM 需求降到 16（vs 32-bit 的 64）→ 可行 buffer 深度 4×；backward 传播——整数距离 SSSP 无精度约束 → 8-bit 保留为合法 → 最终选 8-bit。开发者零手工修改。
- Analyzer 例子：Parallel Lane Count 的 specialized analyzer 过滤阶段丢弃不整占内存总线宽度的 lane 数（如 256-bit 总线须 4/8/16 lane 而非 6/12），选择阶段在 URAM/HBM 容量内选最大化并行吞吐的配置——把硬件约束编码进约束传播而非全局搜索。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：图参数以 DAG 组织，Analyzer 为 per-node 组件；硬件约束来自内置 AMD Alveo FPGA 模板或自定义用户输入；Rust 实现（框架一部分），开源 https://github.com/pku-lemonade/Graph.hls。
- 使用与可扩展性：新增参数只需定义值域、DAG 依赖边与约束逻辑（50–100 行），远低于 monolithic HLS 代码库中"一个优化改十几个文件"的成本；BRAM power-gating（L2：enable 信号由分区索引推导）与 CGRA 式空间映射（L3：cluster 构建方式/大小/lane 分配策略，自动传播到 L2 buffer/lane 数）均为论文给出的扩展示例。价值：把"位宽/分区/流水组合的手工协调"变为约束求解，使跨框架策略可组合且组合合法。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
