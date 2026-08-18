## GH-Architect（自动化硬件生成与设计空间探索引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GH-Architect 是 Graph.hls 的自动设计空间探索与代码生成引擎，把 DSL 算法规格物化为可编译的 Vitis HLS 工程。流程：①DSL → Graph.hls-IR（空间数据流 DAG，节点=基本图计算、边=硬件流数据依赖，5 类节点对应 DSL 原语 iteration_input/map/filter/reduce/return）；②IR 标准优化（常数折叠、死代码消除，IR 结构保持稳定）；③参数解析两阶段——L3 用启发式（设计空间大且图拓扑相关：分析度分布/平均度估计各候选 pipeline 分组的负载均衡、选最高预测利用率），L1/L2 用确定性双向依赖传播（Algorithm 1，约束交集，无启发式）；④代码生成——直接产出 ready-to-compile Vitis 工程（HLS C++ 全部 kernel、host 代码、Makefile、system.cfg 含 SLR/HBM 绑定），约 100ms。
- 每个参数节点配 Analyzer 承担双重角色：约束过滤器（传播时剪掉父集不兼容值）与最终选择器（传播完成后从剩余合法值中选实例化值）。Default Analyzer 只传递+任选；手写 Specialized Analyzer 利用硬件规则与图结构统计做性能最优决策——例：Parallel Lane Count 的 analyzer 过滤阶段丢弃不整占内存总线宽度的 lane 数，选择阶段在 URAM/HBM 容量内选最大化并行吞吐的配置。
- 与 baseline 的对应：解决 Pitfall I（设计探索缺口）——把"跨 200+ 行/10+ 文件的专家手工集成"变为"声明策略 + 自动生成全局一致硬件"。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例（PageRank on Alveo U55C、rmat-21-32，论文完整跟踪）：①DSL→IR 空间数据流 DAG；②L3 启发式：U55C 3 SLR 容纳 14 个 pipeline slot，据 rmat-21-32 幂律度分布（平均度 ~30、~5% 顶点吸引 60%+ 边）分组为 11 little（高密度分区）+ 3 big（稀疏尾部）pipeline；③L1/L2 传播：forward（L3→L2→L1）定硬件可行——每 pipeline 8 PE、每 PE 一个 URAM-backed reduce buffer，32-bit 下 72-bit URAM 行存 2 值，65536 目的节点需 65536/2/8PE=4096 行=8 URAM/PE，8×8=64 URAM/pipeline，14×64=896/960（93%）→ L1 max partition size=65536；backward（L1→L2）定算法正确——PR 贡献 ~rank/out_deg（out_deg=1000 时 ~10⁻³），16-bit 定点最小 delta 2⁻⁸≈0.004 舍入为 0 致假收敛，32-bit delta 2⁻¹⁶≈1.5×10⁻⁵ 足够 → L2 位宽=32。两遍缺一不可：forward 定"装不装得下"、backward 定"算得对不对"；④代码生成：完整 Vitis 工程。
- 编译框架视角要点：决策类型区分——L3 是拓扑相关启发式、L1/L2 是确定性约束求解，二者组合使 DSE 在"组合性（可跨框架）"与"可解性"间取得平衡；切换算法（如 SSSP 整数距离）时 backward pass 自动把位宽放宽到 8-bit（|72/8|=8 值/行），每 pipeline URAM 降到 16、buffer 深度 4×，全部自动重推导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Rust（~10k LOC 框架一部分），CLI `--emit-hls <dsl> [dest]` 生成 HLS 工程；内置 AMD Alveo FPGA 模板或自定义硬件约束输入。开源于 https://github.com/pku-lemonade/Graph.hls（MIT），archived https://doi.org/10.5281/zenodo.19451706。
- 使用与验证：编译产生优化 HLS C++ ~100ms → 可选 Vitis 综合 4–6h 出 bitstream → FPGA 执行 1–10 min/图；或在综合前用 GH-Scope IR 模拟 + golden reference 对比验证。效果：U55C vs ReGraph（L2/L3 匹配、L1-only 探索）平均 2.6×、U200 vs ThunderGP 平均 1.2×、全 L1+L2+L3 DSE 达 4.48×（SSSP）。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
