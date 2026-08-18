## 层次化设计抽象（Hierarchical Design Abstraction：L1 图处理常量 / L2 图微架构配置 / L3 图数据流策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Graph.hls 按**修改成本**把图加速器设计空间组织为三层的层次化抽象（Figure 4，越深阴影成本越高、影响越广），使来自不同研究工作的优化可组合。L1 图处理常量：只改变算法行为、不动硬件数据流结构（顶点属性初值 PageRank 1.0/|V|、SSSP +∞；收敛阈值 ε；活跃顶点过滤阈值——决定 big/little 分区路由），具 graph-semantic locality，修改隔离在 kernel 内。L2 图微架构配置：影响顶点/边/属性在内存的表示、打包与每周期处理方式，传播整个加速器但数据流结构固定（顶点属性位宽 16/32-bit、边属性表示、并行 PE 数、内存总线宽度、片上 buffer 大小）。L3 图数据流策略：顶层数据流决策，影响从预处理到聚合的每个组件与 host 协调代码（pipeline 分组策略、vertex/edge-centric 执行模型、分区到 SLR/HBM 通道的映射）。
- 跨层依赖（Section III-A4）：收敛阈值 ε 决定属性位宽精度（PR ε=10⁻⁶ 需 32-bit 浮点、ε=10⁻³ 16-bit 定点即可）；属性位宽决定分区粒度（16-bit 使 2× 大分区 1M vs 512K 顶点装进 URAM）；分区策略决定边属性表示（vertex-cut 复制顶点需决策权重复制 vs 分存）。
- 动机量化：ReGraph 参数散落 5 makefiles/21 headers/14 sources，32→16-bit 位宽修改级联 200+ 行跨 10+ 文件；L3 级"固定 2-class 分区泛化到 N-class"需改 1000+ 行（分区预处理器、目的 buffer、pipeline 计数、N 个 merger、host 多类调度与内存分配）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在编译框架中的作用：该抽象定义"什么可被调优"，GH-Architect 依此实现"如何调优"——L3 先启发式固定，L1/L2 再确定性传播（Algorithm 1）。运转流程例（SSSP 16-bit 优化）：用户只改一行 L2 位宽参数 → GH-Architect 自动传播级联改动：属性 typedef、HBM packing 位区间（src[i]>>4→>>5）、每周期顶点数、归约树、算法常量 MAX_I 2147483647→32767、host 分配/解包逻辑——全部由编译器重生成，无需手工跨文件修改。
- 例子（Figure 5b）：L1 调顶点过滤阈值 → 改变 big/little 数据分布比、不影响结构；L2 32→16-bit → 内存减半、带宽利用率翻倍、数据流组织不变；L3 2-class→3-class 分区（两个 65K dense + 一个 524K sparse）→ 完整架构重设计自动生成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：参数以 DAG 组织（节点=参数、边=依赖），每个节点有值域 + Analyzer；GH-Architect 用依赖传播做约束交集（forward/backward 交错至固定点）产出唯一有效参数组合；内置 FPGA 模板（如 AMD Alveo）或自定义用户输入提供硬件约束。扩展新优化参数只需定义值域、依赖边与约束逻辑（约 50–100 行），如 BRAM power-gating 可表达为 L2 参数（GH-Architect 为每个 BRAM 数组标注 enable 控制逻辑、enable 信号由分区索引推导）。
- 使用：用户在 DSL/HlsConfig 中声明 L3 策略与可选 L2/L1 值；GH-Architect 自动解析跨层依赖并生成全局一致硬件。价值：使 ReGraph/ThunderGP/GraphLily 等框架的策略（big-little 分区、统一分区+顶点缓存、overlay 重构处理）首次可经配置组合，用 GH-Scope baseline 对比在综合前评估组合设计。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
