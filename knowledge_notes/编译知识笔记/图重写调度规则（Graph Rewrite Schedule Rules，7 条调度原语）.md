## 图重写调度规则（Graph Rewrite Schedule Rules，7 条调度原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QiMeng-Tensify（ISCA'26）定义的动作空间（action space A）：7 条在 TensorIR 计算图上做结构重写的调度规则，每条规则带参数或不带参数——A1 MultiLevelTiling（tiling factors：多级循环分块，适配目标架构存储层次）、A2 AutoInline（自动内联，无参数）、A3 ParallelizeVectorizeUnroll（loop, unroll length：并行化/向量化/展开）、A4 CrossThreadReduction（split factors：跨线程归约切分）、A5 ComputeAtLocation（compute locations：算子融合位置，限定在满足结构/依赖约束的 top-level 单消费者 untiled 中间 block）、A6 AutoBind（自动绑定 CUDA thread/block 维）、A7 InlineConstantScalar（内联常量标量）。这些规则是"粗粒度 program sketch 结构改写"与"细粒度参数配置"的解耦基础。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在编译框架中的运转流程（GatedMLP 优化轨迹 S0→S5，论文 Fig.4）：S0 初始 = GEMM1→SiLU(exp,add,div,mul)→GEMM2→mul 分离 kernel；S1 多次 AutoInline 把 SiLU 子算子折叠成单一 SiLU block；S2-S3 MultiLevelTiling + ComputeAtLocation 在共享 tiling loop (i0,j0,k0) 下 tile 并融合 GEMM1/GEMM2；S4 compute_at(SiLU, GEMM1) 把 SiLU 融进融合后的 GEMM block；S5 再 ComputeAtLocation 把 elementwise MUL 提升进 GEMM2 的 reduction 循环 → 全融合单 kernel、无中间 buffer。参数配置由 Simulation 阶段的 fine-grained parameter specification 填（tile size、unroll length、fusion position、split factors）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：约 1,500 行 C++ 低层调度原语 + Python 层的 TensorIR schedule 操作；与 TVM/MetaSchedule 的调度原语同源但按"图级"组合（跨算子施加，如把一个算子的 compute 位置放进另一算子的循环体）。使用方式：作为 MDP 的动作空间 A，由 LLM 先验 + MCTS 决定"什么状态下应用哪条规则、带什么参数"；与传统编译器（TVM 对"有数据复用且可融合"子图静态选 Tiling+Fusion，表 I）和模板编译器（Mirage 用 CUTLASS/手工模板）的关键差异：规则序列不固定、可任意组合，从而覆盖规则系统/模板系统无法表达的全融合与跨算子 partial reduction。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
