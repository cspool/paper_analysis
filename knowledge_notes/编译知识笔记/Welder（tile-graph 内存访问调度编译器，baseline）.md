## Welder（tile-graph 内存访问调度编译器，baseline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Welder（清华/华为，OSDI'23，commit af53ab1）是面向 GPU 的深度学习编译器：用 tile-graph（tile 级数据流图）显式建模内存访问，通过 tile-graph 级别的 fusion 与调度优化 kernel 的内存访问模式，支持 memory-intensive 与 compute-intensive 算子的融合（把中间结果缓存进 shared memory）。QiMeng-Tensify（ISCA'26）把它作为 exploration-based baseline：Welder 扩大了融合范围（相对 Astitch 只融合 memory-intensive、Chimera 只融合 memory+compute 相邻对），但其 tile 级 fusion 受限于固定规则启发式（编译 <0.01h 的近即时），且有限的 intra-operator 优化使其无法探索 thread 级融合，产生次优 kernel（子图平均 QiMeng-Tensify 快 13.49×，为所有 baseline 中最大差距）。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
作为 baseline 的运转流程（GatedMLP）：输入算子 → 构建 tile-graph（每 tile 标注内存访问模式）→ 按规则化启发式做 tile-graph 融合与调度（默认融合/调度策略按文档推荐）→ 生成 CUDA kernel。局限：规则化启发式把搜索空间剪到极小（近即时编译）但错过优化机会；对 GatedMLP 这类含动态门控/条件执行与多个 GEMM 的图无法全局最优。论文 Table IV 定性：Welder Action space 受限（Limited）、Policy space 全（Full）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Welder 开源（github.com/ucla-vast/welder，BSD 许可），输入 TensorIR/ONNX 等前端，核心是 tile-graph IR + 融合/调度 pass + CUDA codegen；默认配置按文档推荐（论文如此对比）。使用方式：作为论文 exploration-based 对比 baseline（与 TVM、Reasoning Compiler 同类）；其结果差主要来自"规则启发式剪枝过度"，反衬 LLM 先验引导的搜索在保持覆盖度的同时收敛更快。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
