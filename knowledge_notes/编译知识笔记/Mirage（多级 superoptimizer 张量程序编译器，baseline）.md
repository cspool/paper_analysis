## Mirage（多级 superoptimizer 张量程序编译器，baseline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mirage（UC Berkeley，OSDI'25，v0.2.4）是一个多级（block/thread 级）superoptimizer 深度学习编译器：用超优化（superoptimization，穷举式搜索函数等价改写）在模板库（CUTLASS）与手写 kernel 模板（CUDA/Triton）基础上探索张量程序的功能等价重写，能成功把多个算子融合成一个 kernel（如 GatedMLP 的全部算子），是图级融合方向 SOTA。QiMeng-Tensify（ISCA'26）把它作为模板类 baseline：其变换空间相对受限（GatedMLP <1024），因为 (1) 依赖模板库/手工代码模板（CUTLASS/CUDA/Triton），action space 手工设计；(2) 无法在 thread 级融合两个非 elementwise 算子（如两个 GEMM）。

从编译框架角度拆解术语，比如术语所在编译框架的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
作为 baseline 的运转流程（GatedMLP）：输入计算图 → 在 block 级用模板把 GEMM/SiLU/mul 融合成一个 kernel（空间 <1024，参数如 tile size 受限）→ 穷举式搜索模板组合 → 输出融合 kernel。局限：融合粒度到 block 级（不能 thread 级融合 GEMM+GEMM），调度参数空间手工设定；QiMeng-Tensify 案例中 Mirage 最优程序慢 1.47×。论文还指出 Mirage 直接执行有运行时错误（Table VIII 注），对比时用其 tuned 配置，故调优时间对比不可行（报告 up to 4 小时编译开销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Mirage 开源（github.com/ucbepic/mirage，MIT 许可），Python + C++，基于 cuDNN/CUTLASS 模板与手工 kernel 模板，Mirage DSL 描述算子、superoptimizer 搜索融合方案并生成 CUDA。使用方式：作为论文端到端 baseline（LLM 级也对比 Mirage，A100 上 QiMeng-Tensify 平均快 1.30×、H100 上 1.30×）；其"模板库/手工模板限定 action/policy space"正是 QiMeng-Tensify 用 MDP 无约束图变换 + LLM 先验要突破的点（Table IV：Mirage Action 手工设计、Policy 模板化）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
