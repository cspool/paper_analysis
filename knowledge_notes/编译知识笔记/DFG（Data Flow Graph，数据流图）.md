## DFG（Data Flow Graph，数据流图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DFG 是把计算表示成有向无环图（节点=算子/运算，边=数据依赖）的中间表示（IR）：每个节点在操作数就绪时执行（数据流语义），天然暴露并行度与数据依赖，是 CGRA/数据流架构编译的通用 IR。程序（循环 kernel）→ 控制数据流图（CDFG）→ 划分基本块/子图 → 每个子图编译为纯数据依赖的 DFG → 映射（place & route）到 PE 阵列。
- 在 LoRA（ISCA'26）中的作用：LLVM 前端对 #pragma 标注的 loop kernel 做循环分析、优化与 DFG 生成；非线性函数定义为自定义函数（__CGRA__HARDWARE_OP 标注）被替换成 tanh_XCore 等自定义节点进入 DFG。关键收益：复合函数（tanh(x)+1、sin(x)+cos(x)、ln(sin(x))）作为一个 XCore DFG 节点，使 DFG 显著缩小——Swiglu 从 PICACHU 的 37 节点降到 LoRA 的 15 节点（含 1 个 XCore）、Mish 从 36→11、DCT 从 33→21，释放 PE 资源供 loop unrolling 并减少映射压力。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（LoRA 软件工具链 Fig.6）：C 源码（#pragma 标注 kernel 与自定义非线性函数）→ LLVM 前端：循环分析/优化 → 手工分析（确定每非线性函数的类型、输入范围、数据格式）→ 添加自定义函数（__CGRA__HARDWARE_OP 替换原非线性函数为 tanh_XCore 节点）→ DFG 生成（非线性节点作为特殊节点）→ 后端：收集 XCore 配置（由 Chebyshev 算法生成的多项式参数）→ 模拟退火空间映射 + 内存分区算法 → 生成 CGRA calling function（RoCC 指令序列）→ 替换原 loop 编译为 bare-metal。
- 例：Swiglu kernel 的 DFG 在 LoRA 中为 15 节点（#XCore=1）13-17 边，PICACHU 为 37 节点/47 边（Table IV）；XCore 节点由 LNS 单元完成 6 项多项式（7 cycle），其余节点为普通 PE 运算。
- 其它用法（vault 中广泛出现）：DICE 把程序划分成 p-graph 映射 CGRA（每 PE 固定操作、数据流直传）；ChituDiffusion 用 DFG 作为通用 IR 表达任意 diffusion pipeline；Welder 用 tile-graph 调度深度学习访存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 App-Compiler（LLVM 前端，开源于 COFFA 仓库 LoRA-ISCA-AE）做 DFG 生成；自定义函数机制保证非线性节点在 DFG 阶段被特殊处理（配置由 PiecewiseChebFitter 生成并写入 XCore LUT）。DICE/其它 CGRA 工具链用类似流程（LLVM/MLIR → DFG/CDFG → 映射）。
- 使用场景：CGRA/FPGA 数据流加速器的编译前端、图编译器的 IR 抽象（区分于 SSA/控制流图：DFG 只含数据依赖、适合空间映射）。局限：分支/不规则控制流需 CDFG 或 predication（LoRA 用 partial-prediction + IOB 处理非仿射访问 A[B[i]]、A[i*i] 的运行时地址）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
