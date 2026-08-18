## MLIR 与 Affine Dialect（含 Polygeist / Torch-MLIR 前端）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MLIR 是 LLVM 项目下的多级中间表示基础设施：编译器把程序表示为多层 dialect 的混合（高层算子 dialect → 循环/缓冲 dialect → 低层），每层做针对性优化。CODO 构建于 MLIR 之上，主表示用 affine dialect——表达常量界嵌套循环 + 仿射数组访问（affine.for/affine.load/affine.store，索引为循环变量与符号的仿射函数），是循环变换（tiling/permutation/fusion）的理想载体。CODO 面向常量循环界的 affine 程序，覆盖卷积、注意力、ReLU/GeLU、矩阵乘、点积等。
- 两个前端：(1) Polygeist（PACT'21）——基于 Clang 的 C/C++→MLIR 前端，cgeist 把 C/C++ 先降到 scf dialect，再经 raising pass 把可证明合法的 load/store 升为 affine.load/store、循环升为 affine.for、条件升为 affine.if，指针/数组映射为 memref、索引数学折叠进 affine_map（Web 证据：polygeist.llvm.org、llvm/Polygeist）；(2) Torch-MLIR——llvm/torch-mlir 把 PyTorch 程序（torch dialect，自动由 PyTorch JIT IR 算子注册表生成）经 TorchToLinalg conversion 降到 linalg-on-tensors，再用 bufferization（tensor→memref）与 linalg→affine 转换降到 affine（Web 证据：llvm/torch-mlir docs/architecture.md）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CODO 编译流水（Fig. 3）：C/C++ kernel → Polygeist → affine；或 PyTorch 模型 → Torch-MLIR → linalg（元素级算子融合、tensor bufferization）→ affine。此后全部 CODO pass 在 affine dialect 上做：粗/细粒度违例消除 → 缓冲确定 → reuse buffer → 片外传输 → 自动调度，最后经扩展 HLS dialect 降级为 host code + HLS C++ kernel。
- MLIR 基础设施红利（§VII-B）：所有优化为模块化 MLIR pass；每 pass 后跑内建 verification（dominance、SSA、类型）+ canonicalize/cse 清理死代码/冗余计算，IR 合法性由框架保证。CODO 还引入专用 FIFO/ping-pong 数据类型与操作建模通信缓冲，并把 dataflow pragma/array partition 语义放进扩展 HLS dialect（源自 HIDA）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：codo-opt 单命令驱动全流程；开发者写 C++ kernel 或导 PyTorch 模型即可。Polygeist 对"数学风格"代码（PolyBench 类）支持最好（PACT'21 基准）；Torch-MLIR 的 linalg 后端经 dialect conversion 框架支持动态形状。CODO 依赖 LLVM/MLIR 工具链 + Polygeist + Torch-MLIR（官方 README 要求先构建 torch-mlir）。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
