## Dataflow-Scheduling Decoupling (数据流与调度解耦)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dataflow-Scheduling Decoupling 是 TileLang 提出的核心编程模型设计范式：将 AI kernel 编程中的"数据流"（dataflow — 数据在各级内存层次间如何流动、在何处执行何种计算）与"调度空间"（scheduling space — thread binding、memory layout、tensorization、pipeline）分离为两个正交维度。用户在 frontend 仅定义 dataflow（使用 tile operators: copy/gemm/reduce/atomic），而编译器自动探索和应用 scheduling 策略。当编译器默认优化不足时，用户可通过一组 scheduling annotations/primitives（T.Parallel, T.Pipelined, T.annotate_layout, T.use_swizzle）施加更精细的控制。这与 TVM 的 compute/schedule 分离有本质区别：TVM 要求用户显式编写 schedule 程序，而 TileLang 的 annotations 是声明式的可选提示，大部分 scheduling 由 Layout Inference 等自动化 pass 完成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

解耦在 TileLang 编译框架中的体现：
```
Dataflow 维度（用户显式编写）:
  T.copy(A[...], A_shared)    — "数据从 global memory 拷贝到 shared memory"
  T.gemm(A_shared, B_shared, C_local)  — "在 shared memory 数据上执行矩阵乘法"

Scheduling 维度（编译器自动推导 + 用户可选 annotations）:
  T.Pipelined(num_stages=2)   — 用户提示: 循环需 pipeline，编译器推导具体 schedule
  T.annotate_layout(...)       — 用户覆盖: 自定义 memory layout
  T.use_swizzle(10)            — 用户提示: 启用 swizzle 优化 L2 cache locality
  [自动] Layout Inference      — 编译器自动: 为 A_shared/B_shared 推断 SwizzleLayout
  [自动] Thread Binding        — 编译器自动: 将 block 级 register file 映射到 threads
  [自动] Pipeline Derivation   — 编译器自动: 生成 cp.async + commit + wait 序列
```

与 Triton 的对比：Triton 的 `num_stages` 参数是一个"调度 knob"但用户无法定义完全自定义的 pipeline。TileLang 的 T.Pipelined 既支持自动推导，也允许用户显式指定 producer/consumer order。Triton 的 `tl.dot` 自动选择 MMA layout 但用户无法覆盖为自定义 swizzle。TileLang 的 T.gemm 默认应用 SwizzleLayout（自动避免 bank conflict），但用户可通过 T.annotate_layout 覆盖。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

解耦的实现依赖于：(1) TileLang 的 Layout 和 Fragment 抽象——将 scheduling 决策统一编码为 composable layout functions (f: K^n → K^m)；(2) LayoutMap 优先级系统——Gemm(最高) > Element-wise > Copy(最低)，按优先级逐层推断；(3) 自动化 passes——Layout Inference、Pipeline Derivation、Thread Binding 作为编译器 passes 而非用户手写 schedule。用户使用方式：在 Python 中编写 dataflow 代码 → 可选添加 scheduling annotations → tilelang.compile() → kernel binary。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
