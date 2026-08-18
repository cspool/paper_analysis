## 图编译器与后端编译器（Graph Compiler / Backend Compiler，M100）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- M100 编译器工具链的后续阶段：① graph compiler（图编译器）——做图优化（tensor fusion、死代码消除、代数化简、layout 变换）与动态 tensor 的动态内存分配；② backend compiler（后端编译器）——C 扩展编译器，生成利用 M100 硬件能力的 intrinsic 指令（tensor 计算、数据搬运、同步）。两者在 space-time scheduler 之后、NPU 固件 JIT 之前。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：space-time scheduler 输出的映射 → 图编译器优化（算子融合减少中间 tensor 搬运、layout 变换匹配 TWU 访问模式、动态 shape 的内存分配）→ 后端编译器把算子 lower 为 intrinsic（对应 TCU 的 GEMM/卷积、CVU 的向量算子、DTDU 的搬运/转置、SU 的同步操作）→ 生成二进制代码 → NPU 固件 JIT 再翻译为最终 TPB 指令。作用：把高层算子语义精确映射到 TPB 指令（含 shape/通信需求元数据）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：自研编译器栈（类 LLVM 的 C 扩展 + intrinsic 生成）；图优化与主流 DL 编译器（TVM/MLIR 的 fusion/DCE/layout pass）思路一致但针对 M100 指令集。使用：模型端到端经工具链编译后部署；是垂直集成方案的一部分（runtime/driver/firmware 配套）。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
