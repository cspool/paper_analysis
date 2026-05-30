## MLIR (Multi-Level Intermediate Representation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MLIR (Multi-Level Intermediate Representation) 是 LLVM 项目的编译器基础设施，旨在提供可扩展的、支持多层抽象的编译器框架。其核心概念是 dialect：每个 dialect 定义一组具有特定语义的 operations 和 types。一个 MLIR 程序可同时包含多个 dialect，编译器通过 progressive lowering（将高层 dialect 逐步转换/lowered 到低层 dialect）实现优化和代码生成。MLIR 提供了核心基础设施：operation/type/attribute 定义、pass management、dialect conversion、pattern rewriting 等。关键 dialect 包括：memref（结构化内存引用，比 raw pointer 提供更强的 aliasing 保证）、affine（polyhedral loop analysis 的 target）、linalg（线性代数操作）、sparse_tensor（稀疏张量）、gpu（GPU kernel launch）、scf（结构化控制流）、arith（算术运算）等。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

MLIR 在 Diffuse 中的使用方式：

```
MLIR Generator Functions (库开发者编写):
  每个操作 (ADD, MULT, COPY, ...) → MLIR fragment in memref+affine+arith dialect

                            ↓
                      
MLIR Compilation Pipeline (Diffuse 自动执行):
  1. Inline all fragments into single MLIR module
  2. affine-loop-fusion: fuse independent loops
  3. affine-scalar-replacement: eliminate temporary allocations  
  4. affine-parallelize: convert affine.for → affine.par
  5. lower-affine: affine → scf + arith
  6. convert-scf-to-cf: structured control flow → CFG
  7. gpu-kernel-outlining: extract GPU kernel
  8. gpu-map-parallel-loops: map to GPU thread hierarchy
  9. convert-to-nvvm / convert-to-llvm: final lowering

                            ↓
                      
CUDA Binary / CPU Binary (JIT compiled, cached via memoization)
```

MLIR 的核心价值在 Diffuse 中：(1) extensibility——通过 dialect 同时表达高层计算和低层优化；(2) 复用社区 passes（affine optimization, GPU lowering）；(3) 类型安全——memref 提供比 raw pointer 更强的 aliasing 保证，使编译器可安全地进行 aggressive optimization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLIR 开源地址：https://github.com/llvm/llvm-project/tree/main/mlir。使用方式：包含 MLIR 头文件并在 CMake 中链接 MLIR 库。在 Diffuse 中，库开发者通过注册 generator function 使用 MLIR（返回用 C++ MLIR API 构建的 MLIR fragment），终端用户不可见。每个 generator function 创建 MLIR Context、构建 ModuleOp、FuncOp、affine.for 等 operations。MLIR fragment 可灵活使用不同的 dialect——当前 cuPyNumeric 使用 memref+affine+arith，未来可扩展到 linalg+dense/sparse tensor。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
