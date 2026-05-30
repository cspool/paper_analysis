## Kernel Fusion via MLIR (in Distributed Task-Based Runtimes)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Kernel Fusion via MLIR 是 Diffuse 在 task fusion 之后对 fused task 的 task body 进行的编译优化。核心流程：(1) 库开发者为每个操作注册 MLIR generator function，返回描述该 task 计算的 MLIR fragment（使用 memref + affine + arith dialect）；(2) 按程序顺序组合 fused task 中所有子 task 的 MLIR fragment；(3) 应用 polyhedral optimization passes 进行循环融合、临时分配消除、循环并行化；(4) 将优化后的 MLIR lowered 为 CUDA kernel launch 或 OpenMP parallel region。关键设计决策：将分布式推理（distributed dependence analysis）与循环优化（kernel fusion）分离——分布式分析在 Diffuse IR 层完成，kernel 优化在 MLIR 层完成，两者互不干扰。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

MLIR Kernel Fusion 的编译 pipeline（以 Black-Scholes 为例）：

```
Fused Task (67个element-wise操作融合):
  Task bodies: [MULT_body, ADD_body, SUB_body, DIV_body, ...]

Step 1: MLIR Fragment Generation (per original task)
  各 task 的 MLIR generator 生成:
    func.func @kernel(%a: memref<?xf64>, %b: memref<?xf64>, %c: memref<?xf64>) {
      %dim = memref.dim %c, 0
      affine.for %i = 0 to %dim {
        %0 = affine.load %a[%i]
        %1 = affine.load %b[%i]
        %2 = arith.addf %0, %1
        affine.store %2, %c[%i]
      }
    }

Step 2: Sequential Composition
  将 67 个 kernel body 顺序拼接为单一 func.func
  → 67 个独立 affine.for 循环，每个产生一个中间 memref

Step 3: Temporary Store Elimination
  将 temporary store 显式化为 task-local memref.alloca

Step 4: Polyhedral Passes (MLIR affine dialect)
  - Loop fusion: 67 个独立循环 → 1 个融合循环
  - Memref elimination: 消除所有中间 allocation
  - Parallelization: affine.for → affine.par

Step 5: MLIR Lowering
  affine + arith → GPU dialect (via MLIR GPU lowering passes)
  → CUDA kernel grid/block/thread configuration

Step 6: JIT Compilation → GPU Execution
  最终 kernel: 一 pass 完成 67 个 element-wise 操作
  每个 GPU thread: 加载一个 element → 67 次 compute in registers → 写入
```

生成器函数集成工作量：每个操作 50-100 行 C++ 代码，仅库开发者（非终端用户）需编写。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Diffuse 使用 MLIR 的 memref dialect（比 raw pointer 提供更强的 aliasing 保证）、affine dialect（polyhedral compilation 的 target）、arith dialect（算术操作）。Polyhedral optimization passes 用于 fusion + parallelization。也可利用 MLIR 中的 domain-specific kernel fusion 技术（linalg/sparse_tensor dialect for dense/sparse tensor algebra）。最终 lowered 到 GPU (CUDA) 或 CPU (OpenMP)。该设计使 Diffuse 的 kernel fusion 与具体 domain 解耦——任何能生成 MLIR fragment 的库操作都能受益于 fusion。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
