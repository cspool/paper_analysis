## MLIR Polyhedral Loop Fusion for Distributed Task Kernels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MLIR Polyhedral Loop Fusion for Distributed Task Kernels 是 Diffuse 在分布式 task fusion 后对 fused task 的 task body 进行的编译优化。核心流程：(1) 库开发者为每个操作注册 MLIR generator function（使用 memref + affine + arith dialect），返回描述该 task 计算的 MLIR fragment；(2) 按程序顺序组合 fused task 中所有子 task 的 MLIR fragment；(3) 应用 polyhedral optimization passes 进行循环融合、临时分配消除、循环并行化；(4) 将优化后的 MLIR lowered 为 CUDA kernel launch 或 OpenMP parallel region。与直接在 distributed context 中做 loop optimization 的难点不同，Diffuse 将分布式推理与 kernel 优化分离——分布式分析在 Diffuse IR 层完成，kernel 优化在 MLIR 层完成。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLIR Kernel Fusion 的编译 pipeline（以 Black-Scholes 67 个 element-wise 操作融合为例）：

```
输入: 67 个 MLIR fragment (每个一个 element-wise kernel)
  例: ADD kernel:
    affine.for %i = 0 to %dim {
      %0 = affine.load %a[%i]
      %1 = affine.load %b[%i]
      %2 = arith.addf %0, %1
      affine.store %2, %c[%i]
    }

Step 1: Sequential Composition → 67 个独立 affine.for 循环
Step 2: Temporary Store Elimination → 中间 store 降级为 task-local memref.alloca
Step 3: Polyhedral Loop Fusion → 67 个循环融合为 1 个:
    affine.par %i = 0 to %dim {
      %0 = affine.load %a[%i]
      %1 = affine.load %b[%i]
      %2 = arith.addf %0, %1
      %3 = affine.load %d[%i]
      %4 = arith.subf %2, %3
      // ... 继续 63 个操作
      affine.store %result, %out[%i]
    }
Step 4: GPU Lowering → CUDA kernel launch (via MLIR GPU dialect)
```

执行语义：每个 GPU thread 对应一个 element index，在单一 kernel invocation 中完成全部 67 个操作。中间结果保持在 register 中，无需 HBM round-trip。Black-Scholes: 67 kernel → 1 kernel, 67 HBM pass → 1 HBM pass → 10.7× speedup (128 GPUs)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Diffuse 使用 MLIR 的 memref dialect (比 raw pointer 更强的 aliasing 保证)、affine dialect (polyhedral compilation target)、arith dialect (算术操作)。Polyhedral passes 通过 MLIR 的 affine dialect optimization pipeline 执行。每个库操作需注册 MLIR generator function (50–100 行 C++ 代码)，仅库开发者需编写。优化后 kernel 通过 MLIR GPU lowering passes 生成 CUDA binary，或在 CPU backend 生成 OpenMP code。Compilation 结果通过 memoization（基于 canonical De-Bruijn index）缓存，在循环中的重复 task stream pattern 直接命中 cache。Compilation overhead 需要 25–119 次迭代 amortize（Figure 13）。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
