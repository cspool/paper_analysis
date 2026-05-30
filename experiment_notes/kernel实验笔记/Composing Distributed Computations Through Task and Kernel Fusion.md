## Composing Distributed Computations Through Task and Kernel Fusion

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Diffuse 的 kernel fusion 部分属于 kernel 调度/运行时计算优化。实现包括：(i) 融合 task 内部的 MLIR-based kernel 生成与 fusion —— 将顺序的多个 task body 组合为单个 fused kernel，通过 polyhedral 优化融合嵌套循环、消除临时分配（memref dialect allocation）、并行化循环；(ii) 将优化后的 MLIR lowered 为 GPU kernel launch（via MLIR GPU dialect）或 CPU OpenMP parallel region。Kernel fusion 使得原本需要多次 pass over data 的多个独立 kernel 合并为一次 pass，提升 arithmetic intensity 并大幅减少 memory traffic。
  实验比较：Diffuse fused vs unfused 在 7 个应用的 weak-scaling throughput。额外比较 vs MPI-based PETSc 和 vs 手工优化版本。特别关注 kernel fusion 的效果——论文明确指出"task fusion alone can only reduce runtime overhead... did not yield speedups"，kernel fusion 才是加速的主要来源。

- 后端平台是什么，配置是什么。
  NVIDIA A100 DGX SuperPOD：每节点 8×A100 80GB（NVLink+NVSwitch），双路 128 核 AMD 7742 Rome，2TB 内存，InfiniBand 互联（每节点 8 NICs），最多 128 GPUs。

- 评估性能的软件/脚本是什么。修改了什么。
  评估应用（Figure 9）：Black-Scholes（67 data-parallel 操作，全部 fusible），Dense Jacobi Iteration（密集矩阵-向量乘 + 2 个 fusible 向量操作），Sparse Krylov Solvers CG 和 BiCGSTAB（cuPyNumeric/Legate Sparse 实现），Geometric Multi-Grid GMG（CG-based + V-cycle preconditioner），Computational Fluid Dynamics CFD（Navier-Stokes 2D channel flow），Shallow Water Equation TorchSWE（浅水方程求解器）。
  修改：无需修改用户应用代码。库开发者需为每类操作注册 MLIR generator function（每个操作约 50–100 行 C++ 代码）。Diffuse 自动选择 window size（通过逐步增大 window size 直到所有 task 都被融合的过程）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  Diffuse 的 kernel fusion 编译 pipeline 在 CPU 上运行（JIT compilation），生成优化后的 MLIR module，通过 MLIR GPU lowering pass 生成 CUDA kernel，在 A100 GPU 上执行。评估通过 12 次 run 取平均（排除最快和最慢），warmup 迭代不计入 steady-state timing。

  Kernel fusion 评估原理（以 Black-Scholes 为例，67 个 data-parallel 操作，window size=70）：
  1. 输入：cuPyNumeric Black-Scholes 应用生成 67 个 element-wise index task。每个 task body 由 MLIR generator 生成 fragment（Figure 8a 示例：memref<?xf64> 参数 + affine.for 循环 + arith.addf/mulf 计算）。
  2. Task fusion analysis：所有 67 个 task 满足全部 4 个 fusion constraints（相同 launch domain，相同 partition，无 aliasing，无 reduction conflict），全部融合为单个 index task。
  3. MLIR kernel 组合与优化（Figure 8 pipeline）：67 个 task body 顺序组合为初始 fused kernel → temporary store elimination 消除 64 个中间 store（降级为 task-local memref.alloc）→ polyhedral 循环融合将 67 个独立 affine.for 合并为单个 affine.par 循环 → memref.alloca 消除（因 temporary 被 inlined）→ 生成最优 single-pass kernel。
  4. GPU lowering：MLIR affine → GPU dialect lowering pass，将 affine.par 映射到 CUDA grid/block/thread launch configuration。
  5. 执行：在 A100 上以单 CUDA kernel 一次 pass 完成全部 67 个 element-wise 操作的计算。Arithmetic intensity 从 67 次 HBM read/write per element 降至 1 次 read + 1 次 write per element。
  6. 输出：128 GPUs 实现 10.7× speedup vs unfused。Unfused 版本每迭代 67 个 task → fused 后仅 1 个 task（Figure 9: Tasks per Iteration 67→1）。

  Compilation overhead 评估（Figure 13）：测量 warmup 时间（含 compilation）vs unfused warmup。Black-Scholes 编译 0.06s vs unfused 0.38s（更快因为只需编译 1 个 kernel），其他应用需 25–119 次迭代 amortize 编译开销。论文认为这些开销合理，因为科学计算应用通常运行数千至数百万次迭代。
