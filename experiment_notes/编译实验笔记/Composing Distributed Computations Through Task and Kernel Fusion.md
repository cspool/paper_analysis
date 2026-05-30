## Composing Distributed Computations Through Task and Kernel Fusion

- 属于编译框架的实现是什么？实验比较什么？
  Diffuse 是一个在分布式 task-based runtime 系统上动态执行 task fusion 和 kernel fusion 的编译系统。核心实现包括：(i) 一个 scale-free 中间表示（IR），抽象分布式数据（stores/partitions）和分布式计算（index tasks/point tasks），使 IR 大小与目标机器规模无关；(ii) 基于四个 fusion constraint（launch-domain-equivalence, true-dependence, anti-dependence, reduction）的分布式 task fusion 算法，通过贪心数据流分析在 task window 中寻找最大可融合前缀；(iii) 基于 MLIR 的 JIT 编译器，在融合后的 task 内部进行 kernel fusion（循环融合、临时数组消除、并行化）；(iv) 分布式临时存储消除（Definition 4 三约束）和基于 canonical De-Bruijn index 的 analyses memoization。
  实验比较：(1) Diffuse enabled vs Diffuse disabled 在 7 个应用的 weak-scaling throughput（Black-Scholes, Jacobi, CG, BiCGSTAB, GMG, CFD, TorchSWE）；(2) Diffuse vs 手工优化版本（CG hand-fused, TorchSWE numpy.vectorize）；(3) Diffuse vs MPI-based PETSc 库（CG 和 BiCGSTAB）。实验结果：平均 1.86× geo-mean speedup（0.93×–10.7× on up to 128 GPUs），对 PETSc 平均 1.4× geo-mean speedup，对手工优化代码平均 1.23× geo-mean speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 DGX SuperPOD 集群。每个节点 8 块 A100 80GB GPU（NVLink + NVSwitch 互联），双路 128 核 AMD 7742 Rome CPU，2TB 内存。节点间通过 InfiniBand（每节点 8 NICs）互联。最多扩展至 128 GPUs。每个实验 12 次 run，去掉最快和最慢后取剩余 10 次均值。Weak-scaling 实验中 warmup 迭代不计入 timing。

- 开源编译框架是什么。修改了什么。
  底层开源组件：
  - Legion runtime system（https://legion.stanford.edu/）：分布式 task-based runtime，提供 content-based coherence 和动态依赖分析。Diffuse 作为中间层位于高层库和 Legion 之间。
  - MLIR（https://mlir.llvm.org/）：JIT 编译器基础设施。Diffuse 使用 memref、affine、arith 等 dialect，利用 polyhedral optimization pass 进行循环融合和并行化。
  - cuPyNumeric（https://github.com/nv-legate/cupynumeric）：分布式 NumPy drop-in replacement，target Legion。
  - Legate Sparse：分布式稀疏计算库，target Legion。
  
  修改：(i) 修改 cuPyNumeric 和 Legate Sparse 的内部实现，使其动态生成 Diffuse IR（而非直接 target Legion）；(ii) 库开发者为每个操作注册 MLIR generator function（每个操作约 50–100 行 C++），返回描述 task 计算的 MLIR fragment；(iii) Diffuse 实现分布式 task fusion 算法（Section 4）、temporary store elimination（Section 5.1）、memoization（Section 5.2）和 JIT compilation pipeline（Section 6）。Diffuse 本身未明确声明独立开源仓库（ASPLOS '25），依赖组件均为开源。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  Diffuse 的编译框架使用流程（以 cuPyNumeric 5-point stencil 为例，Figure 1）：
  1. 输入：用户编写的 cuPyNumeric Python 程序（Figure 1a），通过 aliasing view 创建多视图分布式数组（center, north, east, west, south），循环内执行 5-point stencil（avg = center + north + east + west + south; work = 0.2 * avg; center = work）。
  2. 高层库分解：cuPyNumeric 将 NumPy 操作分解为 index task 序列，其中 ADD 操作被展开为 4 个 index task（分别计算 t1=center+north, t2=t1+east, t3=t2+west, avg=t3+south），每个 task 内部是 element-wise 的嵌套循环（Figure 1e）。同时为每个中间结果分配临时分布式数组（t1, t2, t3, avg, work）。
  3. Diffuse IR 生成：修改后的 cuPyNumeric 生成 Diffuse IR 的 task stream —— 每个 index task 携带 launch domain、参数 stores 及 partition（Tiling）和 privilege（R/W/Rd/RW）信息。IR 中 partition 的结构化表示（None 表示 replication，Tiling(shape, offset, proj) 表示 n 维仿射 tiling）使 IR 大小为 scale-free。
  4. 窗口缓冲与 fusion analysis：Diffuse 缓冲 task stream 到 window（stencil window size=5），运行 fusion constraints 数据流分析：(a) launch-domain-equivalence 验证所有 task launch domain 相同；(b) true-dependence 检查不存在 write 后通过不同 partition 的 read/write（关键：center/north/east/west/south 虽是 aliasing view 但因 partition 不同检查失败）；(c) anti-dependence 检查不存在 read 后通过不同 partition 的 write；(d) reduction 约束。由于 aliasing，COPY(work, center) 不能与读取 aliasing partition 的 ADD 融合，最终 4 个 ADD + MULT 融合为 FUSED_ADD_MULT，COPY 独立保留（Figure 1d）。
  5. Temporary store elimination：分析发现 avg 由融合前缀完全产生且不被后续 task 或应用引用（满足 Definition 4 三约束），将 avg 从分布式分配降级为 task-local allocation。work 因被 COPY 引用不能消除。
  6. Memoization：通过 canonical De-Bruijn index-like representation 检测 isomorphic task stream（Figure 7），复用之前的分析和编译代码。将 store 参数替换为 canonical 索引后比较 pattern。
  7. MLIR kernel 生成与 fusion（Figure 8）：调用各 task 的 MLIR generator 生成 MLIR fragment → 顺序组合为初始 fused kernel body → 消除 temporary 后生成显式 task-local allocation → polyhedral pass 融合循环 + 并行化 + 消除临时 allocation → 生成最优 single-pass kernel（一次遍历完成 5-way scaled add）。
  8. 代码生成：优化后的 MLIR lowered 为 CUDA kernel launches（GPU backend via MLIR GPU dialect）或 OpenMP parallel regions（CPU backend）。
  9. 输出：优化后的 task stream 转发给 Legion runtime 执行。在 A100 上实现 4× speedup（5-point stencil），Black-Scholes 实现 10.7× speedup（128 GPUs）。
