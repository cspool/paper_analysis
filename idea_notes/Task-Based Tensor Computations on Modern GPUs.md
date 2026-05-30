## Task-Based Tensor Computations on Modern GPUs

- baseline方法是什么？
  Baseline是GPU上现有的三种高性能编程范式和系统：(1) CUTLASS模板库——用户在C++模板层面手动管理数据移动（TMA调用）、同步（barriers）、warp specialization（DMA warp vs compute warps）、pipelining（shared memory buffers深度），能达峰值性能但显式管理复杂且易错；(2) ThunderKittens——提供更简洁的API包装Tensor Core和TMA操作，但仍需程序员显式管理同步和通信；(3) Triton——block-level DSL，用户仅描述thread block级别的计算，编译器自动做thread-level decomposition、memory allocation、scheduling——但对Hopper架构，Triton的heuristic决策常常次优（如不使用TMA默认、heuristic将reduction accumulator放在SMEM而非register file、不自动overlap独立操作），且用户无法干预这些性能关键决策。

  全栈执行例子（以H100 Hopper GEMM在cuBLAS/CUTLASS中的执行）：
  - 算法层：标准GEMM C=A·B，tile-based decomposition，每个SM负责T_M×T_N输出tile
  - 系统框架层：cuBLAS（vendor闭源库）或CUTLASS（开源模板库）。cuBLAS包含hand-tuned assembly实现；CUTLASS C++ templates参数化tile size、pipeline depth、warp specialization策略。
  - 编译框架层：标准CUDA C++ compiler (NVCC)，无特殊编译pass。所有优化由程序员在C++ template level手工表达。
  - kernel调度层：Warp-specialized GEMM（Figure 1b）——DMA warp（32 threads）专门通过TMA异步加载A/B tiles到shared memory（单线程调用TMA_load），通过prod barrier通知compute warpgroup；compute warpgroup（128 threads=4 warps）通过warpgroup_sync→wgmma→warpgroup_wait序列驱动Tensor Core做GEMM，完成后通过cons barrier通知DMA warp buffer可重用。Pipelining (PIPE>1)使DMA预取隐藏global memory latency。程序员必须手动：(a) 插入所有barrier同步（prod/cons/copyout），(b) 管理pipeline buffer indexing [k%PIPE]，(c) 保证write-after-read anti-dependency（backwards edge），(d) 分配shared memory staging buffer (sC)用于TMA store。
  - 硬件架构层：NVIDIA H100 GPU，TMA异步copy单元（支持multicast, shared memory barriers），Tensor Core（wgmma指令，128线程cooperative launch，操作数跨registers+shared memory），named barriers用于warp间同步。

  Baseline缺陷：
  1. **编程复杂性爆炸**：从Ampere到Hopper，GEMM kernel结构根本性改变——从bulk-synchronous（所有线程参与load+compute）变为warp-specialized（DMA warp专做TMA copy，compute warpgroup专做Tensor Core MMA）。程序员必须理解和管理异步fixed-function units之间的producer-consumer同步，正确性难以保证。
  2. **Triton的heuristic次优**：Triton将性能关键决策（数据放置、TMA使用、操作overlap）完全委托给编译器heuristic，无法被程序员控制。Dual-GEMM中Triton未overlap B₂加载与A·B₁计算；GEMM+Reduction中Triton将reduction accumulator放在SMEM而非register file，且未overlap GEMM与reduction。
  3. **修改程序易引入bug**：在CUTLASS/ThunderKittens等低层模型中，添加新功能需要修改partition、同步和通信代码——遗漏任何一处都导致数据竞争或死锁。
  4. **性能调优受限**：Triton不允许程序员干预tile size、memory placement、pipelining等决策；CUTLASS允许但需要大量代码修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Cypress——task-based编程模型和编译器，核心设计是分离Logical Description（顺序语义的task树）和Mapping Specification（task→processor, tensor→memory的绑定），通过编译器自动管理异步和同步。

  **解决缺陷1（异步编程复杂性）**：
  Cypress程序无显式数据移动和同步。程序员编写顺序语义的task描述（含prange/srange loop和launch语句），mapping specification声明性能关键决策。编译器通过dependence analysis自动插入copy-in/copy-out维持coherence，通过event-based IR编码和调度所有依赖关系，最终生成正确的warp-specialized CUDA C++。

  **解决缺陷2（Triton heuristic次优）**：
  Mapping specification允许程序员显式控制：(a) 每个task在哪个processor级别执行（HOST/BLOCK/WARPGROUP/WARP/THREAD），(b) 每个tensor在哪种memory中物化（GLOBAL/SHARED/REGISTER/NONE），(c) tunable参数（tile sizes, warpgroup count, pipeline depth），(d) warpspecialize和pipeline flag。GEMM+Reduction中用户将reduction accumulator放在Register（NONE+partitioned）而非SMEM，Cypress自动利用asynchrony overlap GEMM与reduction——达到2.02-2.18x vs Triton。Dual-GEMM中Cypress自动overlap B₂加载与A·B₁计算——达到1.36-1.40x vs Triton。

  **解决缺陷3（程序修改安全性）**：
  Cypress的sequential semantics保证：任何mapping specification下，编译器保证生成的并发执行与顺序执行等效。添加功能时，partitioning/communication/synchronization由编译器管理，不会引入数据竞争。例子：从单warpgroup GEMM扩展到多warpgroup仅需添加新task variant和调整mapping——现有代码不变。

  **解决缺陷4（性能调优）**：
  Mapping specification隔离所有性能参数——tile sizes, pipeline depth, memory placement, warp specialization——可在不改逻辑代码的情况下独立调整。论文发现通过调整mapping（3 consumer warpgroups替代2个），Flash Attention 2可达到接近FA3的性能。

  论文方法全栈执行例子（以H100 GEMM在Cypress中）：
  - 算法层：同一GEMM算法，通过7个task variants层次化分解到HOST→BLOCK→WARPGROUP→WARP→THREAD各级别。每个variant使用blocks/mma partition operators和prange/srange loops。
  - 系统框架层：Cypress Python embedded DSL编写program + mapping specification。Compiler输出CUDA C++直接编译执行。
  - 编译框架层（核心差异）：Cypress compiler完全自研，6个pass：
    (1) Dependence Analysis: 从entrypoint遍历task tree→插入copy-in/copy-out + event dependencies维持coherence
    (2) Vectorization: flatten implicit pfor loops (warp/thread级别), event arrays保留dependencies
    (3) Copy Elimination: 四类pattern消除冗余copy→同时消除/sync保留
    (4) Resource Allocation: interference graph→最小aliasing allocation
    (5) Warp Specialization: dependence graph partition (DMA vs compute warps) + pipelining
    (6) CUDA C++ Generation: event→sync lowering (barriers, syncwarps)
  - kernel调度层：Generated代码 = DMA warp (TMA async copy) + compute warpgroup (WGMMA) + named barriers + 3-deep pipeline。与baseline CUTLASS实现等效，但所有同步和数据移动由compiler自动生成——程序员在source code中zero lines处理同步。
  - 硬件架构层：同baseline H100。Cypress kernel的TMA/Tensor Core利用率与手写代码相当（0.88x-1.06x cuBLAS GEMM, 0.80x-0.98x FA3）。

  设计思路核心：
  Cypress的本质洞察是**异步GPU编程的两个关注点——算法逻辑（what to compute）和性能策略（how to map）——应该被分离**。算法逻辑用顺序语义的task描述表达（保证正确性、可修改性），性能策略用mapping specification表达（保留下放控制权）。编译器填补两者之间的鸿沟——自动推断parallelism、插入数据移动维持coherence、生成正确同步、优化冗余通信。这避免了CUTLASS的"程序员做所有决定"和Triton的"编译器做所有决定"两个极端，实现在正确性保证、性能控制和编程可用性之间的平衡。关键证据：Cypress的Flash Attention 3实现无需程序员标注任何同步位置——仅需将main loop重写为pipelined形式，编译器推断所有interleaved TMA/Tensor Core通信和同步。
