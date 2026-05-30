## Cypress (Task-Based Programming Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cypress是Stanford/NVIDIA提出的task-based编程模型和编译器（PLDI 2025），用于简化现代GPU上异步固定功能单元（Tensor Cores和TMA）的编程。Cypress的核心创新是将GPU程序分离为两个独立组件：(1) Logical Description——用顺序语义的tasks和tensors描述算法逻辑，无显式通信或同步；(2) Mapping Specification——声明tasks在哪个processor级别执行、tensors在哪种memory中物化、tunable参数值。Cypress编译器将两者合并，通过dependence analysis→vectorization→copy elimination→resource allocation→warp specialization→CUDA C++ generation六个pass自动生成warp-specialized CUDA代码。

Cypress在编程抽象谱系中位于CUTLASS（手动管理所有低层细节）和Triton（编译器做所有heuristic决策）之间——自动化通信/同步管理，但保留用户对性能关键决策的控制。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Cypress编译器处理Hopper GEMM的完整流程：

```
输入：
  Logical Description (Python DSL):
    - gemm_host: prange over output tiles → launch gemm_block
    - gemm_block: srange over K-dimension → launch gemm sub-tasks
    - gemm_inner: partition by mma → prange over pieces → launch gemm_thread
    - gemm_thread (leaf): CuTe_warpgroup_gemm(WGMMA_64x256x16, ...)
    
  Mapping Specification (JSON-like):
    gemm_host@HOST:     mems=[GLOBAL,GLOBAL,GLOBAL], U=256,V=256
    gemm_block@BLOCK:   mems=[GLOBAL,GLOBAL,GLOBAL], W=64, warpspec=T, pipe=3
    gemm_tile@BLOCK:    mems=[NONE,SHARED,SHARED], WGS=2
    gemm_warpgroup@WG:  mems=[NONE,SHARED,SHARED], PIECES=4,PROC=WARP
    gemm_warp@WARP:     mems=[NONE,SHARED,SHARED], PIECES=32,PROC=THREAD
    gemm_thread@THREAD: mems=[REGISTER,SHARED,SHARED]

Pass 1 — Dependence Analysis:
  遍历task tree → 为每个sub-task launch:
    (a) 在callee's memory创建fresh tensor allocation
    (b) 对read tensors: copy-in + event preconditions
    (c) 递归traverse callee → generate IR
    (d) 对written tensors: copy-out + callee completion event
  输出: event-based dependence graph (Figure 8b)

Pass 2 — Vectorization:
  从内到外flatten pfor implicit loops
  替换iteration variable → processor index (thread_id(), warp_id())
  Event数组增加维度，consumers重写索引

Pass 3 — Copy Elimination:
  Spill elimination: 删partition copy→copy back
  Spill hoisting: 循环内copy hoist到pre/postamble
  Duplicate elimination: 删同一tensor重复copy
  Self copy elimination: 删同一allocation的copy

Pass 4 — Resource Allocation:
  构建SMEM tensors interference graph
  迭代删边→找到最小aliasing feasible allocation
  插入last reader→first writer event依赖

Pass 5 — Warp Specialization:
  Dependence graph partition: DMA warp←copies, compute wg←rest
  Pipelining: unroll PIPE次→compact回单迭代
  Backwards anti-dependencies自动插入

Pass 6 — CUDA C++ Generation:
  Event arrays→硬件同步: THREAD broadcast→__syncwarp,
  WARP/WG broadcast→named barriers, point-wise→移除
  TMA events→mbarrier arrivals, Tensor Core events→warpgroup sync

输出: warp-specialized CUDA C++ (DMA warp + compute warpgroup + barriers + TMA + WGMMA)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Cypress为研究原型（Stanford/NVIDIA合作，PLDI 2025），论文未提供开源链接。实现细节：
- 编译器基于MLIR实现，source-to-source translation
- Python embedded DSL作为前端
- CuTe用于数据布局变换和PTX指令dispatch
- 生成的CUDA C++可直接通过NVCC编译
- 评估显示GEMM达0.88x-1.06x cuBLAS性能，Flash Attention 3达0.80x-0.98x best-known实现

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
