## Task-Based Tensor Computations on Modern GPUs

- 属于编译框架的实现是什么？实验比较什么？
  实现是Cypress——一个task-based编程模型和编译器，针对现代GPU上异步固定功能单元（Tensor Cores和TMA）的编程。Cypress编译器将逻辑计算描述（task-based, sequential semantics, 含prange/srange并行/串行task launch、blocks/mma partition operators）和mapping specification（指定tasks在哪个processor级别执行、tensors在哪种memory物化、tunable值）合并，通过dependence analysis→vectorization→copy elimination→resource allocation→warp specialization→CUDA C++ generation六个pass生成warp-specialized CUDA C++代码。编译器基于MLIR实现，生成代码使用CuTe layout algebra处理数据布局变换。

  实验比较了Cypress生成的kernel vs cuBLAS（GEMM/Batched-GEMM，FP16）、cuDNN（Flash Attention）、CUTLASS/ThunderKittens参考实现（Flash Attention 2/3）、Triton nightly（GEMM/Dual-GEMM/GEMM+Reduction/Flash Attention）、Flash Attention 3参考实现。评估六个kernel：标准FP16 GEMM、Batched-GEMM、Dual-GEMM（Gated Linear Units核心计算A·B₁+A·B₂）、GEMM+Reduction（fused C=A·B + y(i)=ΣA(i,k)）、Flash Attention 2、Flash Attention 3。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（Hopper架构，Tensor Cores支持wgmma指令、TMA异步数据搬运、shared memory barriers、warpgroup概念——128线程协同发起Tensor Core操作）。CUDA 12.5.1用于大多数实验；Flash Attention实验各系统取最优CUDA版本（Cypress/ThunderKittens用CUDA 12.5.1，Flash Attention 3/cuDNN用CUDA 12.3.1，Cypress FA3用最新NVCC build）。Triton nightly 3.0.0.post20240716052845。所有结果100次迭代+5次预热取平均。

- 开源编译框架是什么。修改了什么。
  Cypress是自研编译器原型，论文未提供具体开源链接。编译器基于MLIR实现source-to-source translation。未修改现有编译框架而是完全自研：包含event-based IR（event类型为unit或processor-annotated event arrays）、dependence analysis with copy-in/copy-out discipline、vectorization leveraging indexable event arrays、四类copy elimination patterns、基于interference graph的resource allocation、graph partitioning的warp specialization、以及pipelining transformation。Leaf task内可调用任意CUDA C++函数。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  论文为Stanford/NVIDIA合作研究（PLDI 2025），未明确提供开源链接。Cypress为研究原型。

  作用：在CUTLASS（手动管理通信/同步）和Triton（自动但heuristic决策可能次优）之间取中间地带——自动化数据移动和同步管理，但保留用户对性能关键决策的控制。

  全过程（以H100 FP16 GEMM为例）：
  ```
  输入：Cypress程序 = Logical Description + Mapping Specification

  Logical Description:
    7个task variants (gemm_host/gemm_block/gemm_tile/gemm_inner/gemm_thread + clear/copy子树)
    每个variant对应不同processor级别，使用blocks partition和mma partition分解数据和计算
    gemm_host: partition C→U×V tiles, prange over tiles → launch gemm_block per tile
    gemm_block: 创建accumulator, srange over K-reduction → launch gemm subtasks → launch copy
    所有task launch无任何显式同步或数据移动

  Mapping Specification:
    6个TaskMapping entries:
    gemm_host@HOST: mems=[GLOBAL,GLOBAL,GLOBAL], U=256,V=256
    gemm_block@BLOCK: mems=[GLOBAL,GLOBAL,GLOBAL], W=64, warpspecialize=True, pipeline=3
    gemm_tile@BLOCK: mems=[NONE,SHARED,SHARED], WGS=2 (2 warpgroups)
    gemm_warpgroup@WARPGROUP: mems=[NONE,SHARED,SHARED], PIECES=4, PROC=WARP
    gemm_warp@WARP: mems=[NONE,SHARED,SHARED], PIECES=32, PROC=THREAD
    gemm_thread@THREAD: mems=[REGISTER,SHARED,SHARED]

  Step 1 — Dependence Analysis:
    In-order遍历task tree，对每个task launch:
    (1) 创建fresh tensor allocation (callee memory)
    (2) Read tensors: copy-in with event preconditions
    (3) Recursively traverse callee → generate IR
    (4) Written tensors: copy-out (callee completion event←precondition)
    prange→pfor (event数组标注processor kind), srange→for
    broadcast indexing [:]表示所有并行迭代完成

  Step 2 — Vectorization:
    从内层flatten pfor loops, 替换iteration variable为processor index
    Event数组增加维度, consumers重写为point-wise或broadcast索引

  Step 3 — Copy Elimination:
    Spill elimination (删copy partition↔copy back, 消除sync)
    Spill hoisting (循环内copy hoist到pre/postamble)
    Duplicate/Self copy elimination (删重复copy, 保留依赖)

  Step 4 — Resource Allocation:
    构建SMEM tensors complete interference graph→删除辅助边至allocation可行
    最小化aliasing, 插入last reader→first writer event依赖

  Step 5 — Warp Specialization:
    Dependence graph partition: DMA warp←global→shared copies, compute warps←其余
    跨partition边→shared memory barriers
    Pipelining: unroll PIPE次→compact回单迭代, 按[k%PIPE]索引

  Step 6 — CUDA C++ Generation:
    THREAD broadcast→__syncwarp
    WARP/WARPGROUP broadcast→named barrier arrive/wait
    TMA events→SMEM barriers (TMA自动触发)
    Tensor Core events→warpgroup sync assembly
    Point-wise indexing→消除（SSA保证）

  输出：warp-specialized CUDA C++
    DMA warp: TMA_load (单线程) → prod barrier
    Compute warpgroup: 128线程 warpgroup_sync → wgmma → warpgroup_wait → arrive cons barrier
    3-deep pipeline: SMEM buffers [T_M, T_K, 3]
    GEMM 0.88x-1.06x cuBLAS, FA3 0.80x-0.98x best-known
    vs Triton: GEMM 1.05-1.11x, Dual-GEMM 1.36-1.40x, GEMM+Reduction 2.02-2.18x
  ```
