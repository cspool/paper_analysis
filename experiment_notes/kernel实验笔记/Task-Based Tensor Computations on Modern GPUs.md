## Task-Based Tensor Computations on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Cypress编译器生成的warp-specialized CUDA kernel，在NVIDIA Hopper GPU上编排TMA（异步数据搬运）和Tensor Core（异步矩阵乘）之间的producer-consumer pipeline。编译器自动将用户的任务描述转换为warp-specialized代码：(1) DMA warp专门执行TMA异步copy（单线程调用TMA_load/TMA_store），通过completion barriers通知compute warpgroup数据就绪；(2) Compute warpgroup（128线程，4 warps）专门执行WGMMA指令驱动Tensor Core；(3) 软件pipeline（PIPE=3）使DMA warp预取PIPE步后的数据，隐藏global memory访问延迟；(4) Named barriers（prod/cons）管理DMA↔Compute之间的producer-consumer同步；(5) Backwards anti-dependency edges保证pipeline correctness（防止覆盖消费者尚未用完的buffer）；(6) 对于Flash Attention，编译器推断并插入TMA和Tensor Core之间的interleaved communication和synchronization。

  实验比较：(a) GEMM/Batched-GEMM vs cuBLAS（手写汇编/CUTLASS优化）和Triton；(b) Dual-GEMM（fused A·B₁+A·B₂）vs Triton；(c) GEMM+Reduction（fused GEMM + row-wise sum reduction）vs Triton；(d) Flash Attention 2 vs cuDNN/ThunderKittens/Triton；(e) Flash Attention 3 vs Flash Attention 3参考实现/cuDNN/ThunderKittens/Triton。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（Hopper架构，Tensor Cores支持wgmma 64×256×16 MMA指令、TMA支持异步burst copy和multicast、named barriers用于warp间同步、warpgroup概念——128线程协同启动Tensor Core操作）。CUDA 12.5.1（多数实验），Flash Attention实验部分系统用CUDA 12.3.1。Triton nightly 3.0.0.post20240716052845。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Cypress compiler生成的CUDA C++ kernel（warp-specialized, 包含TMA loads, WGMMA instructions, shared memory barriers）直接benchmark。对比的baseline系统：
  - cuBLAS/cuDNN: NVIDIA vendor libraries
  - CUTLASS: 开源模板库，参考实现在CUTLASS Hopper GEMM main loop (sm90_mma_tma_gmma_rs_warpspecialized.hpp)
  - ThunderKittens: 最新Hopper kernel库
  - Triton: 公开示例程序，部分kernel需手动修改启用实验性TMA操作
  - Flash Attention 3: 参考实现[37]

  修改：Cypress compiler生成代码使用CuTe dispatch到PTX WGMMA指令，kernel组织为DMA warp (TMA) + compute warpgroup (Tensor Core)的warp-specialized结构。Flash Attention 3中，用户重写main loop为pipelined方式后，Cypress编译器自动推断所有interleaved通信和同步（原Flash Attention 3需手动标注位置）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供Cypress开源链接。Cypress为Stanford/NVIDIA合作研究原型。

  评估原理：对每个kernel和问题size，100次迭代warmup 5次后取平均throughput (TFLOPs/s)。GEMM-like计算使用相同随机分布矩阵元素normalize power throttling影响。Triton kernel直接使用或适配公开示例程序。

  全过程（以H100 Hopper GEMM kernel执行为例）：
  ```
  输入：C=m×n矩阵，A=m×k矩阵，B=k×n矩阵 (FP16)

  Kernel: warp-specialized GEMM, grid=(ceil(m/T_M), ceil(n/T_N)), block=(128+32 threads, 1 DMA warp + 4 compute warps)

  SMEM: sA[T_M, T_K, 3], sB[T_K, T_N, 3], sC[T_M, T_N]
  Barriers: prod[3], cons[3], copyout

  ┌─ DMA Warp (32 threads, only thread 128 used for TMA) ───────────┐
  │ for k in range(0, K/T_K):                                       │
  │   if k >= PIPE: wait(cons[k % PIPE])    // wait consumer done   │
  │   if tid == 128:                                                │
  │     TMA_load(prod[k%PIPE],                                      │
  │       tile(gA, (blk_x, k)) → sA[:, :, k%PIPE],                 │
  │       tile(gB, (k, blk_y)) → sB[:, :, k%PIPE])                 │
  │                                                                  │
  │ wait(copyout)                                                   │
  │ if tid == 128:                                                  │
  │   TMA_store(sC → tile(gC, blk_x, blk_y))                       │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Compute Warpgroup (128 threads, 4 warps) ──────────────────────┐
  │ for k in range(0, K/T_K):                                       │
  │   wait(prod[k % PIPE])    // wait TMA完成数据加载               │
  │   warpgroup_sync()        // 128线程对齐                         │
  │   wgmma(accum, sA[:,:,k], sB[:,:,k])  // 异步发起Tensor Core    │
  │   warpgroup_wait()        // 等待Tensor Core完成                │
  │   arrive(cons[k % PIPE])  // notify DMA warp buffer可用         │
  │                                                                  │
  │ copy(accum, sC)           // 寄存器accum→shared memory staging  │
  │ syncthreads()                                                   │
  │ arrive(copyout)           // notify DMA warp可写出              │
  └──────────────────────────────────────────────────────────────────┘

  输出：GEMM throughput on H100
    - M=N=K=8192: ~980 TFLOPs/s (0.97x cuBLAS)
    - vs Triton: 1.05-1.11x speedup
    - Dual-GEMM: ~970 TFLOPs/s (与GEMM接近, vs Triton 1.36-1.40x)
    - GEMM+Reduction: 2.02-2.18x vs Triton (Triton未overlap GEMM与reduction,
      且heuristic将reduction accumulator放在SMEM而非register file)
  ```

  关键kernel设计要点：
  - DMA warp不参与compute——释放其registers给compute warpgroup存储更大accumulator
  - Pipelining (PIPE=3): DMA warp跑PIPE步领先compute warps，TMA延迟被完全隐藏
  - Backwards dependencies: DMA warp必须先等consumer用完buffer（cons barrier）才能写入新数据
  - Flash Attention 3 pipelining: 用户显式重写loop body做pipelining后，
    Cypress编译器自动推断所有interleaved TMA→Tensor Core同步和通信位置
  - 持久kernel优化未实现（影响小sequence length下Flash Attention 3的performance gap）
