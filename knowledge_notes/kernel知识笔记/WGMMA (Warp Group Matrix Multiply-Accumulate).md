## WGMMA (Warp Group Matrix Multiply-Accumulate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WGMMA（Warp Group Matrix Multiply-Accumulate）是NVIDIA Hopper架构的Tensor Core PTX指令集，由warpgroup（128线程）协同发起矩阵乘法累加操作。相比Ampere架构的wmma指令（由单个warp的32线程发起），WGMMA代表了Tensor Core编程的根本性变革：(1) 更大的tile尺寸——WGMMA支持如64×256×16、64×128×16等更大MMA shape（vs Ampere的16×16×16），提供更高的算术强度；(2) 操作数来源——操作数A/B可来自线程寄存器或shared memory的任意组合（寄存器-寄存器、寄存器-SMEM、SMEM-寄存器），由指令变体决定；(3) 异步执行——WGMMA是异步指令，发出后线程可继续执行其他工作（如准备下一tile的操作数），通过`wgmma.wait_group`等待完成；(4) 复杂的数据分布——输出矩阵C和操作数A/B按照硬件规定的swizzle pattern分布在128线程的寄存器和shared memory中（见Cypress论文Figure 4的64×N×8 partition pattern）。

Cypress论文的gemm_thread leaf task直接使用WGMMA：
```
CuTe_warpgroup_gemm(WGMMA_64x256x16(), C, A, B)
```
其中WGMMA_64x256x16()是64×256×16的MMA shape参数化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WGMMA在Hopper GEMM kernel中的调度伪代码（compute warpgroup部分）：

```
// Compute warpgroup: 128 threads, 4 warps
// 输入: accum[8] (per-thread accumulator registers, swizzled layout)
//       sA[T_M, T_K, PIPE] (shared memory, 3-deep pipeline)
//       sB[T_K, T_N, PIPE]

pipeline_depth = 3
for k = 0 to K/T_K - 1:
    wait(prod[k % pipeline_depth])      // 1. 等待TMA完成本iteration的数据加载
    
    warpgroup_sync()                     // 2. 128线程对齐
    
    wgmma.fence                         // 3. 确保A/B操作数就绪
    wgmma(accum,                        // 4. 异步发起Tensor Core计算
          sA[:, :, k % pipeline_depth],  //   操作数A来自SMEM
          sB[:, :, k % pipeline_depth])  //   操作数B来自SMEM
    // 线程可继续其他工作（如address计算等）
    
    warpgroup_wait()                     // 5. 等待Tensor Core完成
    // accumulator已更新：C += A_tile * B_tile
    
    arrive(cons[k % pipeline_depth])    // 6. 通知DMA warp buffer可重用

// 最终：warpgroup将accumulator从寄存器写回shared memory staging buffer
//       DMA warp通过TMA_store将结果写入global memory
```

WGMMA指令的关键参数：
- m64nNk16/m64nNk32: m=64固定（warpgroup的行粒度），N可变，k=16或32
- operand A来源：register或shared memory
- operand B来源：register或shared memory  
- 输出accumulator：始终在warpgroup各线程的寄存器中（不可直接写shared memory或global memory）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 直接编写PTX汇编（`wgmma.fence.sync.aligned`, `wgmma.mma_async.sync.aligned`等）
- CUTLASS/CuTe封装：`cute::gemm`或`cute::gemm_rs`（register-smem variant）
- ThunderKittens提供高级C++封装
- 需要与thread block配置匹配——block size必须为warpgroup size的倍数（≥128）
- WGMMA的使用通常与warp specialization协同——DMA warp管理TMA传输，compute warpgroup专注于WGMMA

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
