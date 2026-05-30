## Warp Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warp Specialization是一种GPU编程技术，将thread block内的不同warps分配为不同的角色（如数据搬运、计算、输出写入），利用GPU warp调度器的细粒度warp切换实现不同角色之间的overlap执行。在Hopper架构上，warp specialization是实现高性能的必需手段——因为TMA和Tensor Core都是异步固定功能单元，需要不同的warp专门管理各自的操作以最大化硬件利用率。

Cypress论文详细描述了Hopper上的warp specialization模式：(1) DMA Warp——1个warp（32线程），专门执行TMA异步数据搬运（实际仅thread 0调用TMA指令），其余线程可能执行地址计算或闲置；(2) Compute Warpgroup——4个warp（128线程），专门执行WGMMA Tensor Core操作；(3) 寄存器复用——DMA warp几乎不消耗寄存器，其寄存器资源可通过硬件bank分配到compute warpgroup，允许存储更大的accumulator或更多的pipeline stages。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hopper GEMM中warp specialization的调度伪代码（完整CTA=128+32=160 threads）：

```
// ─── DMA Warp (32 threads, warp_id=4) ───
TMA_dma_warp():
    for k = 0 to K/T_K - 1:
        // Pipelining: DMA领先compute PIPE步
        if k >= pipeline_depth:
            wait(cons[k % pipeline_depth])  // 等待consumer释放buffer
        if thread_id == 128:               // 仅1个线程调用TMA
            TMA_load(
                completion_barrier = prod[k % pipeline_depth],
                src  = gA[:, :, k],       // global memory tile
                dst  = sA[:, :, k % pipeline_depth])  // shared memory
            TMA_load(
                completion_barrier = prod[k % pipeline_depth],
                src  = gB[:, :, k],
                dst  = sB[:, :, k % pipeline_depth])
    // 最终：等待consumer完成后写output
    wait(copyout)
    if thread_id == 128:
        TMA_store(sC → gC[blk_x, blk_y])

// ─── Compute Warpgroup (128 threads, warp_id=0,1,2,3) ───
TMA_compute_warpgroup():
    for k = 0 to K/T_K - 1:
        wait(prod[k % pipeline_depth])     // 等待TMA完成
        warpgroup_sync()                    // 128线程对齐
        wgmma(accum, sA[:,:,k%P], sB[:,:,k%P])  // Tensor Core
        warpgroup_wait()                    // 等待Tensor Core完成
        arrive(cons[k % pipeline_depth])   // notify DMA: buffer free
    
    // Epilogue: accumulator registers → shared memory → TMA store
    copy_reg_to_smem(accum → sC)
    syncthreads()
    arrive(copyout)
```

关键调度特征：
- DMA warp和compute warpgroup交替执行（warp scheduler自动time-multiplexing）
- 当compute wg在等Tensor Core完成（warpgroup_wait）时，warp scheduler自动切换到DMA warp
- 反之，当DMA warp在等consumer释放buffer（wait(cons)）时，compute warpgroup获得执行
- Pipelining (PIPE=3)确保TMA延迟被完全隐藏

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- CUTLASS 3.x：warp-specialized main loop模板（`sm90_mma_tma_gmma_rs_warpspecialized.hpp`）
- ThunderKittens：通过LCSF模板将warp specialization形式化为Load/Compute/Store/Finish四个角色——load worker (warp 0) 执行 TMA 异步 load，compute workers (warpgroup) 执行 tile 计算，store worker 执行 TMA store，finish 处理退出。用户只需填充每个角色的函数体，框架自动管理 barriers 和 pipeline buffer。对比 CUTLASS 中手动管理 ping-pong scheduler 的 warp specialization，TK 的 LCSF 将 attention 实现从 2325 行减至 217 行，同时消除 FA3 的 9.6-way bank conflict。
- CUDA C++手动实现：使用`cooperative_groups::tiled_partition`和显式barrier管理
- 关键难点：(1) 正确管理producer-consumer barriers（避免deadlock和data race）；(2) 寄存器分配——误用导致spilling严重；(3) shared memory banking——pipeline buffers需正确对齐

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
- ThunderKittens: Simple, Fast, and Adorable Kernels
