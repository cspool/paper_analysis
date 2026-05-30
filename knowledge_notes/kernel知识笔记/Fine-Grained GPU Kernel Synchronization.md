## Fine-Grained GPU Kernel Synchronization

术语是什么？
Fine-Grained GPU Kernel Synchronization（GPU kernel 细粒度同步）是一种将多个有依赖关系的 CUDA kernel 之间的同步粒度从 kernel 级下推到 tile 级的技术。传统方法（CUDA Stream Synchronization）要求 consumer kernel 的所有 thread block 必须等待 producer kernel 的**所有** thread block 完成后才能开始执行。细粒度同步则仅同步依赖的 tile（thread block），允许两个 kernel 的 independent tiles 在同一 wave 中并发执行。cuSync 框架通过四个机制实现：(i) 在独立 stream 上发射依赖 kernel 消除 stream 同步；(ii) wait-kernel 确保 producer kernel 先被调度；(iii) 自定义 tile 处理顺序（如 RowMajor）最小化 consumer 等待时间；(iv) 使用 global memory semaphore + memory fence 实现 tile 级的 post/wait 同步。

从kernel调度角度拆解术语：
cuSync 的细粒度同步通过以下伪代码逻辑实现：
```
// Producer Kernel
gemm<<<grid1, tb1, prod.stream()>>>(A, B, C, K, prod_stage):
    prod_stage.start()                    // 设置semaphore通知wait-kernel
    (row, col) = prod_stage.tile()        // 从atomic counter获取自定义顺序的tile索引
    for tk in 0..K step TileK:
        prod_stage.wait(A, row, tk)       // 等待依赖tile(对producer为no-op)
        LoadTileToShMem(Ash, A, row, tk)
        prod_stage.wait(B, col, tk)
        LoadTileToShMem(Bsh, B, col, tk)
        MultiplyAccumulate(C, Ash, Bsh, row, col, tk)
    prod_stage.post(row, col)             // __threadfence_system + atomicAdd(sem, 1)

// Consumer Kernel (不同stream)
cons.waitKernel()                         // 单线程busy-wait确保producer先获得SM
gemm<<<grid2, tb2, cons.stream()>>>(C, D, E, K, cons_stage):
    // 仅等待依赖的producer tile，可与其他producer tile并发
    cons_stage.wait(C, row, tk)
```
在 GPU 硬件上：Producer kernel 的 thread block 计算完其 tile 后，通过 `__threadfence_system()` 确保写入对全局可见，然后 `atomicAdd` 递增 semaphore。Consumer kernel 的 thread block 在加载输入 tile 前，第一个线程在 global memory semaphore 上 busy-wait (`while(*sem != expected)`) 直到 semaphore 达到预期值，其余线程被 `__syncthreads` 阻塞。这种设计使 consumer thread block 可以不必等待所有 producer thread block 完成，仅需等待其直接依赖的 producer tile(s)。

术语一般如何实现？如何使用？
cuSync 以 header-only CUDA 库形式提供（开源：github.com/microsoft/cusync）。使用流程：(1) 用 cuSyncGen DSL 描述 kernel 间 tile 依赖；(2) cuSyncGen 生成 policy 类（sem/value 方法）和 tile 处理顺序函数；(3) 用户修改 CUDA kernel，在 tile 加载前添加 wait() 调用，在 tile 计算后添加 post() 调用；(4) 主函数创建 CuStage 对象、声明依赖、在不同 stream 上发射 kernel。修改量极小（CUTLASS GeMM 约 25 行/0.5%，Conv2D 约 22 行/0.6%）。适用于所有 tile-based kernel（GeMM、Conv2D、Dropout、Softmax）。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
