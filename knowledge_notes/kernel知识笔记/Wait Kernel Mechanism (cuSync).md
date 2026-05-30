## Wait Kernel Mechanism (cuSync)

术语是什么？
Wait Kernel 是 cuSync 框架中的一个轻量级机制，用于确保在细粒度同步场景下，producer kernel 的 thread block 先于 consumer kernel 被调度到 SM 上。它是一个包含单个线程的特殊 CUDA kernel，在 consumer stream 上发射。该线程在 global memory semaphore 上 busy-wait（`while(*sem != expected)`），直到 producer kernel 的第一个 thread block 调用 `stage.start()` 设置该 semaphore。Wait kernel 退出后，CUDA runtime 才会在该 stream 上发射后续的 consumer kernel。

从kernel调度角度拆解术语：
```
// cuSync的wait-kernel确保producer先获得SM
cudaStream_t prod_stream, cons_stream;  // 不同stream

// Producer先发射（高优先级stream或先发射顺序）
gemm<<<grid1, tb1, prod_stream>>>(..., prod_stage);

// Consumer stream: 先发射wait-kernel，再发射consumer kernel
cons_stage.waitKernel();  // 在cons_stream上发射单线程kernel
                          // 该线程busy-wait直到prod_stage.start()设置semaphore
gemm<<<grid2, tb2, cons_stream>>>(..., cons_stage);

// 调度时序:
// T1: Wait-kernel的1个线程占用1个SM，busy-wait
// T2: Producer kernel获得SM，stage.start()设置semaphore
// T3: Wait-kernel退出，释放其SM
// T4: Consumer kernel获得SM

// 优化：如果producer和consumer都能在≤2 waves内完成，可省略wait-kernel
```

术语一般如何实现？如何使用？
Wait kernel 在 cuSync 的 `CuStage` 类中实现，用户通过 `cons_stage.waitKernel()` 调用。其内部实现为：在 consumer stream 上发射一个单线程 kernel（grid=(1,1,1), block=(1,1,1)），该线程在 global memory semaphore 上执行 `while(atomicLoad(sem) == 0);`。当 producer 调用 `stage.start()` 时，其第一个 thread block 的第一个线程执行 `atomicExch(sem, 1)` 并将 `__threadfence_system()` 确保可见性，wait kernel 随即退出。前提假设：CUDA 按 kernel 发射顺序调度 thread block（论文验证 CUDA 11/12 + Volta/Ampere 满足此假设）。cuSyncGen 可自动判断是否能省略 wait kernel（当两个 kernel 的总 thread block 数 ≤ 2×SM 数时）。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
