## CUDA Pinned Memory for H2D/D2H Transfer (CUDA页锁定内存加速主机-设备传输)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Pinned Memory (页锁定内存) 是 CUDA 提供的一种 host memory 分配方式（通过 cudaHostAlloc / cudaMallocHost），分配的内存被操作系统锁定在物理内存中（不会被 swap 到磁盘），使得 GPU DMA 引擎可以直接访问该内存区域进行数据传输，无需经过中间 bounce buffer 的额外 copy。默认的 malloc/new 分配的 pageable memory 在 H2D/D2H transfer 时，CUDA driver 需要先将数据 copy 到临时的 pinned staging buffer，再通过 DMA 传输，导致双倍内存带宽消耗和额外延迟。Pinned Memory 可以消除这一次额外 copy，并通过 cudaMemcpyAsync 在独立的 CUDA stream 上与 GPU kernel 执行重叠。

从kernel调度角度拆解术语：
MoESys 中使用 Pinned Memory + Async Copy 实现 H2D/D2H overlap 的 kernel 级调度：
```
// GPU Stream 0: Default compute stream
for layer i in model.layers:
    // 计算第 i 层
    launch_attention_kernel<<<grid, block, 0, stream0>>>(input_i)
    launch_moe_ffn_kernel<<<grid, block, 0, stream0>>>(expert_input_i)
    cudaStreamSynchronize(stream0)

// GPU Stream 1: Async copy stream
for layer i in model.layers:
    // 异步预取第 i+1 层 expert 参数
    cudaMemcpyAsync(
        dst = gpu_expert_params[i+1],
        src = cpu_pinned_expert_params[i+1],
        size = expert_param_size,
        kind = cudaMemcpyHostToDevice,
        stream = stream1  // 独立 stream，与 stream0 并行
    )
// stream0 和 stream1 并发执行：compute(stream0) || H2D_copy(stream1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 分配方式：`cudaHostAlloc(&ptr, size, cudaHostAllocDefault)` 或 `cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined)`（后者牺牲 read 性能换取更高的 PCIe write 带宽，适合纯 H2D 场景）。
- 注意：过多 pinned memory 会减少 OS 可用的 pageable memory 并可能导致系统不稳定，通常限制在物理内存的 25-50% 以内。
- 在 MoESys 中，pinned memory 用于 Ring Memory Offloading 和 2D Prefetch 中的 CPU→GPU 参数传输，是 computation-communication overlap 的底层支撑。
- 类似优化广泛应用于其他 MoE serving 系统：DeepSpeed-Inference、MoE-Infinity、Klotski 等均使用 pinned memory + async copy。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
