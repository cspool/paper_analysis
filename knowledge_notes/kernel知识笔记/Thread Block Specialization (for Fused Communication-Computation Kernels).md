## Thread Block Specialization (for Fused Communication-Computation Kernels)

术语解释
Thread Block Specialization 是 Comet 提出的 fused kernel 设计模式，将通信（NVSHMEM I/O）和计算（GEMM）分配到同一 GPU kernel 内但隔离到不同的 thread blocks 中，由 GPU hardware scheduler 并发调度。这替代了传统的"垂直融合"（在 GEMM thread block 的 prologue/epilogue 中插入通信 I/O），避免了 fine-grained I/O 干扰高性能计算流水线（尤其在 Hopper TMA 异步流水线中）。

术语是什么？
在 Comet 的 fused kernel 中，两类 thread block 共存于同一 kernel launch：
- **GEMM Thread Blocks**（n^p 个）：使用标准 CUTLASS Hopper 实现——producer warp 用 TMA async copy (cp.async.bulk) 从 global memory 加载到 shared memory，consumer warp 在 tensor core 上执行 MMA。通信 I/O 完全不侵入 GEMM 流水线。
- **通信 Thread Blocks**（n^c 个）：执行 NVSHMEM get/put 进行 token 级跨 GPU 数据传输，以及 top-K reduce 操作。从 global memory 读取 GEMM 输出，处理后写回 local 或 remote memory。

两类 TB 隔离的关键优势：(1) GEMM TB 使用与融合前完全相同的 CUTLASS 实现，零性能退化；(2) 通信 I/O 的延迟波动不传播到计算流水线；(3) 可独立调节 n^c/n^p 比例适配不同负载。

从kernel调度角度拆解术语：

```
# Comet Fused Kernel 的 Thread Block 组织（Hopper, 132 SMs）
# GEMM TB 和通信 TB 在同一 kernel 内并发执行

# GEMM Thread Block（标准 CUTLASS Hopper, 每 TB 占 1 SM）:
def GEMM_thread_block(tile_A, tile_B, output_tile):
    # Producer warp: TMA async load
    cp.async.bulk(shared_A, global_A[tile_A])  # 硬件执行，不占线程
    cp.async.bulk(shared_B, global_B[tile_B])
    mbarrier.arrive_expect_tx(expected_bytes)
    
    # Consumer warp: Tensor Core MMA
    while not mbarrier.ready():
        continue  # or compute on previous tile
    accumulator = mma(shared_A, shared_B)
    # 纯计算，无通信 I/O 侵入
    store(output_tile, accumulator)

# 通信 Thread Block:
def comm_thread_block(token_indices, expert_output):
    # Step 1: 从 global memory 读取 GEMM 输出
    output = load_global_memory(expert_output, token_indices)
    
    # Step 2: Top-K reduce
    reduced = topk_reduce(output, routing_weights)
    
    # Step 3: NVSHMEM 写入 remote 或 local
    for token in reduced:
        if is_remote(token):
            nvshmem_put(token.dst_rank, token.data, token.offset)
        else:
            store_local(token.data, token.offset)
```

与垂直融合的对比：垂直融合中同一个 TB 执行 `TMA load → GEMM → comm I/O`，remote I/O 的延迟（数百 cycles）阻塞 GEMM 流水线，且 Hopper TMA 异步流水线会被同步 I/O 打破。Thread block specialization 通过空间隔离（不同 SM 上的不同 TB）而非时间隔离（同一 TB 内的阶段切换）解决此问题。

术语一般如何实现？如何使用？
- 依赖 CUDA cooperative groups 或 grid-level synchronization 来协调 GEMM TB 和通信 TB 之间的 producer-consumer 依赖
- GEMM TB 使用标准 CUTLASS 模板生成，端口到 Ampere/Volta 仅需替换对应架构的 compute TB 实现
- 通信 TB 使用 NVSHMEM API（`nvshmem_put`, `nvshmem_get`, `nvshmem_wait`）
- SM 资源限制：每个 SM 只能容纳 1 个 thread block（Hopper 上每 SM 1 TB），GEMM TB 和通信 TB 竞争 SM
- 总 TB 数 = SM 数 (132 on H800)，n^c + n^p ≤ 132

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
