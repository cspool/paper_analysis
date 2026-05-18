## Ragged Batching Kernels via Round-Robin Tile-to-Thread-Block Mapping（基于Round-Robin Tile映射的Ragged批处理Kernel）

术语是什么？通过联网搜索让回答具体和精准。

Ragged Batching Kernel是Difflow为支持异构shape请求的高效批处理而实现的kernel层技术。核心分为两类：(1) **Ragged data-independent operation kernels**：对不跨请求共享数据的操作（transpose/reduce等），基于已有regular operator的tiling plan和microkernels，将每个请求划分为tile集合，通过round-robin policy在batch执行时映射tile到GPU thread blocks；(2) **Redundancy Memory Access Elimination**：对attention操作中冗余K/V tensors，运行时压缩K/V沿redundant batch dim + concat Q tensors → 使用FlashAttention等标准kernel执行压缩计算。Difflow实现了四个ragged data-independent operation kernels（Triton + CUDA）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Round-Robin Tile-to-Thread-Block Mapping伪代码：

```
// === Ragged Data-Independent Operation Kernel ===
// 输入: batch of requests, each with different shape
// 已有: regular operator的tiling plan (tile_H × tile_W)和microkernel

1. Function RaggedDataIndependentKernel(batched_requests, op_type):
2.     // Step 1: Tile decomposition (per-request独立)
3.     all_tiles = []
4.     for each request r in batched_requests:
5.         r_tiles = decompose_to_tiles(r.input, op.tiling_plan)
6.         // 例如: 256×256→4个128×128 tiles; 512×768→24个128×128 tiles
7.         all_tiles.append((r.id, r_tiles))
8.     
9.     // Step 2: Round-robin tile dispatch to thread blocks
10.    total_tiles = sum(len(r.tiles) for r in batched_requests)
11.    num_blocks = min(total_tiles, max_thread_blocks)
12.    // Launch num_blocks thread blocks
13.    for block_id in 0..num_blocks-1:
14.        // Round-robin: 每个block轮流处理不同请求的tiles
15.        tile_idx = block_id
16.        while tile_idx < total_tiles:
17.            (request_id, tile) = flatten_tiles[tile_idx]
18.            execute_microkernel(tile, request_id)
19.            tile_idx += num_blocks  // stride = num_blocks

// === Redundancy Memory Access Elimination (Attention) ===
// 例: 3个请求，共享相同prompt → 相同K/V tensors
// Baseline: Q1×K1^T, Q2×K2^T, Q3×K3^T → 3次独立attention (K1=K2=K3)
// Eliminated:
20. K_compressed = compress(K, along redundant batch dim)  // [1, H, S, d] (去重)
21. V_compressed = compress(V, along redundant batch dim)  // [1, H, S, d]
22. Q_concatenated = concat([Q1, Q2, Q3], along batch dim)  // [3, H, S_q, d]
23. output = FlashAttention(Q_concatenated, K_compressed, V_compressed)
24. // 沿batch dim split + broadcast 恢复各请求的output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow基于Triton和CUDA实现了4个ragged data-independent operation kernel。Round-robin mapping策略的关键优势：(1) 无需为不同shape组合设计specialized tiling——每个请求的tile独立decompose然后通过round-robin负载均衡到thread blocks；(2) 与regular kernel共享microkernel实现，仅调度层不同。Redundancy Memory Access Elimination通过等价线性代数变换实现——compress+concat操作是lightweight的（无额外compute），仅需改变tensor layout，随后直接调用FlashAttention等高度优化的regular kernel。该方法避免了为冗余attention场景重写优化kernel。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

