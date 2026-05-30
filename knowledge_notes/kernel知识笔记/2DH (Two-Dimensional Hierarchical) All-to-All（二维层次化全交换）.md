## 2DH (Two-Dimensional Hierarchical) All-to-All（二维层次化全交换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

2DH All-to-All 是 TUTEL 提出的新型 All-to-All 通信算法，解决标准 Linear All-to-All 在大规模 GPU 集群中因消息大小 S/n 过小而无法饱和网络带宽的问题。核心思想：将 All-to-All 分解为节点内（intra-node）和节点间（inter-node）两个层次，通过 stride memory copy 将非连续的小 chunks 聚合为大 chunks，在大规模下保持高链路利用率。2DH 算法包含 4 个 phase（Figure 17）：Phase 1 stride memcpy（对齐同目标本地 GPU 的 chunks）→ Phase 2 intra-node All-to-All（m GPUs 交换）→ Phase 3 stride memcpy（对齐同目标远程节点的 chunks）→ Phase 4 inter-node All-to-All（n/m nodes 交换）。

从kernel调度角度拆解：

2DH All-to-All 算法伪代码（Algorithm 2 in paper）：

```cuda
// 2DH All-to-All: 4-phase algorithm
procedure ALL2ALL_2DH(output, input, S, n, m):
    chunksize = S / n
    nnodes = n / m
    
    // === Phase 1: Stride Memcpy (intra-node alignment) ===
    // 重排使同一本地目标 GPU 的数据连续
    strideMemcpy(buffer, input, chunksize, 
                 row=m, col=nnodes)
    // input[i] → buffer[j], j = i%m * nnodes + i/m
    // 源: GPU0 向 GPU0...GPU_n-1 的 chunks 交错排列
    // 目标: 按本地目标 GPU 分组连续排列
    
    // === Phase 2: Intra-node All-to-All ===
    for g = 0; g < m; g++:
        loc = g * nnodes * chunksize
        peer = g + node_rank * m
        ncclSend(buffer[loc], nnodes * chunksize, peer)
        ncclRecv(output[loc], nnodes * chunksize, peer)
    // 节点内 m 个 GPU 交换 S/m bytes 数据
    // chunk 大小: nnodes * chunksize = S/m (不依赖 n!)
    
    // === Phase 3: Stride Memcpy (inter-node alignment) ===
    strideMemcpy(buffer, output, chunksize,
                 row=nnodes, col=m)
    // 重排使同一远程目标节点的数据连续
    
    // === Phase 4: Inter-node All-to-All ===
    for nid = 0; nid < nnodes; nid++:
        loc = nid * m * chunksize
        peer = local_rank + nid * m
        ncclSend(buffer[loc], m * chunksize, peer)
        ncclRecv(output[loc], m * chunksize, peer)
    // 节点间 n/m nodes 交换合并后的大 chunks
    // chunk 大小: m * chunksize = S/nnodes (也大于原始的 S/n!)
end procedure

// Stride Memory Copy
procedure STRIDEMEMCPY(output, input, chunksize, row, col):
    for i = 0; i < row * col; i++:
        j = i % row * col + i / col    // stride index transform
        output[j * chunksize : (j+1) * chunksize] = 
            input[i * chunksize : (i+1) * chunksize]
end procedure
```

关键性能特性：(1) Phase 1-3 的延迟仅取决于 S（总数据量），与 GPU 数 n 无关；(2) Phase 4 的消息大小为 m·chunksize = S·m/n = S/nnodes，比 Linear A2A 的 S/n 大 m 倍；(3) 避免 naive local aggregation 中 O(n/m) 次非连续内存访问问题（当 n=2048, m=8 时延迟从 ~600μs 降至常数级别）。

术语一般如何实现？如何使用？

基于 NCCL 的 `ncclSend`/`ncclRecv` P2P API 实现（Algorithm 2），通过在 nccl-tests 的 `alltoall_perf` benchmark 中集成和验证。额外通过 MSCCL DSL 描述 2DH 算法并编译优化，利用 LL128 协议在低延迟场景进一步提升效率。在 64~4096 GPU 上验证，小消息（1 MiB）和大消息（256 MiB）均有显著加速（Figure 18），且扩展到 4096 GPU 而 Linear All-to-All 在此规模下无法成功运行。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale
