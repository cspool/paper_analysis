## NVSHMEM Symmetric Memory for MoE Serving（面向MoE服务的NVSHMEM对称内存）

术语是什么？
NVSHMEM (NVIDIA Shared Memory) 是 NVIDIA 提供的基于 OpenSHMEM 标准的 GPU 间通信库，核心特性是 symmetric memory——在所有 GPU 上分配相同大小、相同虚拟地址空间的内存区域，允许每个 GPU 通过直接 put/get 操作访问远程 GPU 的对称内存。PROBE 使用 NVSHMEM symmetric memory 管理 replicated-expert buffer：每 rank 分配固定大小的对称内存区域存放最多 3 个 expert 副本（双缓冲 6 slots），通过 NVSHMEM put 实现高效 expert weight P2P 传输。

从kernel调度角度拆解术语：
NVSHMEM 在 PROBE 中的使用：
```
// 初始化：在 EP group 内分配 symmetric memory
nvshmem_init()
expert_buffer = nvshmem_malloc(sizeof(expert) × 6 × num_ranks)  // 对称内存
// expert_buffer 在所有 rank 上具有相同虚拟地址

// P2P expert 传输（Triton kernel 内）：
nvshmem_putmem_nbi(
    dst = expert_buffer + rank_dst * 6 * sizeof(expert) + slot_idx,
    src = expert_weights + expert_idx * sizeof(expert),
    size = sizeof(expert),
    pe = rank_dst           // 目标 rank
)

// 全局 All-Gather 聚合预测结果：
nvshmem_allgather(
    output = global_prediction_counts,  // [ep, num_experts]
    input = local_prediction_counts,    // [num_experts]
    size = num_experts × sizeof(int)
)
```
与 NCCL collective 对比：NVSHMEM put/get 是单边操作（不需要目标 rank 参与同步），更灵活；支持 GPU kernel 内直接发起，无需 CPU 线程介入；适合 PROBE 的 P2P expert 传输场景。

术语一般如何实现？如何使用？
需要 NVLink/NVSwitch（单机内）或 InfiniBand（跨机）。安装：`nvshmem 3.3.20+`，配合 CUDA 12.x。使用流程：(1) `nvshmem_init()` 初始化；(2) `nvshmem_malloc()` 分配对称内存；(3) `nvshmem_put/get` 进行单边 RDMA 传输；(4) `nvshmem_barrier_all()` 同步。PROBE 中限制 6 slots per rank 的对称内存开销（双缓冲 3 incoming + 3 outgoing），避免过多侵占 KV cache 空间。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
