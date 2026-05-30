## DeepEP

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

DeepEP 是 DeepSeek 团队开发的 MoE 专用通信库（https://github.com/deepseek-ai/DeepEP），为 DeepSeek-V3 训练和推理提供通信后端。构建在 NVSHMEM 之上，通过 warp specialization 和全流水线 IB-NVLink 数据路径实现高效通信。使用 NVSHMEM one-sided put/get 操作进行跨节点通信，配合 IBGDA (InfiniBand GPUDirect Async) 实现 GPU 直接访问远程内存，减少 CPU 介入和软件开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// DeepEP 通信模式 (NVSHMEM one-sided)
// Sender 直接 RDMA put 到 remote GPU 的对称内存:
nvshmem_putmem_nbi(remote_buffer, local_data, size, peer_rank);
// Receiver 不需要 recv —— 数据已由 RDMA 写入 local buffer
nvshmem_fence();
nvshmem_quiet();  // 等待所有 outstanding put 完成

// Warp specialization: SM 的 warp 分两组
//   - 通信 warp: 专职 NVSHMEM put/get + 轮询 completion
//   - 计算 warp: 在通信进行中执行 expert GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 优化与 InfiniBand、NVLink、IBGDA 紧密耦合，portability 受限
- Small message 场景（如 4K seqlen）下 NVSHMEM one-sided 开销低于 NCCL two-sided——FUSCO 在低序列长度下相对 DeepEP 优势较小
- Token deduplication 是局部和静态的，不如 FUSCO hierarchical routing 灵活
- FUSCO 在 real-world 16K seqlen 下比 DeepEP 快 1.13-1.34×，在 single-node routed 场景下快 1.95-2.01×

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
