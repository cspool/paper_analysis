## GPU-initiated token-level communication（GPU 发起的 Token 级通信）

术语是什么？
GPU-initiated token-level communication 是 MoE Expert Parallelism 中的一种通信模式：GPU threads 直接在 token 粒度上发起 RDMA 传输（而非先批量打包到 buffer 再统一发起），实现 per-token 或 per-chunk（如 32 tokens）的 fine-grained 通信。这种细粒度通信通过 IBGDA 技术实现，GPU kernel 为每个 token/chunk 独立构造 work request 并直接提交到 NIC。

从kernel调度角度拆解术语：
```
// 对比 coarse-grained vs fine-grained:
// Coarse-grained (NCCL/RCCL):
//   GPU: 将 T 个 tokens 按 dest_rank 打包到连续 buffer (O(T·C·D) memory)
//   CPU: 为每个 dest_rank 构造一个 bulk WR → NIC send
//   问题: packing 开销大，小 T 时吞吐低

// Fine-grained GPU-initiated (DeepEP/UCCL-EP):
//   GPU: 每 token/chunk (如 32 tokens) 独立提交 TransferCmd
//   NIC: 直接从 GPU buffer DMA 数据 (无需 CPU 参与 packing)
//   优势: overlap token dispatch 与 compute, dedup, hierarchical reduce

// 通信量: 典型 7M ops/s/GPU (DeepSeek-V3, 7KB/activation, 400G network)
```

术语一般如何实现？如何使用？
通过 DeepEP (IBGDA-based) 或 UCCL-EP (CPU-proxy-based) 实现。在 HT mode 中，32 tokens 构成一个 chunk，一次提交传输。GPU-initiated 使 token deduplication 和 hierarchical reduce 成为可能：GPU kernel 在提交传输前检查同节点多专家场景，去除冗余传输。在 UCCL-EP 中，GPU 仍负责 token-level 的 initiation 决策（保持 fine-grained overlap），但实际传输执行委托给 CPU proxy（换取可移植性）。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
