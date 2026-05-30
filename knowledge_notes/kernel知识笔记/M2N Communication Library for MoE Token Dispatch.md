## M2N Communication Library for MoE Token Dispatch

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
M2N Communication Library 是 MegaScale-Infer 中为 MoE 推理场景的 M（attention GPU 数量）×N（expert GPU 数量）非对称通信模式专门设计的高性能通信库（~4900 行 C/C++ + ~5000 行 Python PyTorch extension）。在 Disaggregated Expert Parallelism 中，每个 MoE layer 需将 token embeddings 从 M 个 attention GPU 发送到 N 个 expert GPU，再从 N 个 expert GPU 返回 M 个 attention GPU——这是 many-to-many 通信模式，不同于 NCCL all-to-all（等量对称）。使用 GPUDirect + RDMA write with immediate + CUDA stream blocking（cuStreamWaitValue32）+ GDRCopy flush 消除 NCCL 的三大开销：GPU-to-CPU 中间拷贝、group initialization/closing、GPU synchronization instability。256KB data size 下 vs NCCL：68.2% median latency 降低、92.9% P99 latency 降低、4.2× throughput 提升。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
M2N Sender（Attention GPU → Expert GPU）：
```
// 1. cudaEventSynchronize 等待前序 kernel 完成
// 2. cuStreamWaitValue32(stream, &flag, EQ, 0) 阻塞 stream
// 3. CPU Core Sender: for each receiver in N:
//      ibv_post_send(qp[i], RDMA_WRITE_WITH_IMM, gpu_buffer+offset, len)
//    GPUDirect: 数据从 GPU 显存直接经 NIC 发出，无 CPU buffer 拷贝
// 4. while (ibv_poll_cq(cq, &wc) == 0) spin;  // 确认远端写入完成
// 5. flag = 1;  // 唤醒 CUDA stream
```

M2N Receiver（Expert GPU 侧）：
```
// 1-2. 同上 event wait + stream block
// 3. poll CQ 确认数据到达
// 4. gdr_copy_to_mapping(...)  // GDRCopy flush: 清除 GPU L2 stale cache
//    （RDMA 直接写 GPU 显存绕过 L2 → 需 flush 保证后续 kernel 读最新数据）
// 5. recv_flag = 1;  // 唤醒 stream, Expert FFN kernel 执行
```

NCCL 额外开销（M2N 消除）：(a) GPU→CPU proxy buffer copy；(b) batch-of-8 group operation 限制；(c) general collective setup/verification；(d) GPU sync 引发 P99 instability（NCCL P99 >1000μs, M2N <100μs at 32 receivers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 依赖：CUDA driver API、libibverbs、GPUDirect RDMA、GDRCopy、NVIDIA RDMA completion queue。
- CPU vs GPU 通信（vs DeepEP）：M2N 用 CPU 控制 inter-node 通信，单线程在 ~256KB/pair 下饱和带宽，不占 GPU SM；DeepEP 用 GPU SM 并行管理 QP，需 PTX 优化避免 L2 cache 争用。MegaScale-Infer 场景（每 pair 几百 KB）CPU 方案更优。
- Traffic optimizations：(a) ACK 高优先级队列隔离（避免 ACK 被 data 阻塞）；(b) 拥塞控制微调（适应不均衡流量）。
- M/N scaling：8×8→32×32 均保持 3.3-5.8× throughput 优势 vs NCCL。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---
