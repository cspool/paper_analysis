## All-to-All Communication in MoE Expert Parallelism（MoE专家并行中的全交换通信）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
All-to-All Communication 是 MoE Expert Parallelism (EP) 中的关键集体通信操作，在两个阶段出现：(1) **Dispatch All-to-All**：在 expert 层执行前，token 按 router 选择的 expert 分配结果被发送到持有对应 expert 的 GPU；(2) **Combine All-to-All**：在 expert 层执行后，各 GPU 上的 expert 输出被发送回 token 原始所在的 GPU 进行 merge（如 Top-K 加权累加）。All-to-All 是 barrier 式通信——所有参与 GPU 必须同时调用通信原语，最慢的 GPU 决定整体通信完成时间。在 MoE 推理中，all-to-all 通信开销可占 DeepSeek-V2-Lite MoE layer forward latency 的 59.2%（Sem-MoE 数据），是 EP 在跨节点场景中的主要性能瓶颈。

从kernel调度角度拆解术语：
以 NCCL all-to-all 为例的 MoE EP 通信流程：
```
// Standard EP All-to-All flow (e.g., SGLang):
// 在每个 expert layer 前后:
//
// [Dispatch Phase]
// 每个 GPU 将 token 按目标 expert 分组:
send_counts[i] = 本GPU要发给GPU_i的token数
recv_counts[i] = 本GPU要从GPU_i接收的token数
// NCCL all-to-all scatter: 
//   GPU_j 发送 send_counts[j] 个 token embeddings 给 GPU_j
//   同时从每个 GPU_i 接收 recv_counts[i] 个 token embeddings
// → Barrier: 所有 GPU 必须完成此操作
//
// [Expert Compute]
// 每个 GPU 对收到的 tokens 执行本地 experts 的 FFN
//
// [Combine Phase]  
// NCCL all-to-all gather: 将 expert 输出送回原 GPU
// → Barrier: 所有 GPU 必须完成此操作
```

在 load skew 存在的情况下，hot expert GPU 处理大量 tokens 耗时远超 cold expert GPU。由于 all-to-all 是 barrier 操作，所有 GPU 必须在两个 barrier 点等待 hot expert GPU 完成计算和通信——这是 AEP 论文的核心动机（GPU stall 可占总时间的 70%）。

AMoE 的替代方案：**取消 barrier all-to-all，改用异步 P2P 通信**：
- Phase 1：ZeroMQ（CPU message queue）交换 metadata（tensor size, sender GPU rank）
- Phase 2：NCCL P2P（ncclSend/ncclRecv）直接 GPU-to-GPU tensor 传输
- CPU 启动 NCCL kernel 后立刻处理下一个传输任务（不等待完成）
- 接收方在将 tensor 交 Scheduler 前按需同步（cudaStreamSynchronize）

术语一般如何实现？如何使用？
主流框架实现：NCCL `alltoall` 或 `alltoallv`（支持不等长消息）；DeepSpeed-MoE 提供 hierarchical all-to-all（在节点内 NVLink all-to-all 和跨节点网络 all-to-all 间拆解）。ScaleMoE 揭示 EP 中 all-to-all 的 zero padding 问题（因 expert 选择不均衡，zero ratio 可高达 98%），提出 Adaptive All-to-All 通过精确 slice size 的 NCCL alltoallv 消除 padding。AMoE 则完全取消 all-to-all，改为异步 P2P，是另一种根本性的解决方案。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
- Sem-MoE: Semantic-Aware Model-Data Collaborative Scheduling for Communication-Efficient MoE Inference
