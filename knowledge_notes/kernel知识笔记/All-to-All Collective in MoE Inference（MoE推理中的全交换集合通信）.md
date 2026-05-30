## All-to-All Collective in MoE Inference（MoE推理中的全交换集合通信）

术语是什么？
All-to-All Collective 是 Expert Parallelism 下 MoE 层执行的关键通信原语。每次 MoE 层执行两次 All-to-All：(1) Dispatch — 各 rank 将 token hidden states 按 Router 决策发送到持有对应 expert 的 rank；(2) Combine — 各 rank 将 expert 计算结果发回 token 原始所在 rank。与 Sequence Parallelism 中的 All-to-All（swapping sequence/head layout）不同，MoE 的 All-to-All 是 token-level scatter/gather，通信模式由 Router 输出动态决定。

从kernel调度角度拆解术语：
MoE All-to-All 通信 kernel 的调度流程：
```
// Dispatch Phase
每个 rank r:
  send_buf = []  // 按目标 rank 分组
  for token t in local_batch[r]:
    for expert e in topk_indices[t]:
      target = expert_to_rank[e]
      send_buf[target].append((token_data[t], e, t_idx))
  
  All-to-All Scatter: send_buf[target] → rank target
  
每个 rank 接收后:
  recv_tokens = 从各 rank 收到的 tokens
  // 按 expert 分组用于 Grouped GEMM

// Combine Phase (对称反向)
每个 rank r:
  按 t_idx 排序 expert outputs
  All-to-All Gather: expert_outputs → token 原始 rank
```
PROBE 论文揭示了 MoE All-to-All 的 Double Penalty：hotspot rank 同时是最大收发量 rank——Dispatch 时接收最多 unique token，Combine 时发送最多 output。DeepEP 通过 token deduplication 和 topology-aware routing 优化了通信效率，但无法消除 skew 导致的瓶颈 rank。

术语一般如何实现？如何使用？
主流实现：NCCL All-to-All（通用）、DeepEP（MoE 专用，优化 token dedup 和 NVLink 拓扑）、SGLang 集成的 DeepEP normal mode。跨节点时使用 RDMA (InfiniBand/RoCE)。关键优化维度：(1) token deduplication — 同一 rank→同一 remote expert 的多 token 合并为单次 send；(2) 与 GEMM overlap — 利用 CUDA stream 将通信与计算流水线化。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

PopFetcher (USENIX ATC '25) 重点解决 MoE 训练中 All-to-All 占单层总时间 50-60% 的瓶颈：通过非 MoE 计算（Attention 层）期间异步预取热门 expert，使被预取 expert 的 token 本地计算而无需 All-to-All dispatch；在 backward pass 中将 All-to-All 和 All-Reduce 分解为 micro-operations 流水线交错执行，All-to-All 优先级高于 All-Reduce，避免 gradient aggregation 阻塞 token 回传。训练 latency 公式：Lat_w^origin = 3×4B_wαH²/P_w + 4H Σ B_{n,w}^i / W_{n,w}；预取后，token transfer 项变为仅未预取 expert 的 token（即 B_{n,w}^i(1-δ_{n,w}^i)），加上梯度 reduction 开销 2αH² Σ δ_{n,w}^i / W_{n,w}。
