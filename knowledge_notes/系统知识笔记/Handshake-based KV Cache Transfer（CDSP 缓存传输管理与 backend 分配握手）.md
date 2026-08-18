## Handshake-based KV Cache Transfer（CDSP 缓存传输管理与 backend 分配握手）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Handshake-based cache transfer 是 Tetris（ISCA'26）的 prefill→decoding KV cache 传输管理机制，解决 Backend Starvation 问题：大多数传输 backend（NCCL/NVSHMEM/Mooncake 等）需要 GPU buffer，长上下文服务产生超大中间张量后可能没有足够显存为每个 prefill 实例预留专用传输 backend，导致某些实例永远拿不到 backend、decoding 实例收不齐完整 KV cache（请求被卡、部分填充 cache 长期占用 decoding 显存）。机制：prefill 实例的 send manager 在发 KV 前先向目标 receive manager 发起握手申请 backend 分配；若 receive 侧 buffer 空闲或有足够 backend 则直接用自己的专用 backend 传输；否则按首次握手时间排序，receive manager 顺序为各请求保留 backend 直到其全部 chunk 传完，防止某请求的后续 chunk 传输被打断。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# CDSP cache transfer 流程（图 9）
请求 chunk 到达 → 同时分发给 GPU workers 与 send manager
GPU workers 计算 chunk prefill 的同时：
  ① send manager 向 receive manager 发 handshake（申请 backend 分配）
  ② receive manager 确认分配（buffer-free 或排队保留）
  ③ send/receive manager 发起 KV cache 传输（NCCL v2.26+ 并发 communicator，专用 buffer+stream）
  ④ 各 chunk 传输重复上述过程；receive 侧收齐全部 chunk 后通知 local scheduler
  ⑤ local scheduler 用 iteration-level scheduling 把请求插入 decoding batch
```
Annotations: 握手与 prefill 计算独立、可无缝叠进 layer-wise 传输（与计算跨层重叠隐藏延迟）；NCCL 自 v2.26 支持并发 communicator 执行使多传输并行；专用 buffer/CUDA stream 提升带宽利用率。
论文量化：CDSP cache balancing（chunk 间 KV 重分布）仅 ≤1.8% 额外开销；CDSP handshake 传输 0.6%-11.8%（平均 2.1%）开销，backend 减半压力测试下 RPC 开销 1.5%-5.4%（平均 3.8%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Tetris 推理引擎的一部分（send/receive manager + backend 池），传输底层用 NCCL（https://github.com/NVIDIA/nccl，v2.26+ 并发 communicator）、层间传输复用 ring 通信器；与层式传输（layer-wise transmission）与 cache balancing 重叠机制组合。使用场景：prefill-decoding 解耦集群中多 prefill 实例向 decoding 实例汇聚 KV cache（CDSP 下每请求 KV 分散在多个 prefill 实例）；与 Mooncake/KVDirect 等分离式 KV 传输方案的差别在于显式握手仲裁有限 backend，防 starvation。Web 证据：NCCL 并发 communicator 支持（v2.26 release notes）。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
