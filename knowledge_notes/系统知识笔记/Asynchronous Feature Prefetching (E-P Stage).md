## Asynchronous Feature Prefetching (E-P Stage)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Asynchronous Feature Prefetching 是 EPD-Serve 中用于 E-P 跨阶段数据传输的优化机制。核心思想：不直接传输 Encode 阶段产生的完整特征向量（可能很大，如 4K 分辨率图像产生 [16206, 3584] 的 tensor），而是仅传输特征的 hash 值（轻量事件），让 Prefill 实例通过 MM Store 异步检索和预取特征数据。传输与 Encode 计算重叠，隐藏通信延迟。三个关键子机制：Multimodal Cache Pooling（MM Store 缓存）、Event-Driven Asynchronous Prefetching（事件驱动的 hash 通知）、Fault-Tolerant Recomputing（缓存 miss 时的本地重算）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

E-P 异步预取在不同分辨率下的 overlap 效果（EPD-Serve Table 3）：

```
分辨率      传输数据Shape    传输延迟   调度延迟   Overlap Ratio
280×280     [100, 3584]      8.15ms    30.80ms    100%
560×560     [400, 3584]     15.82ms    42.41ms    100%
720×1280   [1196, 3584]     38.78ms    81.03ms    100%
1080×1920  [2691, 3584]     80.77ms   151.77ms    100%
4096×3112  [16206, 3584]   729.72ms   728.11ms     99.78%
```

关键洞察：主流分辨率下传输延迟 < 调度延迟（inter/intra-instance scheduling），传输完全被 mask；超高分辨率时 overlap ratio 降至 99.78%，仍接近完全重叠。

```
Timeline:
Encode:   [======= ViT 编码 =======]
                                 ↓ hash event (轻量)
Prefill:                           [listener 等待] [检索特征] [计算]
                                  ↑ overlap: hash传输被隐藏
Transmission:                      [feature get from MM Store]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现：基于 Mooncake Store 的异步 I/O 接口。Encode 完成后调用 put(hash, V_m)，然后通过 event-driven 消息队列（如 RDMA send/recv 或 TCP notification）向 Prefill 实例发送 hash。Prefill 的 listener 线程接收 hash → 异步调用 get(hash) → V_m 写入 Prefill 本地缓存。当 Prefill 计算需要 V_m 时已提前就绪。对比 baseline（同步传输完整 tensor）：EPD-Serve 的 E-P 异步预取在 2-3 req/s 下降低 TTFT 16.6-21.7%。限制：(1) 需要 MM Store 的额外存储开销；(2) hash collision 或 cache miss 时触发本地重算，产生额外延迟。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
