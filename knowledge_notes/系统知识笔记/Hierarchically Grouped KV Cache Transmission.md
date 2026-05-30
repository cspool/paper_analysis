## Hierarchically Grouped KV Cache Transmission

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchically Grouped KV Cache Transmission 是 EPD-Serve 中用于 P-D 跨阶段 KV Cache 传输的优化机制。针对 naive 的同步全量传输——所有 Transformer 层 KVCache 在 Prefill 完成后一次性传输——导致的通信拥塞和高 TTFT，EPD-Serve 提出三层优化：(1) Layer-wise Transmission：按 Transformer 层分拆传输单元，当 Prefill 计算 L+1 层时传输 L 层 KVCache；(2) Grouped Packaging：将相邻多层 KVCache 打包为一个 group 传输，减少握手频率和 metadata overhead；(3) Precise Delayed Scheduling：延迟调度以避免通信峰值拥塞。核心目标：最大化通信-计算重叠（communication-computation overlap）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

分层分组 KV 传输的 overlap 效果（EPD-Serve Figure 7 & Table 4）：

```
seq_len=1024:
  Baseline (layer-wise, 无分组):
    KV Latency: 1127ms, Exposed: 955ms, Prefill: 6794ms
    Overlap Ratio: 15.27%, Bandwidth: 7.98 GB/s
  Optimized (hierarchically grouped):
    KV Latency: 716ms, Exposed: 8.8ms, Prefill: 6611ms
    Overlap Ratio: 98.78%, Bandwidth: 12.58 GB/s (+58%)

seq_len=2048:
  Baseline: Overlap 25.08%, BW 10.66 GB/s
  Optimized: Overlap 99.92%, BW 11.71 GB/s (+10%)
```

为什么分组传输大幅提升 overlap？
- Baseline 逐层传输：每层 KV 传输需 metadata handshake（unpredictable latency），传输单元小导致带宽利用率低，handshake 延迟打断了通信-计算重叠。
- 分组传输：多层 KV 打包 → 减少 handshake 次数 → 更大的传输 payload → 更高的带宽利用率 → 延迟调度对齐 Prefill 计算 pipeline → 几乎完全隐藏传输延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Group size 根据 MLP 计算负载和 handshake 延迟动态确定。实现：(1) Prefill 阶段维护 KV 队列——每层 KVCache 完成后入队；(2) Grouper 线程按配置的 group size 从队列取 KV chunk → 打包 → 调用 Mooncake Store 异步传输接口发送；(3) Decode 实例分层接收 KV chunk → 解包 → 拼装完整 KVCache；(4) Precise scheduling：监控当前通信链路负载，避开峰值时段发送。EPD-Serve 的消融实验：启用 P-D 分层分组后 TTFT 降低 11.9-16%。与 E-P 异步预取联合启用：TTFT 降低 26.1-31.6%。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
