## KV Cache Reload Overhead in Chunked Prefill（分块预填充中的KV Cache重复加载开销）

术语是什么？

KV Cache Reload Overhead是chunked prefill调度策略下的一个性能缺陷：当长prefill被拆分为多个chunk后，第i个chunk的attention计算需要访问前i-1个chunk已写入的KV cache。这导致近似 N(N+1)/2 次KV cache重复读取（N为chunk数量），而非unchunked prefill的1次读取。Bullet在16k-token prefill上量化了这一效应：1k chunk下compute efficiency从71%降至61%，最后一个chunk处理时间为第一个的1.9x，总prefill latency比unchunked高1.13x。

从系统架构角度拆解术语：

KV Cache Reload的发生流程：

```
// Unchunked prefill（无reload）：16k token，1次prefill forward
// KV cache写入：每层1次，共L层
// KV cache读取：attention中每层读取已计算的KV，无需重复

// 1k chunk prefill（严重reload）：16k token → 16个chunk
for chunk_i in range(1, 17):
    // chunk_i有1000个新token
    // attention需读取前(i-1)*1000个token的KV cache
    // chunk_1: 读取0个历史KV（仅self-attention）
    // chunk_2: 读取1000个历史KV token
    // chunk_3: 读取2000个历史KV token
    // ...
    // chunk_16: 读取15000个历史KV token
    // 总KV读取量 ∝ N(N+1)/2 = 16*17/2 = 136 倍于单chunk的KV量

// 实际测量（Bullet论文§2）：
// 16k prefill, 1k chunk: total prefill latency 1.13x higher than unchunked
// 最后一个chunk latency 1.9x of first chunk（因为KV reload + wave quantization叠加）
// compute efficiency drop: 71% → 61%
```

与PD Multiplexing的对比：intra-GPU PD disaggregation通过空间上分离prefill和decode的SM，不再需要将prefill切分为chunk（或大幅减少切分次数），从而从机制上消除KV cache reload开销。

术语一般如何实现？如何使用？

在生产系统中，KV cache reload不可避免于chunked prefill。缓解手段包括：
- 增大chunk size（如从1k增到2k），减少chunk数量，但会增加per-chunk latency（Bullet测2k chunk的per-chunk latency为1k的1.86x）
- 使用FlashInfer的融合attention kernel，在single kernel内完成prefill chunk attention和cross-attention with KV cache，减少HBM往返
- 使用intra-GPU PD disaggregation（如MuxWise/Bullet），通过SM空间分区消除chunk切分的需要
- PD disaggregation到不同GPU，将prefill完整执行而无需切分

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---
