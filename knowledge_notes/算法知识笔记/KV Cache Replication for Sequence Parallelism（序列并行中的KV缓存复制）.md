## KV Cache Replication for Sequence Parallelism（序列并行中的KV缓存复制）

术语是什么？

KV Cache Replication for SP是Shift Parallelism为Sequence Parallelism在inference场景下添加的机制：当使用GQA的模型中KV head数（h_kv）小于SP degree时，无法将KV head直接均匀分配到所有GPU（因为head数不够），通过all-to-all通信的send buffer中复制KV head数据来"虚拟地"扩展KV head数。这与TP的KV cache复制思路不同：TP可以通过在QKV projection weight中复制KV列来重计算KV cache（因为是weight-level partitioning），而SP每个rank只持有部分input sequence，无法通过重计算覆盖全部sequence位置，因此必须通过通信中的buffer复制实现。

典型场景：Qwen-30B-A3B有4个KV heads，但需要在8 GPU上使用SP=8进行推理。每个GPU只能原生持有0.5个KV head，无法工作。通过KV cache replication，在all-to-all send buffer中将每个KV head复制到2个target rank，实现8-way SP。

从算法pipeline角度拆解术语：

KV Cache Replication在SP inference forward pass中的位置和流程：

1. **QKV Projection**（Line 3, Algorithm 1）：输入embedding `[n/SP, d]`与QKV weight `[d, h + 2×h_kv]`乘，得`qkv_heads[n/SP, h + 2×h_kv]`。此处`h + 2×h_kv`替代了标准MHA的`3×h`。——这是与GQA的对接点。

2. **Send Buffer构建**（Line 4前）：为SP all-to-all准备send buffer。对于KV heads部分（`2×h_kv`个heads），如果`h_kv < SP×TP`（即KV heads不足以覆盖所有all-to-all target ranks），则在send buffer中复制KV数据。例如SP=8、h_kv=4时，每个KV head复制到2个target ranks。对于Q heads无需复制（Q head数通常充足）。

3. **Fused All-to-All**（Line 4）：单次all-to-all同时承载Q、K、V的head重分布和KV复制。接收端每个GPU获得完整序列的`(h + 2×h_kv)/(SP×TP)`个unique或replicated heads。

4. **Attention计算**（Line 5）：每个GPU对本地head shard执行attention，使用本地或复制的KV cache条目。由于KV cache replication保证了每个GPU都能访问所需KV heads，attention正确性不受SP degree超过#KV heads的影响。

5. **Decoder侧一致性**：在shift config（full TP）下，KV cache replicate的条目与TP weight replication产生的K/V一致——因为两者都是进行head-level replication，保证了KV Cache Invariance。

术语一般如何实现？如何使用？

实现在ArcticInference的GQA extension中。通过将QKV projection的head维度从3×h适配为`h + 2×h_kv`，并在fused all-to-all的send buffer构建阶段按需复制KV head数据。用户无需手动配置——系统根据模型config（#Q heads, #KV heads）和SP degree自动判断是否需要replication。这种机制使SP能够扩展到任意模型，不受#KV heads限制，是SP从training（通常MHA无此问题）适配到inference（大量GQA模型）的关键一步。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

---

