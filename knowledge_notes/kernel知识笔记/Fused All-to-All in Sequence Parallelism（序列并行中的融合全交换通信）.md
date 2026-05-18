## Fused All-to-All in Sequence Parallelism（序列并行中的融合全交换通信）

术语是什么？

Fused All-to-All是Sequence Parallelism中将Q、K、V三组attention投影的all-to-all通信融合为单次collective操作的技术。原始Ulysses SP需要对Q、K、V分别执行all-to-all（3次），而融合版本将QKV projection的结果`qkv_heads[n/SP, 3×h/TP]`视为一个整体矩阵，通过单次all-to-all完成sequence layout↔head layout的转换。GQA场景下（Q head数≠KV head数），融合all-to-all变得更关键：将`qkv_heads`的head维度从`3×h`替换为`h + 2×h_kv`，单次通信同时处理Q、K、V的重分布和KV cache replication（当#KV heads < SP degree时）。

从kernel调度角度拆解术语：

Fused All-to-All的通信模式：

1. **发送端**：每个SP rank持有`[n/SP, h + 2×h_kv]`的QKV数据。需将数据重排列为head-parallel layout：每个接收GPU应获得完整序列长度n但仅处理`(h + 2×h_kv)/(SP×TP)`个heads。

2. **Buffer准备**：rank i构建send buffer，将`(h + 2×h_kv)/SP`个heads的数据打包发送给rank j（j ∈ SP group）。若#KV heads不足，在send buffer中复制KV数据——这是KV cache replication的机制。

3. **单次NCCL All-to-All调用**：通过NCCL all-to-all collective完成SP group内所有rank的全交换。通信量约为`O(n × (h + 2×h_kv) × d / SP)` per rank。

4. **接收端**：每个rank从all-to-all接收完整序列的`(h + 2×h_kv)/(SP×TP)`个heads，直接用于本地attention计算。

5. **反向All-to-All**：attention输出需再次通过fused all-to-all从head-parallel layout（`[n, h/(SP×TP)]`）转回sequence-parallel layout（`[n/SP, h/TP]`）。

融合的优势：相比3次独立all-to-all，融合减少NCCL launch overhead、提高网络利用率（更大message size），并在GQA场景天然支持KV cache replication（send buffer中复制KV head）。

术语一般如何实现？如何使用？

在ArcticInference实现中，fused all-to-all通过单次NCCL collective调用实现。send/recv buffer按head维度layout预先分配，GQA path通过send buffer中的KV head复制处理h_kv < SP的场景。用户无需手动配置——系统根据模型配置（#Q heads, #KV heads）和(SP, TP)组合自动选择通信策略。fused all-to-all是SP inference path的核心通信原语，在base config每次forward中执行2次（attention前后各一次）。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

---

