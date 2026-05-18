## Wave Quantization in GPU Compute（GPU计算的波量化效应）

术语是什么？

Wave Quantization是GPU计算利用率的一种损失现象：当kernel发射的thread block数量不能被GPU的SM数量整除时，最后一"波"（wave）执行中只有部分SM有活干，其余SM空闲。在LLM推理中，prefill阶段的attention和FFN计算的thread block数由序列长度决定，短序列或小chunk下O-proj和attention的block数较少，导致严重的wave quantization —— 部分SM idle，compute utilization从理论峰值下降至约70%-76%。Bullet识别这是chunked prefill利用率低的根源之一。

从kernel调度角度拆解术语：

Wave Quantization在LLM prefill kernel中的表现：

```
// 以Llama3.1-8B的O-proj为例，hidden_dim=4096, block_size=128
// O-proj kernel: matmul(attention_output, W_o), shape=[seq_len, 4096] × [4096, 4096]

// GPU有108个SM，每个SM可运行若干thread block
// 总thread block数 = ceil(seq_len / block_size) * ceil(4096 / thread_tile)
// 假设seq_len=256（1k chunk prefill），matmul tile=128
// blocks_per_seq_dim = ceil(256 / 128) = 2
// blocks_per_hidden_dim = ceil(4096 / 128) = 32
// total_blocks = 2 * 32 = 64

// GPU A100有108个SM，64个block只能填满59%的SM
// Wave 1: 64个block → 64个SM在工作, 44个SM idle
// 这就是wave quantization：block数不足导致SM idle
// compute utilization = 64/108 ≈ 59%

// 实际完整transformer layer有多个kernel（QKV, attention, O-proj, gate, up, down）
// 每个kernel的block数因tensor shape不同而异
// Bullet测量：完整layer的average compute utilization ≈ 70%-76%
```

当chunked prefill使用1k token chunk时：
- 每个chunk的compute efficiency进一步下降至61%
- 最后一个chunk处理时间是第一个chunk的1.9x（因为KV reload叠加wave quantization）
- 总prefill latency比unchunked高1.13x

术语一般如何实现？如何使用？

在LLM serving系统中，缓解wave quantization的方法包括：
- **增大batch/sequence长度**：更多token产生更多thread block，自然填满SM（但受SLO约束不可无限增大）
- **Kernel fusion**：将多个小kernel融合为大kernel，减少wave量化损失（如FlashInfer的融合attention kernel）
- **Persistent kernel**：使用persistent thread block设计，每个block从global work queue动态取活，消除静态block分配导致的SM idle
- **Bullet的方案**：通过intra-GPU prefill-decode concurrency，在prefill SM idle时让decode kernel在这些SM上执行，利用互补的compute/memory特性填充bubble

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

