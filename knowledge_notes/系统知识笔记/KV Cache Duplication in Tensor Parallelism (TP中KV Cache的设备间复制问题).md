## KV Cache Duplication in Tensor Parallelism (TP中KV Cache的设备间复制问题)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Duplication in Tensor Parallelism 是指在 Tensor Parallelism (TP) 分布式推理中，某些 attention 变体的 KV cache 结构导致每设备必须持有完整或部分复制的 KV cache，从而削弱了 TP 本应带来的 per-device 内存节省。

该问题的根本原因在于 TP 的切分方式与 attention head 结构的交互。TP 通常沿 attention head 维度切分 Q/K/V 投影权重和 up-projection 矩阵（列并行），每个 TP rank 负责部分 head 的计算。对于 GQA/MHA：每 rank 持有其负责的 KV head 的 cache，总 cache 随 TP degree 线性缩减。对于 MLA：由于 K/V 不是直接缓存而是从 latent c^{KV} 解压，而 up-projection 矩阵的列并行切分要求每 rank 都能访问完整 latent 以重建其负责的 head 的 K/V——因此 latent 必须在**所有 TP rank 上复制**。

MLA 的 TP=2 时每 device KV cache = 4d_h（与 unsharded 相同），TP=4 时仍为 4d_h——TP 完全无法减少 MLA 的 per-device KV cache。这是 MLA 设计的核心可扩展性缺陷。

从系统架构角度拆解术语。

**MLA vs GLA 的 TP 行为对比（h_q=16, d_h=128, d_c=4d_h=512 for MLA, h_c=2, d_c=2d_h=256 for GLA-2）：**

```
MLA (TP=2):
  Rank 0: 负责 head 0..7 → 需完整 c^{KV} (512 dims) 重建 K/V_0..7
  Rank 1: 负责 head 8..15 → 需完整 c^{KV} (512 dims) 重建 K/V_8..15
  Per-device: 512 dims/token × 2 bytes = 1024 bytes + d_R
  → TP 无节省，latent 被全复制

GLA-2 (TP=2):
  Rank 0: c_0^{KV} (256 dims), Q_0, W^{VO}_0 → 仅需 c_0^{KV}
  Rank 1: c_1^{KV} (256 dims), Q_1, W^{VO}_1 → 仅需 c_1^{KV}
  Per-device: 256 dims/token × 2 bytes = 512 bytes + d_R/2
  → TP=2 时 per-device cache 减半

MLA (TP=4): 仍 512 dims/device（latent 被 4× 复制）
GLA-4 (TP=4): 128 dims/device（每 rank 1 个 latent head）
```

**To mitigate MLA's duplication, prior systems fall back on hybrid TP+DP**——将 attention submodule 在不同 DP group 间复制，不同 batch 序列分配给不同 DP rank。但此方案引入严重的 straggler 问题：DP barrier 要求所有 rank 完成计算后同步，一个 DP rank 处理的长序列阻塞所有其他 rank。

术语一般如何实现？如何使用？

GLA 通过 multi-latent-head 设计从根本上解决此问题——latent head 沿 TP 分片而非复制。实现要点：(a) 每 latent head 有独立的 up-projection 矩阵，吸收进 Q/O 投影后直接对 latent 做 attention；(b) 每 rank 仅持有其 latent head + 对应 Q group + O projection slice；(c) 通过 AllReduce 聚合各 rank 的 partial output。

评估结果（DeepSeek-Coder-V2 236B, 8×H100）：MLA TP=2+DP=4 在 131K prefill 不均匀负载下 throughput 37 tok/s，GLA-8 TP=8 100 tok/s（+2.7×），pure TP 避免了 DP 的 straggler barrier。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---
