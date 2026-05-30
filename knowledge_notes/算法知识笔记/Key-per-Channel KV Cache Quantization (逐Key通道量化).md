## Key-per-Channel KV Cache Quantization (逐Key通道量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Key-per-Channel Quantization 是 KV Cache 量化中的一种分组策略：沿 Key tensor 的 channel 维度（hidden dimension）独立计算每个 channel 的 scale factor 和 zero-point，而非跨 token 或跨整个 tensor。在 LogQuant 中，采用 "Key-per-channel strategy" 作为 Quanto 后端的量化配置，对 Key 矩阵的每个 channel 独立量化。

与 per-token 量化（沿序列长度 L 维度分组：每个 token 有独立 scale）和 per-tensor 量化（整个 K tensor 共用一个 scale）相比，Key-per-channel 在 K cache 的 channel 维度上提供更精细的量化粒度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**三种分组策略对比**（K ∈ R^{L×d}）：
```
Per-Tensor:         scale ∈ R, zero ∈ R                     // 全局1组参数
Per-Token:          scale ∈ R^L, zero ∈ R^L                 // 每组1个token的d维度值
Key-per-Channel:    scale ∈ R^d, zero ∈ R^d                 // 每组1个channel的L个值
Group-wise (G=64):  scale ∈ R^{L×ceil(d/64)}, ...           // 每组64个channel的值
```

**LogQuant 采用 Key-per-Channel 的原因**：
K cache 中不同 channel 的值分布差异大（outlier channel 问题）——少数 channel 的值幅度可达正常 channel 的 6 倍以上。Key-per-channel 量化将 outlier channel 隔离在自己的组内，避免其极值放大同组正常 channel 的量化误差。而 V cache 通常不存在明显的 channel-wise outlier 模式（LogQuant paper, Section 2.1; JanusQuant paper, Section 2.3）。

**Key-per-Channel 量化流程**：
```
// K ∈ R^{L×d}, target bits=2
for c in 1..d:
    scale[c] = max(|K[:,c]|) / (2^{bits-1} - 1)     // 每channel独立scale
    zero[c] = 0                                      // 对称量化zero=0
    K_quant[:,c] = round(K[:,c] / scale[c])
    K_quant[:,c] = clamp(K_quant[:,c], -2^{bits-1}, 2^{bits-1}-1)

// 解量化
K_deq[:,c] = K_quant[:,c] * scale[c]
```

术语一般如何实现？如何使用？

在 Quanto 和 HQQ 等量化后端中，Key-per-Channel 通过 `quantize(tensor, axis=1)` 实现（axis 沿 hidden dim）。在 HuggingFace 集成中，LogQuant 将 Quanto 的 `qtype` 设置为 per-channel 量化模式。

适用场景：(1) 当 K cache 存在 channel-wise outlier 时必须使用 per-channel（否则 outlier 会严重损害量化精度）；(2) 与 per-token 互补——V cache 可用 per-token（无显著 channel outlier），K cache 用 per-channel；(3) JanusQuant 的 RtSmooth 在 Key-per-Channel 之前先对 K 做 per-token 平滑变换，使 outlier channel 的值更均匀，再用 Key-per-Channel 量化获得更好精度。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---
