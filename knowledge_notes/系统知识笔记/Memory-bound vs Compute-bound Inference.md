## Memory-bound vs Compute-bound Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 推理的工作负载特征分为 memory-bound（内存受限）和 compute-bound（计算受限）：(1) **Memory-bound**：瓶颈在于 HBM 内存带宽而非计算能力，典型场景为 decode stage（batch=1, 每步仅计算 1 token）。此时 arithmetic intensity（FLOPs per byte loaded）极低，GPU 的 FLOPS 利用率 <1%；(2) **Compute-bound**：瓶颈在于计算能力，典型场景为 prefill stage（所有 prompt tokens 并行处理）或大 batch decode。Block Transformer 论文将此分类应用到推理优化设计：Token Decoder 的 decode 是 memory-bound → 可安全增加 prefix length 或模型容量（更多 FLOPs）而几乎不增加延迟，因为瓶颈在内存而非计算。Block Decoder 的 prefill 是 compute-bound → 减少输入 block 数直接降低计算量和延迟。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Arithmetic intensity = FLOPs / Bytes_loaded_from_HBM

Prefill stage (compute-bound):
  FLOPs: O(L² × D) attention + O(L × D²) FFN ≈ 14 GFLOPs/token × L tokens
  Bytes: O(L × D) KV cache write + O(D²) params ≈ 604MB params + 3GB KV
  Intensity: high → GPU compute units fully utilized → MFU ~50-70%

Decode stage (memory-bound, B=1):
  FLOPs: ~14 GFLOPs (single token)  
  Bytes: ~604MB params + L × 2 × D_heads × ... ≈ 604MB + KV cache
  Intensity: ~14e9 / 600e6 ≈ 23 FLOPs/Byte → MFU <1%
  → Adding more FLOPs (larger model, longer prefix) has minimal latency impact
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block Transformer 利用了这个特性：通过将 Token Decoder 的 KV cache 从 L 降至 LB（~4），大幅降低 decode 阶段的内存访问，使更多 HBM 带宽可用于参数读取。由于参数 IO 可通过 batching 摊薄，Block Transformer 的高 batch 能力转化为大幅吞吐量提升（10-25×）。理解 memory-bound vs compute-bound 的转换点是设计 efficient inference 架构的核心。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)
