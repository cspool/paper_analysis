## KV Cache IO Bottleneck

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache IO Bottleneck 是自回归 LLM decode 阶段的核心性能限制：每步 decode 生成一个 token 时，需从 HBM 加载所有先前 token 的 key/value 状态用于 self-attention。KV cache 总量 = $2 \times n_{layers} \times L \times B \times D_{head} \times n_{heads}$（字节），随序列长度 L 线性增长。decode 阶段的 arithmetic intensity（计算量/内存访问量）极低（约 1 FLOP/Byte），导致 GPU 的 HBM 带宽成为瓶颈，计算单元利用率（MFU）通常仅 ~1%。Block Transformer 论文通过分析 (Llama 7B, 2048 tokens, batch=16, KV cache=16GB) 量化了该问题：HBM 带宽达参数量级，但 compute throughput (FLOP/s) 是 HBM 带宽的 2-3 个数量级，且差距呈指数扩大（"memory wall"）。Block Transformer 通过两个机制缓解：(1) Block Decoder 减少 block 数 → KV cache 大小 ↓LB 倍；(2) Token Decoder 仅用 local KV cache (LB tokens) → KV cache IO ↓L/LB 倍，最终将 KV cache IO 从 $O(L^2)$ 降至 $O(L \cdot L_B)$。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Vanilla vs Block Transformer KV cache IO 对比 (L=2048, LB=4, D=1024, FP16)：
```
Vanilla Transformer:
  每层 KV cache = 2048 tokens × 2(KV) × 1024 dim = 4MB per layer
  32 layers: 128MB KV cache total
  Per decode step IO: 128MB KV + 604MB params = 732MB
  KV cache IO per full decode (L tokens): L × L × ... = O(L²)
  B=16: KV cache = 16GB, often exceeds GPU memory → batch constrained

Block Transformer:
  Block Decoder: 512 blocks × 2(KV) × 1024 dim = 1MB/layer (↓4×)
  Token Decoder: 6 tokens × 2(KV) × 1024 dim = 12KB/layer (↓341×)
  KV cache total = 512 × (block decoder layers) + 6 × (token decoder layers)
  Token Decoder prefill: 0 (completely skipped)
  Decode KV cache IO: O(L · LB) instead of O(L²)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解 KV cache IO 瓶颈的方法分为两类：(1) 系统层面——PagedAttention（vLLM）、prefix caching（SGLang）、KV cache offloading（FlexGen）、KV cache quantization；(2) 架构层面——Multi-Query Attention (MQA)、Grouped-Query Attention (GQA)、Multi-head Latent Attention (MLA)、Sliding Window Attention (SWA)、Block Transformer 的 global-to-local 架构。Block Transformer 可与 FlashAttention 兼容，后者进一步减少 attention 计算的内存访问次数。YOCO 通过 decoder-decoder 架构将 KV cache 从 O(L×N×D) 降至 O(N×D)（共享单层全局 KV cache），从架构层面根本性地减少缓存总量。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

---
