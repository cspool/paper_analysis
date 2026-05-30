## Prefill / Decode Stages (LLM Inference)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
自回归 LLM 推理分为两个阶段：(1) **Prefill stage（预填充阶段）**：处理输入 prompt 所有 token，一次性并行计算所有 token 的 KV cache 并存储于 GPU 显存。此阶段为 compute-bound——所有 token 并行处理，计算需求远超内存带宽限制；(2) **Decode stage（解码阶段）**：生成输出 token 一个接一个。每步仅计算一个 token，但需从显存加载全部先前 token 的 KV cache（用于 self-attention）和模型参数。此阶段为 memory-bound——仅处理单个 token，内存 IO（参数+KV cache）成为主要瓶颈。生产系统中通常使用 batching 摊薄参数 IO，但在充足上下文长度下 KV cache IO 成为主导瓶颈（因 batch 增大后 KV cache 总量也增大）。Block Transformer 通过在 Token Decoder 中跳过 prefill（除最后一个 block）和在 Block Decoder 中降低 decode KV cache IO 来系统性缓解两个阶段的开销。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 vanilla Pythia 302M, L=2048, B=16 为例的推理阶段：
```
Prefill stage (compute-bound):
  prompt: "Hello, how are" (2048 tokens padded)
  → 24层 transformer, 每层:
    Q,K,V = Linear(input)             # [B, L, D] → [B, L, D]
    attn = Softmax(Q·K^T / √d) · V   # [B, L, D], 所有token并行
    cache K, V → GPU memory           # 24层 × 2048 × 2(KV) × 1024 dim × ... 
  → 输出第一个token: "you"
  Latency dominated by compute (FLOPs)

Decode stage (memory-bound):
  生成: "you" → "doing" → "today" → ...
  → 每层: Q = Linear(1 new token)
          K, V = Load full KV cache from memory (L tokens)
          attn = softmax(Q · K^T / √d) · V
          append new K,V → KV cache extends
  → 每token需读: 604MB params + up to 3.2GB KV cache (B=16)
  Latency dominated by HBM bandwidth
  MFU typically ~1% (most of the time waiting for memory)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
现代 LLM serving 系统（vLLM、SGLang、TensorRT-LLM）通过 PagedAttention 和 continuous batching 优化这两个阶段。Block Transformer 从架构层面而非系统层面解决瓶颈：Token Decoder 跳过 prefill（无需预填充 prompt KV 即可开始生成）, Block Decoder 降低 decode KV cache IO。FlashAttention/FlashDecoding 作为 kernel 优化也可应用到两者的 attention 计算中（Block Transformer 验证了 FlashDecoding 兼容性）。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---
