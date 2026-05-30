## LMDeploy (LLM Compression, Deployment, and Serving Toolkit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LMDeploy 是由上海人工智能实验室（InternLM 团队）开发的开源 LLM 压缩、部署和 serving 工具包，原生支持 PagedAttention + FlashAttention，并提供高效的量化 kernel——4-bit KV cache 量化性能优于 vLLM（BentoML benchmark 验证）。其 TurboMind engine 是 C++ 实现的高性能推理后端，支持 continuous batching、tensor parallelism、weight-only 量化和 KV cache 量化。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**LMDeploy 推理全流程**：
```
[Client] → HTTP POST /v1/chat/completions (prompt)
  → [LMDeploy API Server]
     → Tokenizer: prompt → token IDs
     → [Request Router (可选)]
        → Throughput Predictor: profiled table → T_decode, T_prefill
        → Length Predictor: LongFormer → response_length_est
        → GPU选择: argmin(prefill_time + decode_time)
     → [TurboMind Engine on selected GPU]
        → Prefill: FlashAttention + PagedAttention
          → KV cache 写入 page blocks (FP16)
        → Decode loop:
          → 若压缩: quantize/dequantize KV (INT4) → FlashAttention
          → Attention output → lm_head → sample → next token
        → Response: detokenize → text
[Client] ← JSON response
```

术语一般如何实现？如何使用？

开源：https://github.com/InternLM/lmdeploy。论文使用 LMDeploy v6.0.1。安装 `pip install lmdeploy`，serving: `lmdeploy serve api_server <model>`。论文选择 LMDeploy 而非 vLLM 的原因：(1) 4-bit quantization kernel 更高效；(2) KV cache 压缩算法开发接口更友好；(3) 论文 Observation 2 之外的核心结论不依赖于特定 serving engine。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
