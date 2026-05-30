## KV Cache Bottleneck & Bottleneck Shift in LLM Inference（LLM推理中的KV缓存瓶颈与瓶颈转移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Bottleneck 指在长上下文 LLM 推理中，KV cache 的内存占用和内存带宽成为限制性能的主要因素。随着 sequence length S 和 batch size B 同时增大，KV cache 总大小 = B × S × n_layers × 2 × n_kv_heads × d_head（bf16 格式），KV cache 的加载时间随 B 和 S 线性增长，最终超过模型参数加载和计算时间，使推理从 compute-bound 转向 memory-bound。

Bottleneck Shift 描述了推理瓶颈随 (B, S) 变化的转移过程：短序列时，线性层（MLP + query/key/value projection）计算量随 B 增大 → compute-bound；长序列时，KV cache size 膨胀超过 parameter size → memory-bound。MagicDec 利用这一瓶颈转移来解释为何 SD 在长序列大 batch 下重新有效——当 KV loading 是瓶颈时，验证成本 T_V 的主要部分（KV loading）与目标解码 T_T 相同，T_V/T_T ≈ 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Bottleneck 判定（基于 roofline model）
# 对每次 decode step:

# Compute time:
FLOPs_per_step = B * (2 * d_model^2 + 2 * S * d_model)  # MLP + Attention FLOPs
compute_time = FLOPs_per_step / GPU_peak_FLOPS

# Memory time:
# Model params loading: n_layers * (4 * d_model^2) bytes (bf16)
# KV cache loading: B * S * n_layers * 2 * n_kv_heads * d_head * 2 bytes (bf16)
total_bytes = param_bytes + kv_cache_bytes
memory_time = total_bytes / GPU_memory_BW

if memory_time > compute_time:
    bottleneck = "memory-bound"   # KV cache 是瓶颈
    # T_V/T_T ≈ 1, SD 有效
else:
    bottleneck = "compute-bound"   # 线性层计算是瓶颈
    # T_V/T_T > 1, SD 大 batch 下失效

# S_inflection: 对于给定 B, 使得 memory_time == compute_time 的 S
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Roofline model 判断 bottleneck：计算 arithmetic intensity = FLOPs / bytes_transferred，若低于 GPU 的 FLOPS/BW 拐点则为 memory-bound。MagicDec Figure 3c 展示了 LLaMA-2-7B 和 LLaMA-3.1-8B 在不同 S 下的 arithmetic intensity。实操：通过 profiling 测量不同 (B, S) 下的 compute time vs memory loading time 比例（Figure 1a），判断是否进入 memory-bound regime。MagicDec 利用 bottleneck shift 决定何时对当前 (B, S) 启用 SD——仅在 memory-bound（S > S_inflection）时启用 SD 可实现 speedup > 1。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
