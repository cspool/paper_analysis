## KIVI (Tuning-Free Asymmetric 2-bit KV Cache Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KIVI（Liu et al., 2024e）是一种免调参（tuning-free）的 KV cache 量化算法，核心设计是非对称量化策略：对 **Key 使用 per-channel 量化**，对 **Value 使用 per-token 量化**。这一设计基于以下观察：Key 张量的不同 channel 之间数值分布差异大（某些 channel 的数值范围比其他 channel 宽很多），因此需要 per-channel 量化为每个 channel 独立计算 scale/zero-point；Value 张量的不同 token 之间数值分布差异大（被 attention 高度关注的 token 的 value 数值范围更宽），因此需要 per-token 量化为每个 token 独立计算 scale/zero-point。KIVI 使用 2-bit 或 4-bit 精度（可配置），并保留最近 $R$ 个 token 为全精度（FP16）以保护近期上下文精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KIVI 量化/反量化流程**：
```
# 超参数：group_size=G=32 (key), residual_length=R=128

# Key 量化（per-channel）：
# K_cache shape: [num_tokens, num_kv_heads, head_dim]
# Q: 每 channel（head_dim 维度）独立计算 min/max
for c in range(head_dim):
    channel_data = K_cache[:, :, c]           # [num_tokens, num_kv_heads]
    max_val = channel_data.max()
    min_val = channel_data.min()
    scale[c] = (max_val - min_val) / (2^bits - 1)
    zero[c] = round(-min_val / scale[c])
    K_int4[:, :, c] = quantize(channel_data, scale[c], zero[c])

# Value 量化（per-token）：
# Q: 每 token 独立计算 min/max
for t in range(num_tokens - R):    # 跳过最近 R 个 token（保留 FP16）
    token_data = V_cache[t, :, :]           # [num_kv_heads, head_dim]
    max_val = token_data.max()
    min_val = token_data.min()
    scale[t] = (max_val - min_val) / (2^bits - 1)
    zero[t] = round(-min_val / scale[t])
    V_int4[t, :, :] = quantize(token_data, scale[t], zero[t])

# 注意：最近 R=128 个 token 的 KV 保持 FP16 不量化
# 这是为了在近期上下文上保持全精度 Attention

# Attention 计算时反量化：
K_deq = dequantize_per_channel(K_int4, scale_K, zero_K)
V_deq = dequantize_per_token(V_int4, scale_V, zero_V)
# 前 R 个 FP16 token 和其余 INT4 token 拼接
K_full = concat([K_deq, K_cache_fp16[-R:]])
V_full = concat([V_deq, V_cache_fp16[-R:]])

scores = Q @ K_full^T / sqrt(d_head)
output = softmax(scores) @ V_full
```

**Annotations**: Per-channel 量化意味着同一 channel 的所有 token 共享 scale/zero，适合 channel 间分布差异大的 Key。Per-token 量化意味着同一 token 的所有 channel 共享 scale/zero，适合 token 间分布差异大的 Value。Residual length $R$ 保留最近 token 为全精度，兼顾近期上下文质量。

术语一般如何实现？如何使用？

KIVI 开源：https://github.com/jy-yuan/KIVI。关键参数：group_size $G$（key per-channel 量化时 channel 分组大小，默认 32）、residual length $R$（保留 FP16 的最近 token 数，默认 128）、bit-width（2-bit 或 4-bit）。KIVI 作者曾指出 integrate into vLLM 存在困难（GitHub issue #4），主要因为 window-based quantization（保留最近 R token 为 FP16 + 其余为 INT4）与 PagedAttention 的 fixed-type page block 管理不兼容。论文 "Rethinking KV Cache Compression" 将 KIVI 集成到 LMDeploy v6.0.1 中评估：LLaMA-7B TP=1 prefill 1.06×（略超 FP16 baseline），但 decode 仅 0.98×（TP=1）到 0.88×（TP=2），说明量化带来的 memory reduction 在实际 serving 框架中的吞吐收益有限。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
