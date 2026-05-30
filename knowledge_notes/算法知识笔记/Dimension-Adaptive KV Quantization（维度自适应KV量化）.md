## Dimension-Adaptive KV Quantization（维度自适应KV量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dimension-Adaptive KV Quantization 是 XStreamVGGT 提出的针对视觉 Transformer 中 KV cache 的量化策略。核心思想：根据 K 和 V 张量的不同分布特性选择不同的量化粒度（quantization granularity），而非使用统一的量化方案。具体地：(1) Key tensors 使用 **per-channel 量化**（每个 channel 独立计算量化参数），以应对 Key 中显著的 channel-wise outliers；(2) Value tensors 使用 **per-token 量化**（每个 token 独立计算量化参数），因为 Value 分布更均匀且 per-channel 量化对 Value 的 MSE 改善不大。

决策依据来自对 StreamVGGT 的 KV 分布分析：
- K 的 per-channel 量化 INT4 MSE：$9.181 \times 10^{-3}$（per-token: $5.183 \times 10^{-2}$，改善 5.6×）
- V 的 per-channel vs per-token 量化 INT4 MSE：$4.704 \times 10^{-4}$ vs $5.035 \times 10^{-4}$（差异极小）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
维度自适应量化的执行流程：

```
# 输入: K ∈ R^{T×C} (Key, T=tokens, C=channels)
#        V ∈ R^{T×C} (Value)

# === Key: per-channel quantization ===
for c in range(C):
    K_c = K[:, c]                                 # shape: T (所有 token 的 channel c)
    s_K[c] = (max(K_c) - min(K_c)) / (2^b - 1)   # per-channel scale
    z_K[c] = round(-min(K_c) / s_K[c])            # per-channel zero-point
    K̂_c = clamp(round(K_c / s_K[c]) + z_K[c], 0, 2^b - 1)

# === Value: per-token quantization ===
for t in range(T):
    V_t = V[t, :]                                 # shape: C (token t 的所有 channel)
    s_V[t] = (max(V_t) - min(V_t)) / (2^b - 1)   # per-token scale
    z_V[t] = round(-min(V_t) / s_V[t])            # per-token zero-point
    V̂_t = clamp(round(V_t / s_V[t]) + z_V[t], 0, 2^b - 1)

# 存储: K̂ (4-bit), s_K (FP16 × C), z_K (FP16 × C)
#       V̂ (4-bit), s_V (FP16 × T), z_V (FP16 × T)

# Metadata overhead: per-channel K: 2C × 2 bytes; per-token V: 2T × 2 bytes
# 典型值 (C=1024, T=2000): K overhead = 4KB, V overhead = 8KB → negligible
```

与 LLM 量化策略的区别：流式视觉 Transformer 以帧为单位批量产生 token → per-channel 量化对 K 天然友好（大量 tokens 共享同一 channel 统计量，scale 稳定）；LLM decode 每步仅 1 个 token → per-channel 量化需要跨 step 累积足够 tokens 才稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 KIVI 框架实现：`kivi/quantization.py` 中的 `KIVIQuantizer` 支持 per-channel 或 per-token 配置。PyTorch 原生实现：`torch.quantize_per_channel(K, scales, zero_points, axis=1, dtype=torch.qint8)`（axis=1 表示 per-channel along token dim，即 channel-wise along feature dim）。量化紧耦合在 pruning 之后：先 pruning 减少 token 数（T → L_max），再对精简后的 cache 量化。XStreamVGGT 开源代码：https://github.com/ywh187/XStreamVGGT/。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression
