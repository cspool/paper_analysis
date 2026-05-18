## Channel-wise vs Tensor-wise KV Cache Quantization Scaling

术语是什么？通过联网搜索让回答具体和精准。

Channel-wise 和 Tensor-wise 是低比特 KV cache 量化中两种不同的 scaling granularity，决定 scale factor 和 zero-point 按什么维度计算和存储。Channel-wise quantization: 沿 Key/Value tensor 的 channel 维度（hidden dimension）分组计算 scale/zero——每组 channel 有独立的量化参数。例如 K ∈ R^{L×d}, 沿 hidden dim 维度 groupsize=128 → d/128 个 scale/zero per token → total params = L × d/128 × 2。Tensor-wise quantization: 沿 token（序列）维度分组——整个 tensor 的所有元素共享一组 scale/zero→params = 1 × 2 per tensor。Channel-wise 更精细（per-channel quantization handles channel-specific value range, 尤适合处理 KV cache 中的 outlier channel），但 metadata overhead 更大。Tensor-wise metadata overhead 极小但可能放大量化误差（outlier channel 的极端值拉大整个 tensor 的 scale）。主流量化算法倾向不同：KIVI 使用 per-channel quantization（group_size=128 for K）；KVQuant 使用 per-channel with dense-and-sparse decomposition；Atom/QServe 支持 tensor-wise；Gear 和 JanusQuant 使用 per-channel。BitDecoding 同时支持两种 scaling，通过 Residual Kernel 的 residual block 内按不同维度做 reduction 实现。

从算法 pipeline 角度拆解术语：

```
// === Channel-wise vs Tensor-wise Quantization Example ===
// K ∈ R^{L×d}, L=128K tokens, d=4096

// --- Channel-wise (KIVI-style, group_size=128沿hidden dim) ---
for each token t in [0..L-1]:
    for each group g in [0..d/128-1]:  // 32 groups
        group_vals = K[t, g*128 : (g+1)*128]
        s[g] = (max(group_vals) - min(group_vals)) / (2^β - 1)
        z[g] = min(group_vals)
        K_q[g*128:(g+1)*128] = quantize(group_vals, s[g], z[g])
// metadata size: L × 32 × 2 = 64L scalars (FP16 each = 128L bytes)
// 优点: outlier channel仅影响其所在group，不污染其他channels

// --- Tensor-wise (Atom/QServe style) ---
for each token t in [0..L-1]:
    all_vals = K[t, :]  // 整行4096元素
    s = (max(all_vals) - min(all_vals)) / (2^β - 1)
    z = min(all_vals)
    K_q[t, :] = quantize(all_vals, s, z)
// metadata size: L × 2 scalars (FP16/2 = 4L bytes)
// 缺点: 1个outlier channel的极值拉大s→所有channels量化粒度变粗

// --- BitDecoding Residual Block内统一执行两种scaling ---
// Residual block: K_res ∈ R^{Nr×d}
// Channel-wise: reduction沿seq_len维度 → per-channel scale across Nr tokens
for each channel c in [0..d-1]:
    s_c = (max(K_res[:, c]) - min(K_res[:, c])) / (2^β - 1)
// Tensor-wise: reduction沿hidden维度 → per-tensor scale across d channels
for each token t in [0..Nr-1]:
    s_t = (max(K_res[t, :]) - min(K_res[t, :])) / (2^β - 1)
```

术语一般如何实现？如何使用？

Channel-wise 和 tensor-wise 的选择需权衡 accuracy 和 metadata overhead。Channel-wise 通常用于 Key cache（outlier 沿 channel 集中），tensor-wise 可用于 Value cache（distribution 更均匀）。BitDecoding 的 Residual Kernel 用 warp-level __shfl_xor_sync reduction 计算 min/max：对于 channel-wise，每组沿 seq_len 方向对 Nr 个 token 做统计 → scale/zero 按 hidden dim 输出（Nr 个 scale, d 个 zero）；对于 tensor-wise，每 token 沿 hidden dim 方向对 d 个值做统计。Scale/zero 以 half2 格式存储（两个 FP16 pack 到一个 INT32）以最小化 memory traffic。与 weight quantization 的 per-channel scaling（offline 预计算）关键区别：KV cache scaling 在 runtime 在线计算，必须高效（BitDecoding decode overhead 仅 0.008ms）。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

