## KV Cache Quantization（KV Cache 量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Quantization 是将高精度（通常 FP16/BF16）存储的 Key 和 Value 张量压缩到低比特整数表示（如 INT4、INT2）的技术，以减少 KV cache 的内存占用。与权重量化不同，KV cache 量化是在**推理时动态执行**的——每生成新 token 后，新的 K/V 需要即时量化存储，attention 计算前需反量化回浮点精度。

关键挑战：
1. **Outlier 问题**：K tensors 中存在显著的 channel-wise outliers（少数 channel 值远大于其他 channel），如果使用 per-tensor 或 per-token 量化，这些 outliers 会主导 scale factor，导致大量有效精度损失。
2. **分布不对称**：K 和 V 具有不同的分布特性——K 的 channel-wise outlier 更显著，V 分布相对均匀。
3. **动态范围变化**：随序列增长，KV cache 中的数值分布可能漂移。

XStreamVGGT 通过发现 StreamVGGT 中 K 的 channel-wise outlier 和 V 的相对均匀分布，提出了 per-channel K + per-token V 的维度自适应量化方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中的量化流程（基于 KIVI，INT4，group size 64）：

```
# 量化存储（pruning 后对保留的 cache 执行）
# 对 Key 使用 per-channel 量化
for c in range(C):  # 每个 channel 独立
    K_c = Cache.K[:, c]                          # shape: T_keep
    s_c = (max(K_c) - min(K_c)) / (2^4 - 1)      # scale (INT4)
    z_c = round(-min(K_c) / s_c)                  # zero-point
    K̂_c = clamp(round(K_c / s_c) + z_c, 0, 15)   # 量化值

# 对 Value 使用 per-token 量化
for i in range(T_keep):  # 每个 token 独立
    V_i = Cache.V[i, :]
    s_i = (max(V_i) - min(V_i)) / (2^4 - 1)
    z_i = round(-min(V_i) / s_i)
    V̂_i = clamp(round(V_i / s_i) + z_i, 0, 15)

# Attention 计算时反量化
K_deq = (K̂ - z_c) * s_c     # INT4 → FP16
V_deq = (V̂ - z_i) * s_i
Out = FlashAttn(Q_t, K_deq, V_deq)
```

内存节省（以 StreamVGGT 为例）：FP16 每元素 2 bytes → INT4 每元素 0.5 bytes（4× 压缩），加上 scale 和 zero-point metadata（per-channel K: C × 2 × 2 bytes ≈ 小开销），总计约 4× 内存减少。配合 pruning（cache 从无界到 2K tokens），总计 4.42× 内存减少。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
常用实现：
- **KIVI** (Liu et al., 2024)：asymmetric uniform quantization，支持 INT2/INT4，per-channel + group-wise 量化，tuning-free。本论文采用的方法。
- **KVQuant** (Hooper et al., 2024)：per-channel + per-group with dense-and-sparse，支持 3-bit/4-bit。
- **GEAR** (Kang et al., 2024)：codebook-based quantization with residual compensation。
- **Atom** (Zhao et al., 2025)：hardware-efficient INT4 with per-token group quantization。
- **RotateKV**：通过 Hadamard rotation 平滑 outlier 后执行 per-channel 量化。

PyTorch 中通过 `torch.quantize_per_channel` / `torch.quantize_per_tensor` 或自定义 kernel。vLLM 通过 `--kv-cache-dtype fp8` 或 `--quantization kv-cache` 自动启用。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression
