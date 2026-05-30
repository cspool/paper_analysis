## Grouped Latent Attention (GLA / 分组潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped Latent Attention (GLA) 是 Zadouri、Strauss 和 Dao（Princeton, 2025）提出的并行友好 latent attention 变体，是对 MLA 的核心改进。GLA 将 MLA 的单头 latent compression 扩展为**多头** latent compression（h_c 个 latent head，每 head d_c=2d_h），使 latent head 可在 TP rank 间分片，解决 MLA 的 KV cache 跨设备全复制问题。

MLA 的单头 latent（d_c=4d_h）因 TP 按 head dimension 切分 up-projection 矩阵，每 rank 必须持有完整 latent——TP 不减少 MLA 的 per-device KV cache。GLA 通过多 latent head 分组设计：每 latent head 仅服务于一组 query head，该组的上投影矩阵完整存于对应 rank，无需复制 latent。

以 GLA-2（h_c=2, d_c=2d_h）为例：总 KV cache = 4d_h（与 MLA 相同），但 TP=2 时每 device 仅 2d_h（MLA 仍为 4d_h）。算术强度约 2gq（双倍于 GQA），与 MQA 的 h_q 相当但质量远超。

从算法pipeline角度拆解，给出具体例子。

```
# GLA-2: hq=16, h_c=2, gq=8, dh=128, d_c=256, d_R=32

# === 训练时: Down + Up ===
c_0^{KV} = X @ W^{DKV}_0   # [B, L, 256]
c_1^{KV} = X @ W^{DKV}_1   # [B, L, 256]
K_0 = c_0^{KV} @ W^{UK}_0   # per-group K
V_0 = c_0^{KV} @ W^{UV}_0   # per-group V

# === 解码时: Weight Absorption ===
# W^{UK} 吸收进 W^Q, W^{UV} 吸收进 W^O
Q_0 = X @ W^Q_absorbed_0   # [B, 1, 8, 256]
Q_1 = X @ W^Q_absorbed_1   # [B, 1, 8, 256]
O_0 = softmax(Q_0 @ (c_0^{KV})^T / 16) @ c_0^{KV}
O_1 = softmax(Q_1 @ (c_1^{KV})^T / 16) @ c_1^{KV}

# === 分布式 (TP=2) ===
# Rank 0: c_0^{KV}, Q_0, W^{VO}_0 → O_0 @ W^{VO}_0
# Rank 1: c_1^{KV}, Q_1, W^{VO}_1 → O_1 @ W^{VO}_1
O = AllReduce(O_0 @ W^{VO}_0 + O_1 @ W^{VO}_1)
```

**Serving benchmark 关键结果（DeepSeek-Coder-V2 236B FP8, 8×H100）**：
- GLA-8 TP=8 vs MLA TP=8（64 并发, 8K/4K）：throughput 1461 vs 859 tok/s（+70%），E2E latency 179s vs 381s（-53%）
- GLA-8 TP=8 vs MLA TP=2+DP=4（131K/4K 不平衡负载）：throughput 100 vs 37 tok/s（+2.7×）
- GLA kernel L_q=2（推测解码）：2× faster than FlashMLA

术语一般如何实现？如何使用？

开源实现：https://github.com/Dao-AILab/grouped-latent-attention。包含完整 CUDA kernel（warp specialization + software pipelining + distributed offset calculation）、PyTorch 模型定义和 SGLang serving 集成。

适用场景：(a) 大规模 distributed inference（TP≥2），每 device KV cache 是核心瓶颈；(b) 推测解码（L_q>1），GLA 算术强度更高；(c) 混合负载（不均匀序列长度），避免 DP straggler。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---
