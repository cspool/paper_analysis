## Decoupled Rotary Position Encoding (解耦旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

解耦 RoPE（Decoupled Rotary Position Encoding）是 MLA 和 GLA 中处理位置信息的关键技术。传统 RoPE（Su et al., 2023）直接对每个 head 的完整 K/Q 维度做旋转变换。但 MLA/GLA 的 latent compression 导致 K 被压缩为低维 latent vector——若直接在 latent 上加 RoPE，则 weight absorption 技巧（将 up-projection 矩阵吸收进 Q/O 投影）失效，因 RoPE 旋转矩阵与 up-projection 矩阵不满足交换律。

解耦方案：(1) Key 拆分为内容部分（来自 latent compression，不加 RoPE）和位置部分（额外单独投影 + RoPE，维度 d_R 通常远小于 d_h）；(2) Q 同样拆分；(3) Attention score 由两部分点积求和：content QK^T + positional Q_R K_R^T。位置部分 K_R 通常跨所有 head 共享（单头），进一步减少 KV cache。

从算法pipeline角度拆解，给出具体例子。

```
# MLA/GLA 的解耦 RoPE 张量计算
X ∈ R^{B×L×d}

# Content Key（来自 latent，不加 RoPE，可做 weight absorption）
K_C = c^{KV} @ W^{UK}         # [B, L, h_kv, d_nope]

# Positional Key（额外投影 + RoPE，通常单头跨 head 共享）
K_R = X @ W^{KR}              # [B, L, 1, d_R]
K_R = apply_rope(K_R)

# Query 同样拆分
Q_C = X @ W^{QC}              # [B, L, h_q, d_nope]
Q_R = X @ W^{QR}              # [B, L, h_q, d_R]
Q_R = apply_rope(Q_R)

# 分开计算后求和
attn = (Q_C @ K_C^T + Q_R @ K_R^T) / sqrt(d_nope + d_R)
```

**GTA 的变体应用**：GTA 不使用 latent compression，也不需 weight absorption，但采用类似的 partial RoPE——仅 half head dimension（d_h/2）加 RoPE 作为位置 key，剩余 half 作为无位置编码的内容 key（K_NoPE），后者与 value 共享 tied KV state。

术语一般如何实现？如何使用？

FlashMLA 和 GLA kernel 中，Positional（RoPE）和 Content（non-RoPE）部分的 attention 分别在 Tensor Cores 上计算后通过 FMA 合并。d_R 典型取 32（约 25-50% d_h），在位置信息保真度和 KV cache 开销间平衡。Cohere 的 Command-R 和 Llama 4 进一步通过仅在部分层应用 RoPE 来减少 d_R 的 KV cache 开销。

**MTLA 的 Temporal Compression of RoPE Keys**：MTLA 进一步将 decoupled RoPE keys 沿 temporal 维压缩——对每 s 个相邻 token 仅保留最新的 RoPE key 到 cache。推理时 j-th slot 的 RoPE key 缓存更新策略：若 i%s==1 则追加 k_i^R；否则用 k_i^R 覆盖当前 slot（ĥ_j^R = k_i^R）。训练时原始 K^R 配合 stride-aware causal mask 直接参与 attention 计算。此压缩不增加额外参数量，进一步减少 64/layer（per d_R=32, s=2, bf16）的 KV cache 开销。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding
- Multi-head_Temporal_Latent_Attention
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（首次提出 Decoupled RoPE 作为 MLA 架构的关键组件）

**补充（来自 X-EcoMLA）**：X-EcoMLA 采用统一共享 RoPE Key 设计（类似 DeepSeek MLA），即所有 attention head 共享单一 K^R ∈ R^{d_r} 向量（而非 per-head 各分配 d_r/n_h 维）。这在不同 head 数下提供 n_h× 的位置编码容量优势——以 8-head 模型为例，per-head RoPE 设计下每 head 仅获 32/8=4 维位置编码，而 X-EcoMLA 的共享设计每 head 获完整 32 维。X-EcoMLA 通过对比 MHA2MLA（per-head RoPE）展示了这一设计对极端 KV 压缩下性能保持的关键作用。K^R 的初始化使用所有 KV head 的 W^K 的平均值（W_K_avg = mean(W_K.view(d, n_kv, d_h), dim=1)），取最后 d_r 列。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---
