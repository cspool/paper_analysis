## Hardware-Efficient Attention for Fast Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  提出两种新的硬件高效注意力变体：(1) **GTA (Grouped-Tied Attention)**：将 key 和 value 的投影参数绑定为单一的 *tied KV* 状态，对每组 query head 共享一个 distinct tied KV head。Key 路径仅使用 tied KV 的前半维度（不加 RoPE），另一半 key 维度来自独立的单头 RoPE 投影并广播到组内所有 head；Value 路径使用 tied KV 的完整维度。KV cache 大小相对于同 group 数的 GQA 减半、算术强度翻倍。(2) **GLA (Grouped Latent Attention)**：将 MLA 的单头 latent 压缩扩展为多 latent head，每个 latent head 维度 d_c = 2d_h（MLA 为 4d_h），latent head 可在 TP rank 间分片，避免 MLA 的 latent 在全设备复制的问题。解码时吸收低秩 up-projection 矩阵，每个 latent head 仅服务于其 group 内的 query head。实验比较：在 Small (183M)、Medium (433M)、Large (876M)、XL (1.471B) 四个 GPT-3 规模上与 MHA、MQA、GQA-4、MLA 对比 perplexity（FineWeb-Edu + 5 数据集平均）和 downstream accuracy（SciQ, OpenBookQA, ARC-Easy, HellaSwag, PIQA, WinoGrande, MMLU 共 7 benchmark）。

- 硬件平台是什么，配置是什么。
  训练：论文未明确说明具体 GPU 型号和数量。推理 kernel benchmark：NVIDIA H100 80GB SXM5 GPU（BF16 峰值 989 TFLOPS/s，HBM 带宽 3350 GB/s）。多 GPU serving benchmark：8× H100 80GB GPU，NVLink 互联，使用 DeepSeek-Coder-V2 Base（236B 参数，21B active）FP8 量化模型。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT-3 配置 + Llama 3 架构（SwiGLU FFN, RMSNorm, RoPE），四个规模——Small (183M, 12 layers, d=768, hq=12, dh=64)、Medium (433M, 24 layers, d=1024, hq=16, dh=64)、Large (876M, 24 layers, d=1536, hq=16, dh=96)、XL (1.471B, 24 layers, d=2048, hq=16, dh=128)。各 variant 通过加宽 FFN 匹配 MHA 的参数总量。训练数据：FineWeb-Edu-100B（small 25B tokens，其余 50B tokens），Llama 3 tokenizer（vocab 128K），AdamW (β1=0.9,β2=0.95, weight decay=0.1, grad clip=1.0)，cosine LR decay to 1%。Perplexity 评估：FineWeb-Edu validation、Cosmopedia、RedPajama v1 C4、RedPajama v1 Wikipedia、Pile（各 100M tokens）。Downstream benchmark：SciQ、OpenBookQA、ARC-Easy、HellaSwag、PIQA、WinoGrande、MMLU（zero-shot）。Serving benchmark 模型：DeepSeek-Coder-V2 Base (236B, 21B active, FP8)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Dao-AILab/grouped-latent-attention（permissive license）。

  **GTA 算法 pipeline**：
  ```
  # 输入: hidden states X ∈ R^{B×L×d}
  # hq query heads, hkv KV heads (group size gq = hq/hkv), dh per-head dim

  # 1. Q 投影
  Q = X @ W^Q          # [B, L, hq, dh]

  # 2. Tied KV 投影（单一投影矩阵替代 W^K 和 W^V）
  KV = X @ W^{KV}      # [B, L, hkv, dh] — 单一 tied state

  # 3. 构造 K 和 V
  V = KV               # [B, L, hkv, dh] — 完整维度用于 value
  K_NoPE = KV[:,:,:,:dh/2]      # 前半维度，不加 RoPE
  K_RoPE = X @ W^{K_RoPE}       # [B, L, 1, dh/2] — 单头 RoPE key
  K_RoPE = apply_rope(K_RoPE)
  K_RoPE_bcast = broadcast(K_RoPE, hkv)  # 广播到所有 KV head
  K = concat([K_NoPE, K_RoPE_bcast], dim=-1)  # [B, L, hkv, dh]

  # 4. Attention (GQA-style grouping)
  for each group g of gq query heads:
      attn[g] = softmax(Q_g @ K_g^T / sqrt(dh)) @ V_g

  # 5. Output projection
  O = concat(all groups) @ W^O
  ```
  KV cache per token: hkv × dh × sizeof(dtype)（vs GQA: hkv × 2 × dh），算术强度 ≈ 2gq（vs GQA: ≈ gq）。结合了低秩 key 的洞察（仅部分维度需要 RoPE）、KV 状态共享（key 和 value 源自同一 state），以及 GQA 的分组并行设计。

  **GLA 算法 pipeline (GLA-2, hc=2 latent heads)**：
  ```
  # hc=2 latent heads, dc=2dh, gq = hq/hc

  # --- 训练时: down+up projection ---
  c_0^{KV} = X @ W^{DKV}_0   # [B, L, 2dh]
  c_1^{KV} = X @ W^{DKV}_1   # [B, L, 2dh]
  K_0 = c_0^{KV} @ W^{UK}_0   # [B, L, gq*dh]
  V_0 = c_0^{KV} @ W^{UV}_0
  # 类似地 K_1, V_1

  # --- 解码时: weight absorption ---
  # W^{UK} 被吸收进 W^Q, W^{UV} 被吸收进 W^O
  # 直接对 latent c^{KV} 计算 attention:
  Q_0 ∈ R^{B×1×gq×(2dh)}, Q_1 ∈ R^{B×1×gq×(2dh)}
  O_0 = softmax(Q_0 @ c_0^{KV}^T / sqrt(2dh)) @ c_0^{KV}
  O_1 = softmax(Q_1 @ c_1^{KV}^T / sqrt(2dh)) @ c_1^{KV}

  # --- 分布式执行 (TP=2) ---
  # rank 0: c_0^{KV}, Q_0, W^{VO}_0 → partial O_0
  # rank 1: c_1^{KV}, Q_1, W^{VO}_1 → partial O_1
  O = AllReduce(O_0 @ W^{VO}_0 + O_1 @ W^{VO}_1)
  ```
  KV cache: unsharded = hc × d_c = 2 × 2dh = 4dh（与 MLA d_c=4dh 相同）。但 TP≥2 时每 device 仅 d_c = 2dh（MLA 因单头 latent 全复制仍为 4dh）。算术强度 ≈ 2gq（双倍于 GQA）。RoPE 维度 d_R = 32（默认），通过 decoupled RoPE 机制保留位置信息。

  **关键结果**：XL (1.471B) 上 GTA-4 达到 PPL 10.129（vs GQA-4 10.202）、downstream avg 60.2%（与 GQA-4 持平）；GLA-2 达到 PPL 10.218（vs MLA 10.256）、downstream avg 60.0%（vs MLA 59.1%）。GLA kernel 在 speculative decoding (L_q=2) 下比 FlashMLA 快 2×，标准 decoding (L_q=1) 快约 20%。端到端 serving：GLA-8 (TP=8) 在 64 并发下 throughput 1461 tok/s（vs MLA TP=8 的 859 tok/s，提升 70%）；在 131K 长 prefill 不平衡负载下 GLA-8 吞吐 100 tok/s（vs MLA hybrid TP+DP 的 37 tok/s，提升 2.7×）。
