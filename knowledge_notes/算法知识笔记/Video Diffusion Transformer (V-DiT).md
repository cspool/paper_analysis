## Video Diffusion Transformer (V-DiT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Diffusion Transformer (V-DiT) 是将 Diffusion Transformer (DiT) 架构扩展到视频生成任务的模型。DiT (Peebles & Xie, ICCV 2023) 用 Transformer 替代传统 U-Net 作为扩散模型的去噪骨干网络。在视频生成中，DiT 的 latent 表示 Z∈R^{n×d}，其中 n = s × t（s 为空间 token 数，t 为时序 token 数，即 t 帧）。与图像 DiT (I-DiT, n=s) 相比，V-DiT 通过额外的时序维度处理多帧视频，token 数量是 I-DiT 的 t 倍，表达能力更强，信息密度更高。代表性 V-DiT 包括 Open-SORA（基于 PixArt-α 架构，t 帧 latent 并行去噪）和 Latte（在 UCF-101 上训练的 class-conditioned 视频 DiT）。Q-VDiT 首次针对 V-DiT 提出专门的量化方案，因为视频生成的高信息密度使得直接应用图像量化方法会导致严重的信息丢失和帧间不连贯。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
V-DiT 的推理 pipeline（以 Open-SORA 为例）：
```
# 输入: text prompt P, latent shape (t, c, h, w)
# Step 1: 文本编码
text_emb = T5_encoder(P)                     # T5-XXL text encoder

# Step 2: 初始化噪声
Z_T = randn(t, c, h, w)                       # t 帧高斯噪声

# Step 3: 去噪循环 (100-step DDIM)
for step in range(T, 0, -1):
    # 拼合时空 token: flatten spatial dims + concat temporal
    Z_flat = Z_cur.reshape(t*s, d)            # n=s×t token

    # DiT forward: 在每个 timestep t 预测噪声
    noise_pred = DiT(Z_flat, timestep, text_emb)
    # DiT 由交替的 self-attention (跨所有 s×t token) 和 FFN 组成
    # 时序信息通过时空 attention 隐式建模

    # DDIM update
    Z_cur = ddim_step(Z_cur, noise_pred, timestep)

# Step 4: VAE decode (帧独立)
video = VAE_decoder(Z_0)                     # 解码为像素空间视频
```
量化时主要挑战：V-DiT 的 token 数远大于 I-DiT（×t），量化误差在 n=s×t 个 token 上累积传播，加上视频帧间有强时空语义关联，帧间不一致的风险更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
V-DiT 的实现基于 HuggingFace Diffusers 或自定义框架（Open-SORA 使用自己的实现），推理使用标准 Transformer forward（self-attention + cross-attention + FFN），通常配备 CFG (Classifier-Free Guidance) 和 DDIM/DDPM sampler。Q-VDiT 量化 V-DiT 时仅量化线性层权重（channel-wise）和激活（token-wise dynamic），保持 attention 和 LayerNorm 为 FP16。Open-SORA 使用 100-step DDIM + CFG=4.0，Latte 使用 20-step DDIM + CFG=7.0。量化校准从 10 个 prompt 的 50 个去噪步采样校准数据。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---
