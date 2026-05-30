## Video Diffusion Transformer (V-DM / VDiT, 视频扩散Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Diffusion Transformer（V-DM，也称VDiT）是将扩散模型（Diffusion Model）的去噪过程与 Transformer 架构结合的生成式模型，专门用于视频生成任务。其核心结构为：输入为文本 prompt 经编码后的 embedding 和随机噪声隐变量，经过多层 Transformer Block 进行空间-时间联合建模，逐步去噪生成视频。与 Image Diffusion Transformer (I-DM，如 DiT/FLUX) 的关键区别在于 token 维度：I-DM 的 token 数 n = s（仅空间维度），V-DM 的 token 数 n = s × t（空间 × 时间维度），其中 t 随帧率（FPS）和视频时长线性增长。例如，6 秒视频在 8 FPS 下 t = 49，每帧数千 token，总 token 数可达数万。V-DM 使用全空间-时间注意力（Full Spatial-Temporal Attention），即每个 token 关注所有空间位置和所有帧的所有 token，计算复杂度 O((s×t)²)，导致极端的显存和计算需求。代表性模型包括 CogVideoX (2B/5B, Yang et al. 2024)、HunyuanVideo (13B, Kong et al. 2024)、Open-Sora 等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 CogVideoX V-DM 的推理流程为例：
```
# V-DM 推理 Pipeline (DDIM sampling, T 步去噪)
x_T ~ N(0, I)  # 随机噪声初始化，x_T ∈ R^{n×d}, n = s×t
for timestep in [T, T-1, ..., 1]:
    # 1. 将当前 timestep 的 noisy latent 和 text embedding 送入 V-DM
    h_0 = concat(x_t, text_emb)  # 输入 embedding

    # 2. 通过 L 层 Transformer Block
    for block in V-DiT.blocks:
        # Multi-Head Self-Attention (Full Spatial-Temporal)
        # Q, K, V ∈ R^{n×d}, n = s×t
        A = softmax(Q @ K.T / sqrt(d_head))  # A ∈ R^{H×n×n}
        h = A @ V  # 每个 token 关注所有空间位置和所有帧

        # FFN (Feed-Forward Network)
        h = FFN(h)

    # 3. 预测噪声 ε_θ(x_t, t, text)
    ε_pred = output_projection(h)

    # 4. DDIM 去噪步
    x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(1-α_t) * ε_pred) / sqrt(α_t) + sqrt(1-α_{t-1}) * ε_pred

return x_0  # 去噪后的隐变量 → VAE decoder → 视频帧
```
V-DM 中 token 数 n 极大，导致：(1) 单样本显存消耗高，校准预算（样本数 N）受限；(2) Attention 计算占主导（O(n²)），全序列 sparse attention 成为可挖掘的优化空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
V-DM 的主流实现基于 PyTorch + Diffusers 库。HuggingFace Diffusers 提供 CogVideoX 的预训练模型管道：`from diffusers import CogVideoXPipeline; pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b")`。量化部署方面，S²Q-VDiT 使用 ViDiT-Q 和 FlatQuant 的 CUDA kernel 实现 INT4 weight dequantize + INT6 activation online quantize 推理。V-DM 因其极长 token 序列和全注意力开销，是当前量化压缩最具挑战性和实用价值的场景之一。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---
