## Diffusion Transformer (DiT) for Video Generation（视频生成扩散Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diffusion Transformer (DiT) 是一种将 Transformer 架构应用于扩散模型去噪过程的生成模型。在视频生成中，DiT 将视频 latent token（经 VAE 压缩的视频表示）与文本条件 token 拼接，通过多层 Transformer block（含 self-attention、cross-attention、FFN）迭代去噪，生成目标视频。与早期基于 U-Net 的视频扩散模型（如 Stable Video Diffusion）相比，DiT 的 Transformer 架构具有更好的可扩展性和生成质量。Sora 的出现证明 DiT 架构可实现高质量视频生成。EasyAnimate 使用 MMDiT 变体（对文本和视频两种模态使用独立的 FFN 和 FC 结构），结合 3D RoPE 位置编码和 rectified flow 采样，48 层 Transformer，支持 text-to-video、image-to-video、inpaint、control 等多种生成模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EasyAnimate DiT 的去噪 pipeline 如下：

```
# 输入: z_T ~ N(0,I) (视频 latent noise), c (文本条件), timesteps T..1
# 输出: z_0 (去噪后的视频 latent)

def dit_denoising_step(z_t, t, c):
    # 1. 位置编码: 对 latent token 施加 3D RoPE
    pos_enc = compute_3d_rope(z_t.shape, h_channels)

    # 2. 文本和视频 token 拼接后进入 MMDiT block
    tokens = concat([c_text, z_t + pos_enc])

    # 3. MMDiT: 两种模态共用 self-attention，但各自独立的 FFN
    for layer in 1..N_layers:
        if layer in window_layers:
            attn_out = multidirectional_swa(tokens)
        else:
            attn_out = full_3d_attention(tokens)
        tokens_video = tokens_video + ffn_video(attn_out.video_part)
        tokens_text  = tokens_text  + ffn_text(attn_out.text_part)

    # 4. 预测速度场 v(t) — rectified flow 的 ODE 向量场
    v_pred = output_proj(tokens_video)
    return v_pred
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DiT 视频模型通常通过以下方式实现：(1) VAE 压缩 —— 使用 3D causal VAE 在空间和时间维度压缩视频（如 8x 空间压缩 + 4x 时间压缩），latent 维度远小于像素空间；(2) 多阶段训练 —— 从低分辨率到高分辨率渐进训练（PixArt 策略），如 256^2 x 49f -> 512^2 x 49f -> 1024^2 x 49f；(3) 文本编码 —— 使用 Qwen2-VL-7B（EasyAnimate）、T5-XXL（CogVideoX）、CLIP+T5（SD3）等提取文本特征；(4) 联合训练 —— 图像和视频数据联合训练（如 34M video + 3M image pairs）。推理时使用 classifier-free guidance 或 rectified flow 快速采样。主要框架包括 EasyAnimate（开源，Apache 2.0）、CogVideoX、HunyuanVideo、OpenSora 等。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
- LongLive__Real-time_Interactive_Long_Video_Generation
