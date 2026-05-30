## 3D Causal VAE（三维因果变分自编码器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
3D Causal VAE 是一种对视频同时在空间和时间维度进行压缩的变分自编码器，其"因果性"（causal property）体现在：编码/解码当前帧时只能依赖当前及之前的帧信息，不能看到未来帧。这与 image-based VAE（仅空间压缩逐帧处理）和 non-causal 3D VAE（可访问所有帧）形成对比。压缩率通常为 8x 空间 + 4x 或 8x 时间，将原始视频从 (T, H, W, 3) 压缩到 (T/k_t, H/k_s, W/k_s, latent_dim)。因果性的关键优势：在解码长视频时，可以缓存前帧的 latent state，连接当前帧进行增量解码，极大降低显存使用，支持生成长视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 编码器
def encode_3d_causal_vae(video_frames):
    """video_frames: (T, H, W, 3) -> z: (T//4, H//8, W//8, latent_dim)"""
    for layer in encoder_layers:
        x = causal_3d_conv(x)  # temporal padding only on past side
        x = group_norm(x)
        x = silu(x)
    mu, logvar = head_mu(x), head_logvar(x)
    z = mu + exp(0.5 * logvar) * eps  # reparameterization
    return z

# 增量解码 (利用因果性)
def decode_incremental(z_prev_cache, z_current_frame):
    z_combined = concat_causal([z_prev_cache, z_current_frame], dim='t')
    for layer in decoder_layers:
        x, cache = causal_3d_deconv(x, cache=layer.cache)
    return x, cache  # 返回当前帧像素 + 更新缓存
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
3D Causal VAE 在 EasyAnimate 中：(1) 训练阶段使用变帧间隔采样提升跨帧编解码鲁棒性；(2) 遵循 MovieGen 添加 latent encoding penalty loss 减少 speckle artifacts；(3) 使用 spatial/temporal slicing 降低长视频高分辨率解码时的 GPU 显存；(4) 在 Reward BP 中的关键作用 —— causal 属性意味着只需解码第一帧（F=1）即可通过因果关系推断后续帧质量，避免多帧 reward 导致的 dynamics 损失和 reward hacking。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
