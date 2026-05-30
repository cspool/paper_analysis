## Video DiT (Video Diffusion Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video DiT（Video Diffusion Transformer）是将Diffusion Transformer架构应用于视频生成的模型范式。核心流程：(1) VAE编码器将原始视频clip压缩为latent表示（如stability-ai VAE做8×8空间下采样），得到shape为[F, H, W]的latent token grid，F/H/W分别为frames、高度、宽度方向的token数；(2) 对latent表示加入扩散噪声（DDPM或flow matching范式），带噪latent与conditioning（timestep、text prompt等）一起输入DiT模型；(3) DiT模型由多个DiT block堆叠而成，每个block包含self-attention模块（捕获视频token间的时空依赖）和cross-attention模块（对齐text prompt）。Self-attention有两种范式：interleaved spatial-temporal attention（交替在spatial和temporal维度做attention，计算高效但信息捕获不足）和3D full attention（所有token间全对全attention，质量最佳但O(S²)复杂度）。跨模态对齐通过cross-attention实现，其中Q来自视频token（S个），K/V来自text prompt（<120 tokens，复杂度远低于self-attention）。

从算法pipeline角度拆解，Video DiT训练的pipeline：
```
# Video DiT training pipeline (flow matching variant)
for each training step:
    # 1. VAE encoding
    latent = VAE.encode(video_clip)  # [F, H, W] latent grid
    
    # 2. Noise injection (flow matching)
    t ~ Uniform(0, 1)
    noise ~ N(0, I)
    z_t = t * latent + (1-t) * noise  # 线性插值路径
    
    # 3. Text encoding
    text_emb = TextEncoder(text_prompt)  # ~120 tokens
    
    # 4. DiT forward
    h = z_t + timestep_embedding(t)
    for block in DiT_blocks:
        # Self-attention (3D full attention)
        h = SelfAttn(LN(h)) + h           # O(S²d), S = F*H*W
        # Cross-attention (text conditioning)
        h = CrossAttn(LN(h), text_emb) + h  # O(S·T·d), T < 120
        # FFN
        h = FFN(LN(h)) + h
    velocity = output_proj(h)
    
    # 5. Flow matching loss
    loss = MSE(velocity, latent - noise)  # predict velocity field
```

典型配置：0.8B (28层, 12头, head size 96)、2.7B (32层, 16头, head size 128)、30B (42层, 24头, head size 256)。SOTA模型如Meta MovieGen、HunyuanVideo、CogVideoX均采用类似架构。self-attention在长序列下占>90%训练时间（200K tokens时forward 92%、backward 93%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现：基于LLM的transformer架构改造，主要在attention范式（spatial-temporal vs full 3D）和conditioning方式（timestep embedding + text cross-attention）上与LLM不同。主要框架包括HunyuanVideo（Tencent, GitHub开源）、OpenSora（潽方AI, GitHub开源）、CogVideoX（智谱, GitHub开源）、MovieGen（Meta, 闭源）。训练使用Adam optimizer, lr=1e-4, gradient checkpointing, flow matching或DDPM范式。开源实现通常基于PyTorch + FSDP/DeepSpeed分布式训练。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
