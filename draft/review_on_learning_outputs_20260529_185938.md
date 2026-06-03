
1.1-1.2是方法概述.

1. patchfy和sampler能否融合?
    <!-- ? 应该能融合,本质就是地址映射+逐点步进,每一步都作patchify和unpatchify是通用的写法.patchify是为了Transformer的token处理语义而设计,UNet中使用Conv就没有patchify模块. -->
    <!-- ! DiT -->
    # ═══ DiT Forward (单步去噪预测) ═══
    <!-- ! 每个采样步骤中,进入Backbone前都需要作patchify,Backbone结束后unpatchify??? -->
    <!-- ! 步进公式是针对latent z分布,但patchfy只改变数据排布,而不改变分布,为何不改写步进公式到patch版本? -->
    # 将 latent 展平为 patch tokens
    z_patches = **Patchify**(z)                         # [N_patches, d_model]
                                                    # N_patches = (H/p) × (W/p)
                                                    # 例: 32×32 / 2 = 256 patches

    # === DiT Block × L (所有 tokens 通过相同 Transformer) ===
    h = z_patches  # [N_patches, d_model]
    for l = 1..L:                                    # L=28 for DiT-XL/2
        ...
    
    
    <!-- ! Video -->
    z = z_T  # [F, H, W, C]

    for step s = T down to 1:
        # === Step 1: 3D **Patchify** ===
        <!-- ! [F,H,W,C]划分为[F/2*H/2*W/2, C*2*2*2] -->
        z_patches = Patchify3D(z, patch_size=(2,2,2))
        # z_patches: [F/2 × H/2 × W/2, C·8]
        # → [7 × 20 × 32, 128] = [4480, 128]

        # === Step 2: Text-Video Conditioning ===
        <!-- !image的DiT中text和时间作为condition,video的DiT需要text和video的跨模态计算 -->
        <!-- !时间t作为条件计算gate,scale,shift -->
        t_emb = TimestepEmbedding(s)

        <!-- !text token投影到video的z空间,需要参与video的跨模态Attn -->
        c_text = TextEncoder(c)                           # CLIP/T5 text features
        # Text features 经 RMSNorm+FC 变换后与 video tokens 拼接

        # === Step 3: MMDiT Blocks (共享 Self-Attn + 独立 FFN) ===
        h_text, h_video = c_text, z_patches
        for l = 1..L:                                     # L=28~48 for video DiT
    
    **取决于每步的patchfy是否包含额外的全局embedding投影计算?**
    patchify和unpatchify对应,可能包含额外的投影计算(patch-dim到model-dim).

