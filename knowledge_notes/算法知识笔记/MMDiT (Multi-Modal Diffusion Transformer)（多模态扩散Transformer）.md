## MMDiT (Multi-Modal Diffusion Transformer)（多模态扩散Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MMDiT 是 SD3 (Stable Diffusion 3) 提出的多模态 DiT 架构变体。核心设计：文本和视频两种模态共享 self-attention 层进行跨模态交互，但使用各自独立的全连接层和 FFN，以处理两种模态特征在数值尺度和语义空间上的差异。这种"共享注意力 + 独立 FFN/FC"的设计使模型既能实现文本-视频对齐，又能保持各自模态的特征表达能力。EasyAnimate 中文本特征经 RMSNorm + FC 变换后与视频 latent token 拼接进入 MMDiT。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MMDiTBlock(nn.Module):
    def forward(self, h_text, h_video):
        # 共享 Self-Attention
        h_combined = torch.cat([norm_text(h_text), norm_video(h_video)], dim=1)
        attn_out = self.self_attn(h_combined)
        attn_text, attn_video = split(attn_out)

        # 独立 FFN (残差连接)
        h_text  = h_text  + attn_text
        h_text  = h_text  + self.ffn_text(norm(h_text))
        h_video = h_video + attn_video
        h_video = h_video + self.ffn_video(norm(h_video))
        return h_text, h_video
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MMDiT 最初由 SD3 (Sauer et al., 2024) 提出用于文生图，EasyAnimate 扩展到视频。与标准 DiT（所有 token 共享同一 FFN）或 Cross-Attention DiT 相比，MMDiT 在模态对齐和特征表达方面取得更好平衡。EasyAnimate 的 7B 和 12B 版本均基于 MMDiT。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
