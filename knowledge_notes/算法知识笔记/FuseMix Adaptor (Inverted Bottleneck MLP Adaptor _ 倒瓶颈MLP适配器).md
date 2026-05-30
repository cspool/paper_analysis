## FuseMix Adaptor (Inverted Bottleneck MLP Adaptor / 倒瓶颈MLP适配器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FuseMix Adaptor 是 LLM2CLIP Stage 2 中连接 CC fine-tuned LLM 与 CLIP Vision Encoder 的关键组件，采用 FuseMix (Vouitsis et al., CVPR 2024) 提出的 inverted bottleneck MLP 架构。由 4 层倒瓶颈线性块组成，每层结构为 Linear(d_in → d_hidden) → GeLU → Linear(d_hidden → d_in) + residual connection。最后通过一个最终投影层将 LLM 的 hidden dimension (4096 for Llama 3.1 8B) 映射到 CLIP 的 embedding dimension (1280)。总参数量约 67.1M。Adaptor 放置于 LLM 输出之后、CLIP contrastive loss 之前，作为"可学习的桥梁"，将 LLM 的文本嵌入空间对齐到 CLIP Vision Encoder 的表示空间。由于 Adaptor 完全独立于 LLM，配合 Offline-loading 策略可在训练时不加载 LLM 到 GPU 显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FuseMix Adaptor 结构 (4-layer inverted bottleneck MLP)
# 输入: LLM avg-pooled sentence embedding ∈ R^4096
# 输出: embedding ∈ R^1280 (与 CLIP visual embedding 对齐)

class FuseMixAdaptor(nn.Module):
    def __init__(self, d_in=4096, d_hidden=8192, d_out=1280, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_in, d_hidden),   # 4096 → 8192 (expand)
                nn.GELU(),
                nn.Linear(d_hidden, d_in),   # 8192 → 4096 (project back)
            ))
        self.final_proj = nn.Linear(d_in, d_out)  # 4096 → 1280

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)           # expand → activate → project
            x = x + residual        # residual connection
        x = self.final_proj(x)     # project to CLIP embedding space
        return x  # [B, 1280]

# Stage 2 训练中 Adaptor 的使用:
# LLM 冻结 → precomputed sentence embeddings[4096] → Adaptor → [1280]
#                                                                  ↓
#                                                        CLIP contrastive loss
#                                                                  ↑
#                           ViT(image) → visual embedding [1280] ─┘
```

Annotations: `d_hidden=8192` 为倒瓶颈的扩展维度（2× d_in）；residual connection 保证梯度流动和训练稳定性；Adaptor 总参数量 67.1M = 4 × (4096×8192 + 8192×4096) + 4096×1280 ≈ 4 × 67.1M + 5.2M。LLM2CLIP Table 8/Table A7 消融显示：4-layer Linear Adaptor 性能 (80.4/77.9) 与 1-layer Transformer Adaptor (80.2/77.3) 相当，选择更简单的 Linear 结构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FuseMix Adaptor 的消融发现（LLM2CLIP Table A7）：(1) 无 Adaptor → 1-layer → 2-layer → 4-layer 性能递增 (78.3→79.2→80.1→80.4 Avg I2T)；(2) 4-layer Linear Adaptor (80.4) 与 1-layer Transformer Adaptor (80.2) 性能相当，Linear 结构更简单；(3) Stage 1 中是否使用 Adaptor 对最终结果影响微小（80.4 vs 80.5）。LLM2CLIP 默认配置：Stage 1 不使用 Adaptor (直接对 LLM 输出做 avg pooling)，Stage 2 使用 4-layer Linear Adaptor。一般使用 FuseMix Adaptor 的场景：需要将一个预训练模型的 frozen embedding 投影到另一模型的 embedding space 时，通过 MLP 层级联提供足够的非线性映射能力。与 LoRA 的区别：Adaptor 是独立模块位于模型输出之后，LoRA 是注入到模型内部权重的低秩分解。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
