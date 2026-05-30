## Hierarchical Adapter for Vision-Language Models / 层次化适配器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Adapter 是 ReVisionLLM 中连接冻结 CLIP ViT-L/14 和 Vicuna-7B 的适配器模块，负责将视频帧特征转化为稀疏和密集两种时间表示。由三个轻量子模块组成：(1) Cross-Attention (2 layers, 8 heads) — 以视频段特征为 query、文本特征为 key/value 实现跨模态语义对齐；(2) Self-Attention (2 layers, 8 heads) — 将可学习 sparse token 与文本对齐段特征 concatenate 后压缩为单个 768 维向量（段级压缩比 250:1）；(3) Linear Projection — 将 CLIP CLS token (768维) 投影到 LLM embedding space (4096维) 生成密集特征。整体仅 2+2 attention layers vs CLIP 24 layers，几乎无额外计算开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class HierarchicalAdapter:
    cross_attn = MultiheadAttention(d=768, heads=8, layers=2)  # 跨模态对齐
    self_attn = MultiheadAttention(d=768, heads=8, layers=2)   # 稀疏压缩
    ffn = Sequential(Linear(768,3072), GELU, Linear(3072,768))
    linear_proj = Linear(768, 4096)  # CLIP → LLM embedding

    def forward(C_i, Q, S_learnable):
        # C_i: (250, 768), Q: (N_s, 768), S_learnable: (1, 768)
        C_tilde = cross_attn(query=C_i, key=Q, value=Q)  # 文本对齐
        attn_out = self_attn(concat([S_learnable, C_tilde]))  # 压缩
        S_i = ffn(attn_out[0])  # (768,) sparse feature
        D_i = linear_proj(C_i)  # (250, 4096) dense feature
        return S_i, D_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时，Stage 1 先冻结 Linear Projection 微调 LLM LoRA，再冻结 LoRA 微调 Cross-/Self-Attention + FFN (1 epoch, batch=32, LR=1e-3) 学习 sparse 生成。Stage 2 冻结全适配器仅微调新 LoRA。推理时全冻结。预训练 Linear Projection 使用 LCS-558K (LLaVA) 1 epoch 对齐 CLIP 和 LLM 的 embedding space。

涉及论文标题：
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
