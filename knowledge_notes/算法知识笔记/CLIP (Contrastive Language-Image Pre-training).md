## CLIP (Contrastive Language-Image Pre-training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CLIP (Contrastive Language-Image Pre-training) 是由 OpenAI 提出的跨模态基础模型，通过在数亿至数百亿 image-text pairs 上进行对比预训练，将图像和文本映射到共享表示空间。核心架构为双塔结构：Vision Encoder（通常为 ViT，如 ViT-B/16 86M、ViT-L/14 307M）和 Text Encoder（轻量自回归模型，约 1/3 ViT 参数量，上下文窗口限制为 77 tokens）。训练目标为对比损失：最大化匹配 image-text pair 的 cosine similarity，最小化非匹配 pair 的 similarity。CLIP 支持 zero-shot 分类（通过文本模板如 "a photo of the {classname}"）、图像-文本检索、跨模态特征提取。CLIP 的视觉特征被广泛用于 Multimodal LLMs (如 LLaVA、Qwen-VL) 和图像/视频生成模型 (如 Stable Diffusion 3、Wan) 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIP 训练流程
# Vision Encoder: ViT (Vision Transformer)
# Text Encoder: 轻量自回归 Transformer (~1/3 ViT params)
# 输入: N 个 (image, text) 对

def clip_training(images, texts):
    # 视觉编码
    I_f = ViT(images)          # [N, d], d=embedding_dim (e.g., 512/768/1280)
    I_e = L2_normalize(I_f)    # [N, d]

    # 文本编码 (causal attention, max 77 tokens)
    T_f = TextEncoder(texts)   # [N, d]
    T_e = L2_normalize(T_f)    # [N, d]

    # 对比损失 (双向)
    logits = I_e @ T_e.T * exp(t)  # [N, N], t 为可学习 temperature
    labels = arange(N)             # 对角线为正样本
    loss_i2t = CrossEntropy(logits, labels)
    loss_t2i = CrossEntropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss

# Zero-shot 分类推理
def clip_zero_shot_classify(image, class_names):
    I_e = L2_normalize(ViT(image))
    texts = [f"a photo of the {c}" for c in class_names]
    T_e = L2_normalize(TextEncoder(texts))
    scores = I_e @ T_e.T
    return argmax(scores)
```

Annotaions: `I_f`/`T_f` 为 raw features；`I_e`/`T_e` 为 L2 归一化后 embedding；`t` 为 temperature 参数（CLIP 原生使用可学习 logit scale，类似 temperature 的倒数）；对比损失同时计算 image→text 和 text→image 两个方向。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CLIP 的开源实现：OpenAI 官方 CLIP 在 https://github.com/openai/CLIP，预训练权重通过 HuggingFace (openai/clip-vit-base-patch16 等) 发布。后续改进包括：(1) SigLIP — 使用 sigmoid loss 替代 softmax 对比损失，支持更大 batch size；(2) EVA-CLIP/EVA02 — 改进训练技巧；(3) MetaCLIP — 数据筛选优化；(4) Long-CLIP — 扩展文本上下文长度。CLIP 的典型使用场景：(a) 作为多模态检索器的 backbone；(b) 作为 Multimodal LLM 的视觉编码器（LLaVA 系列用 CLIP-ViT-L/14-336 + MLP projector 连接 Vicuna）；(c) 作为扩散模型的文本编码器（SD3 用 CLIP 文本分支）。LLM2CLIP 论文在预训练 CLIP 基础上通过两阶段微调注入 LLM 能力，将 Text Encoder 替换为 CC fine-tuned LLM + Adaptor。

ReVisionLLM 使用 Frozen CLIP ViT-L/14 作为视频编码器，仅提取每帧 CLS token (768维) 而非全部 spatial tokens，显著降低视觉特征维度（每帧 1 token vs 257 tokens）。CLIP text encoder (12-layer) 用于提取 query 文本特征以参与 Hierarchical Adapter 的 Cross-Attention 对齐。CLIP similarity 被用作 baseline 排序方法（CONE ranking），但被 ReVisionLLM 的 LLM entropy-based 置信度取代。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
