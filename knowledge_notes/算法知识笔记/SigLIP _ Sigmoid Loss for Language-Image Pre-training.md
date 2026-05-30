## SigLIP / Sigmoid Loss for Language-Image Pre-training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP (Sigmoid Loss for Language-Image Pre-training) 是 Zhai et al. (ICCV 2023) 提出的 CLIP 训练变体，将 CLIP 原生的 softmax 对比损失替换为 sigmoid loss。核心差异：CLIP 使用 softmax + cross-entropy 对 batch 内所有 image-text pairs 进行全局归一化对比（需要大 batch size 提供足够多的负样本）；SigLIP 将每个 image-text pair 独立处理，用 sigmoid 二元分类器判断 pair 是否匹配，负样本从 batch 内其他样本中获取，免除全局 softmax 归一化。优势：(1) batch size 不再受限于 softmax 的分母精度要求，可支持更大 batch；(2) 训练更稳定；(3) 在处理大规模数据时性能更优。后续版本 SigLIP-2 (Tschannen et al., 2025) 进一步扩展至多语言支持（109 语言）、改进语义理解、定位和 dense features，使用 12B alt-text pairs 训练。LLM2CLIP 使用 SigLIP2-SO/14 (428M) 作为 SOTA baseline，在 40B image-text pairs 上预训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLIP Loss (softmax-based, 用于对比)
def clip_loss(I_e, T_e, t):
    # I_e, T_e: [B, d] L2-normalized embeddings
    logits = I_e @ T_e.T * exp(t)  # [B, B]
    labels = arange(B)              # diagonal = positive pairs
    loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
    # 分母对所有 B 个 texts/images 做全局 softmax
    return loss

# SigLIP Loss (sigmoid-based, 用于对比)
def siglip_loss(I_e, T_e, t, b):
    # I_e, T_e: [B, d] L2-normalized embeddings
    logits = I_e @ T_e.T * exp(t) + b  # [B, B], b = learnable bias

    # 对角线 = positive (label=1), 非对角线 = negative (label=-1)
    labels = 2 * eye(B) - 1  # [B, B]: 对角=1, 其他=-1

    # Sigmoid binary cross-entropy, 每个 pair 独立计算
    loss = -log(sigmoid(labels * logits)).sum() / B
    # 无需全局 softmax 归一化
    return loss
```

Annotations: `t` = log-temperature (可学习)；`b` = learnable bias for sigmoid；softmax 的全局归一化使负样本数量必须充足（batch size → 大），sigmoid 的独立处理解耦了 batch size 与负样本质量的关系；SigLIP 实验中使用 batch size 32K~64K 且性能稳定，而 softmax CLIP 在小 batch 下性能退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SigLIP 的开源实现：(1) Google 官方: https://github.com/google-research/big_vision；(2) HuggingFace: google/siglip-so400m-patch14-224, google/siglip2-so400m-patch14-224；(3) OpenCLIP 复现。LLM2CLIP 使用 SigLIP2-SO/14 作为最强 baseline —— 在 40B data 预训练的基础上，LLM2CLIP 仅用 60M fine-tuning data 即实现 Flick30K +1.0/+1.9 (I2T/T2I)、long-caption +14.8/+15.8、multilingual +11.9/+15.2 的提升。SigLIP 的适用场景：大规模 CLIP 式预训练，特别是训练数据量极大（10B+ pairs）且需要稳定训练时。Sigmoid loss 在 batch size ≥ 16K 时表现出最优性能。

涉及论文标题：
- LLM2CLIP__Powerful_Language_Model_Unlock_Richer_Visual_Representation
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
