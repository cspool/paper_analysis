## Token Merging (ToMe) for Video MLLMs / 面向视频多模态大模型的Token合并

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Merging (ToMe) 是一种视觉 token 压缩技术（Bolya et al., 2023），通过 ViT 内部 token 相似度将冗余 token 合并为更少的 token。在视频 MLLM 中，ToMe 通常应用在 ViT 编码器与 LLM 之间的 Projector 层。每个视频帧首先由 ViT 编码为大量视觉 token（如 SigLIP 输出 768 tokens/frame @ 384×384），ToMe 基于 token 间余弦相似度合并相似 token，将每帧压缩为固定数量（如 16 tokens/frame）。TimeViper 和 VideoChat-Flash 均使用 ToMe 作为视频 MLLM projector 层压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Merging for Video: 768 -> 16 tokens/frame
def ToMe_video_frame(tokens_f):  # [768, D]
    S = cosine_similarity(tokens_f, tokens_f)  # [768, 768]
    merged = tokens_f.clone()
    for _ in range(768 - 16):
        i, j = argmax(S)  # most similar pair
        merged[i] = (merged[i] + merged[j]) / 2
        merged = remove_row(merged, j)
    return merged  # [16, D]
# In TimeViper: for each frame f_t: v_t = ToMe(SigLIP_ViT(f_t))
```
Annotations: 768 tokens = 24×24 patch grid from SigLIP @ 384×384; ToMe = 48× 帧内压缩; 与 TransV 区别：ToMe 在 LLM 外做帧内压缩，TransV 在 LLM 内做帧间+任务感知压缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ToMe 原始实现：https://github.com/facebookresearch/ToMe。在视频 MLLM 中，每帧独立处理（非跨帧 merging）以保持时序信息。TimeViper 实验：vanilla 模型 128 frames OOM；+ToMe 扩展到 ~5K frames。结合 TransV 后总 vision token 压缩比：768→16 (ToMe, 48×)→0.8 (TransV, 20×) ≈ 960×/frame。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
