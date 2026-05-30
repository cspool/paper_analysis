## Cross-Modality Alignment (跨模态对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Modality Alignment（跨模态对齐）是多模态大模型训练的第一阶段（Alignment Stage），目标是将不同模态（图像、视频、音频、语音等）的编码器输出映射到统一的 LLM 语言表示空间，使 LLM 能"理解"非文本模态的输入。该阶段仅训练模态连接器（Connector），冻结编码器和 LLM，通过最小化生成文本的交叉熵损失来学习模态-语言映射。

Uni-MoE 的跨模态对齐阶段针对每种模态独立执行：(1) 视觉对齐——使用 CLIP-V + MLP 线性投影（沿用 LLaVA 预训练权重）；(2) 语音对齐——使用 Whisper-small + Speech-QFormer + 线性投影，训练数据为 Common Voice (1.7M 短语音)；(3) 音频对齐——使用 BEATs + Audio-QFormer + 线性投影，训练数据为 WavCaps/AudioCaps/MELD/Clotho (194K)。

从算法pipeline角度拆解术语：

跨模态对齐训练（以语音对齐为例，对应 Algorithm 1 Stage 1）：

```
for each step:
    (x, y) = sample(PD_speech)          # 采样语音-文本对
    x_speech = Whisper(x)               # 冻结语音编码
    x_q = Speech-QFormer(x_speech)      # Q-Former 蒸馏
    x_tokens = Linear(x_q)              # 线性投影到 LLM 空间
    prediction = LLM(x_tokens)          # 冻结 LLM 前向
    loss = CE(prediction, tokenize(y))  # 交叉熵生成损失
    # 仅更新: Q-Former + Linear projection 参数
    θ = θ - α ∇_θ loss
```

此阶段的关键性质：
- 仅训练 Connector（Q-Former + 投影层），编码器和 LLM 冻结
- 每种模态独立训练，互不干扰
- 使用模态-文本配对数据（如 speech-transcription pairs）
- Loss 为标准语言建模交叉熵
- 学习率 2e-5，global batch size=32，AdamW

术语一般如何实现？如何使用？

在 Uni-MoE 中，跨模态对齐在 2 块 A100 GPU 上进行，分别处理 1.7M 短语音数据和 194K 音频字幕数据。视觉对齐部分复用 LLaVA 已有的 CLIP+MLP 视觉连接器（预训练完成）。跨模态对齐是多模态 LLM 训练的必要第一步——没有此阶段，LLM 无法将非文本模态的 continuous embeddings 解释为有意义的语义信息。与 Meta-Transformer 的统一 tokenizer 思路不同，Q-Former 方法通过 cross-attention 机制实现了更具表达力的模态特征蒸馏。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
