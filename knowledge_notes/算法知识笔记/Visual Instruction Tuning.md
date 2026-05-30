## Visual Instruction Tuning

术语解释
Visual Instruction Tuning 是将预训练 LLM 转换为多模态 LLM 的核心训练范式，由 LLaVA (Liu et al., 2023) 提出。核心思想是将图像-文本对数据转换为 instruction-following 格式（类似 NLP 的 instruction tuning），让 LLM 学会遵循包含图像的指令并生成文本回复。

术语是什么？
流程：(1) 图像通过冻结的视觉编码器提取 visual tokens；(2) 可训练的 MLP 连接器将 visual tokens 投影到 word embedding 空间；(3) visual tokens + text tokens 拼接后输入 LLM；(4) LLM 仅对 text tokens 计算自回归交叉熵损失（visual tokens 不计算损失）。

```
# Visual Instruction Tuning 的训练样本格式
# 单轮对话:
# <image> USER: <question> ASSISTANT: <answer>
# 多轮对话:
# <image> USER: <q1> ASSISTANT: <a1> USER: <q2> ASSISTANT: <a2>

def visual_instruction_tuning_step(image, conversation):
    visual_tokens = vision_encoder(image)          # CLIP ViT
    word_embed_tokens = mlp_connector(visual_tokens)  # 投影到 LLM 空间
    
    # 拼接 visual tokens + text tokens
    text_tokens = tokenize(conversation)             # "USER: ... ASSISTANT: ..."
    input_ids = concat([visual_tokens, text_tokens])
    
    # 仅对 ASSISTANT 回复部分计算 loss
    logits = LLM(input_ids)
    loss = CrossEntropy(logits[assistant_mask], labels[assistant_mask])
    loss.backward()
```

从算法pipeline角度拆解术语：
Visual instruction tuning 位于多模态 LLM 训练 pipeline 的核心阶段。数据来源于各种 VQA 和看图理解数据集，统一转换为 "USER: <question about image> ASSISTANT: <answer>" 格式。CuMo 在此阶段加入 Co-Upcycled MoE blocks，训练数据混合包括 LLaVA-665K、ShareGPT4V、DocVQA、ChartQA 等约 1.65M 样本。

术语一般如何实现？如何使用？
- 基础：LLaVA 系列（v1, v1.5, NeXT）
- 数据来源：开源 VQA 数据集（VQAv2, GQA, TextVQA 等）+ GPT-4V 生成的高质量指令数据（ShareGPT4V, ALLaVA）
- 典型超参数：学习率 2e-5 ~ 4e-6，batch size 128-256，使用 DeepSpeed ZeRO-3
- 评估：贪心解码，multiple choice / GPT-API 评分（LLaVA-Wild 用 gpt-4-0613, MathVista 用 gpt-3.5-turbo）
- CuMo 的扩展：在 visual instruction tuning 阶段引入 Co-Upcycled MoE blocks + bzloss

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts
