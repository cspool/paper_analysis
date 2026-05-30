## Sandwich Prompt

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sandwich Prompt 是 VisualRWKV 针对 RNN-based VLM 设计的一种多模态提示策略。传统 Transformer VLM（如 LLaVA）由于 self-attention 机制可以随时访问任意历史 token，对 prompt 的 image token 位置不敏感。但 RNN 模型（如 RWKV）因其序列特性无法"回溯"已处理的信息——模型看到 token 后立即决定是否存入固定大小的 hidden state，无法直接访问原始输入。Sandwich Prompt 将 image token 插入 instruction token 中间，形成"指令前缀 → 图像 → 指令后缀"的三明治结构。前半段指令帮助模型确定从图像中提取什么信息（激活正确的检索意图），后半段指令确保问题在图像处理完成后仍被牢记。实验证明 Sandwich Prompt 显著优于 Image First（图像在前，模型处理图像时不考虑问题）和 Image Last（图像在后，模型先读问题但被图像 token 覆盖后遗忘问题）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种 prompt 方法的对比伪代码：
```
# Image First Prompt：
Input = [<image_tokens> | System | Question | "### Assistant:"]
# 问题: 模型处理图像时尚未读到 question，缺少上下文引导

# Image Last Prompt：
Input = [System | Question | <image_tokens> | "### Assistant:"]
# 问题: 模型读到问题后在处理 576 个 image tokens 期间，RNN state 逐渐遗忘问题内容

# Sandwich Prompt (最优)：
Input = [System | "### Human:" | <image_tokens> | "\nQuestion: ...\n### Assistant:"]
# "### Human:" 激活回答意图 → 读图时带着意图提取相关特征
# → "\nQuestion:" 再次提醒问题内容 → 生成答案
```
Sandwich Prompt 在减少 image tokens 时表现出更强的鲁棒性（Table 9），这是因为它建立了"两端指令夹图像"的信息冗余——即使中间的图像信息被压缩，两端的文本指令仍能保持语义锚定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 tokenized 序列构建阶段，将 vision encoder 输出的 visual tokens（576 个 for CLIP-L/14@336×336）插入到 tokenized text instruction 的指定位置。训练和推理时 Sandwich Prompt 保持一致格式。VisualRWKV 7B 上 Sandwich Prompt 比 Image First 在 ScienceQA 上提升 +5.49 点（69.71 vs 65.59？实际上 Table 3 显示 Image First 67.93 vs Sandwich 69.71）。特别适用于 RNN/SSM 架构的 VLM，但设计理念也可推广到 Transformer VLM 中优化长距离信息保留。

涉及论文标题：
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
