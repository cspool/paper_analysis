## Interleaved Multimodal Reasoning（交织多模态推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Multimodal Reasoning 指推理轨迹中文本 token 和视觉 token（latent 或 explicit）交替排列的推理模式，而非将所有视觉信息放在序列开头或在固定位置插入。Mirage 的推理链格式：text_pre → [latent_1, latent_2, latent_3, latent_4] → text_post。与 vision-first (图像在 prompt 开头一次性输入) 和 text-only CoT 的关键区别：模型在推理过程中可以动态决定 "何时需要视觉信息"，形成 text-vision 混合的 reasoning trajectory。数据合成方式：helper image I 嵌入到 textual reasoning chain 中间 (o = o_pre ⊕ I ⊕ o_post)，训练模型学习这种交织模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Interleaved Reasoning 轨迹示例 (VSP Spatial Reasoning)
# Query: "Will the agent safely reach the goal after: Go Left, Go Down?"

# 自回归生成如下交织序列:
# text_pre (模型直接生成):
#   "Moving to [(1,3),Hole] ends the game instantly..."
#   触发 <image_placeholder> → latent generation

# latent_tokens (k=4, bypass LM head, 连续 embedding):
#   [e_1, e_2, e_3, e_4]  ← encode 路径空间信息

# text_post (attend 到 latent tokens):
#   "...making failure certain. The answer is \boxed{B}."

# Attention pattern for text_post token at position t:
# Q_t · [K_{text_pre}, K_{e1}, K_{e2}, K_{e3}, K_{e4}, K_{post_<t}]^T
# latent K/V entries 为 text_post 提供 task-specific visual cues

# GRPO RL 阶段: 模型可自由探索不同的 interleaved 模式
# latent tokens 排除于 KL penalty (λ_kl 仅应用于 text tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 数据合成 pipeline 生成 interleaved 训练样本（helper image 嵌入推理链中间，o = o_pre ⊕ I ⊕ o_post）；(2) 特殊 `<image_placeholder>` token 触发 latent generation 模式切换（bypass LM head → direct hidden state as embedding）；(3) GRPO RL 阶段 latent tokens 不受 KL constraint，允许模型探索不同交织模式。对比 unified model (Anole/MVoT) 的 interleaved generation：Mirage 的 latent visual tokens 不需要 external image decoder，避免了 pixel generation 开销和推理质量退化。适用场景：需要推理过程中多次参考视觉信息的多模态任务（spatial reasoning, jigsaw puzzle, navigation）。

涉及论文标题：
- Machine_Mental_Imagery__Empower_Multimodal_Reasoning_with_Latent_Visual_Tokens
