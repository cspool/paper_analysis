## Implicit and Explicit Omni-Modal Learning

术语是什么？
OmniVinci 提出的两种互补的 omni-modal 学习策略：

**Implicit Omni-Modal Learning（隐式全模态学习）**：利用现有 video QA 数据集中自然存在的同步音频轨进行隐式监督。大多数先前 video LLM 仅使用视频的视觉帧，丢弃了同步音频轨中的信息。OmniVinci 将音频轨作为额外输入，让模型在 video QA 任务中隐式学习视觉-音频的联合理解，无需额外的 omni-modal 标注。关键洞察："Videos are naturally omni-modal when visual and audio streams are present simultaneously but remains under explored."

**Explicit Omni-Modal Learning（显式全模态学习）**：通过 Omni-Modal Data Engine 合成带有显式 omni-modal 标签的对话数据，直接监督模型的视觉-音频联合理解能力。与 Implicit Learning 的"间接"监督不同，Explicit Learning 的 QA 对明确要求模型同时利用视觉和音频信息回答问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Implicit Learning 数据流
Video QA dataset (如 Video-MME) 
  → 使用 video frames + audio track 作为输入（而非仅 frames）
  → 在 video QA 任务上 fine-tune
  → 模型隐式学习利用 audio 辅助视觉理解

# Explicit Learning 数据流
Video corpus with audio
  → Omni-Modal Data Engine 合成 omni-modal QA
  → 明确要求模型结合视觉+音频信息回答
  → 模型显式学习 omni-modal 联合推理

# 训练混合策略（Omni-Modal Joint Training）
每个 batch: 随机采样
  - modality-specific data (image-only, audio-only, video-only)
  - omni-modal implicit data (video QA with audio track)
  - omni-modal explicit data (synthetic omni-modal QA)
```

消融实验：Visual Alone baseline (Video-MME w/o sub. 61.67) → +Implicit Learning (63.76, +2.09) → +Explicit Learning (67.37, +5.70)，证明两种学习策略均有显著增益且互补。Implicit Learning 即使在有 subtitle 的情况下也带来提升 (66.37→66.96)，说明直接从音频学习与从文本 subtitle 学习是不同的信息通道。

术语一般如何实现？如何使用？
Implicit Learning 实现简单——在 video QA 训练时附加音频 encoder 的输出作为额外 token，无需额外数据标注，可以充分利用任何已有 video QA 数据集。Explicit Learning 需要先运行 Data Engine 合成数据，成本较高但效果更好。最佳实践是两者结合使用，如 OmniVinci 的 Omni-Modal Joint Training 阶段。训练细节：200B tokens, cosine LR schedule (warmup 3%), base LR=2e-5, vision/audio encoders frozen。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM
