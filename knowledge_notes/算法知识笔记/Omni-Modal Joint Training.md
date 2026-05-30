## Omni-Modal Joint Training

术语是什么？
Omni-Modal Joint Training 是 OmniVinci 提出的两阶段渐进式训练策略中的第二阶段（第一阶段为 Modality-Specific Training），目标是使 LLM 获得统一的 omni-modal 理解能力。该阶段的核心设计是**多模态数据混合采样**——在每个 batch 中同时包含：(1) modality-specific 数据（纯视觉、纯音频、纯视频），防止单一模态能力退化；(2) omni-modal implicit learning 数据（带音频轨的 video QA），提供隐式跨模态监督；(3) omni-modal explicit learning 数据（Data Engine 合成的 omni-modal QA），提供显式跨模态监督。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
Omni-Modal Joint Training 是连接各模态训练阶段的枢纽。全训练流程共 7 个阶段：
```
Stage 1-5: Vision Training (follow NVILA recipe)
  1. Vision Projector Alignment → 2. Vision Encoder Alignment
  → 3. Vision Pre-Training → 4. Image Instruction Tuning
  → 5. Video Instruction Tuning
  → 产出 "Vision Preliminary Checkpoint"

Stage 6: Audio Training
  1. Audio Projector & Encoder Alignment (50K audio-language pairs)
  2. Audio Instruction Tuning (9.6M audio-SFT samples, full model)
  → 注意：此时 visual understanding 能力会退化

Stage 7: Omni-Modal Joint Training ← 本术语
  数据: 24M samples (image 36%, sound 21%, speech 17%, omni 15%, video 11%)
  配置: cosine LR + linear warmup (3%), base LR=2e-5
  冻结: vision encoder + audio encoder (仅训练 projector + LLM)
  token 总量: ~200B (0.2T)
  → 恢复并提升 visual + audio 联合理解能力
```

训练配置的关键约束：(1) vision/audio encoder 冻结——防止大规模多模态训练破坏预训练好的编码器表示；(2) 低学习率 (2e-5)——在前阶段 checkpoint 附近微调；(3) modality-specific 数据占比 > omni-modal 数据——防止遗忘单模态能力；(4) 200B tokens 总量远小于 Qwen2.5-Omni 的 1.2T，证明良好的架构+数据设计可大幅降低训练成本。

术语一般如何实现？如何使用？
基于 PyTorch 分布式训练（NVIDIA DGX H100 集群），使用标准 LM loss + 可选的 OmniAlignNet contrastive loss 作为联合优化目标。多模态 batch 采样策略：按数据集原始大小加权采样（weighted sampling），确保小数据集不被忽略。Omni-Modal Joint Training 可视为"多模态 SFT + 能力恢复"阶段，后续可选 GRPO post-training（18K omni-modal MCQ, rollout=8, temperature=1.0, top-p=0.99）进一步提升 omni-modal reasoning 能力。

涉及论文标题：
- OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

---
