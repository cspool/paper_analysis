## Reinforcement Learning from Human Feedback (RLHF) / 基于人类反馈的强化学习

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reinforcement Learning from Human Feedback (RLHF) 是一套将 LLM 输出与人类价值观和偏好对齐的训练范式，由 Ouyang et al. (2022, InstructGPT) 和 Ziegler et al. (2019) 确立。标准 RLHF 分三阶段：(1) Supervised Fine-Tuning (SFT) — 在高质量人工标注的 (prompt, response) 对上微调 base model；(2) Reward Model (RM) Training — 收集人类对同一 prompt 的多个模型输出的排序数据，基于 Bradley-Terry 模型训练 reward model r_φ，使得 P(y_w ≻ y_l) = σ(r_φ(x,y_w) - r_φ(x,y_l)) 最大化；(3) Reinforcement Learning — 使用 PPO 优化 policy π_θ，目标为 max E[r_φ(x,y)] - β·KL(π_θ || π_SFT)，KL 散度约束防止 policy 偏离 SFT 模型太远（避免 reward hacking）。RLHF 使 InstructGPT (1.3B) 在人类评估中优于 175B GPT-3，成为 ChatGPT 的核心对齐技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准 RLHF pipeline 伪代码：
```
# === Stage 1: SFT ===
For each (x, y_demo) in demonstration_data:
    L = -log π_θ(y_demo | x)
    θ ← θ - η * ∇L
π_SFT ← π_θ  # 保存 SFT 模型

# === Stage 2: Reward Model Training ===
For each (x, y_w, y_l) in comparison_data:  # y_w ≻ y_l
    r_w = r_φ(x, y_w)  # reward model 对 preferred 的评分
    r_l = r_φ(x, y_l)  # reward model 对 dis-preferred 的评分
    L = -log σ(r_w - r_l)  # Bradley-Terry 损失
    φ ← φ - η_r * ∇L

# === Stage 3: PPO Fine-tuning ===
π_θ ← π_SFT  # 初始化
For each x in prompt_data:
    y = sample(π_θ(x))      # 当前 policy 生成
    r = r_φ(x, y)           # reward model 评分
    KL_penalty = log π_θ(y|x) - log π_SFT(y|x)
    R = r - β * KL_penalty  # KL-正则化奖励
    θ ← PPO_update(θ, R)    # PPO policy gradient
```
DPO 简化了 RLHF：直接将 Stage 2+3 合并为一步优化，利用 reward-policy 双射关系 r = β log(π/π_ref) 隐式表示 reward，在偏好对数据上直接优化 policy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：TRL (https://github.com/huggingface/trl) 提供 PPOTrainer、RewardTrainer、DPOTrainer 完整 RLHF pipeline。主要变体：(1) RLAIF (RL from AI Feedback) — 用 AI（如 GPT-4）替代人类进行偏好标注，降低成本；(2) Online RLHF — 在训练过程中迭代收集新偏好数据，而非仅用固定 offline 数据集；(3) DPO/SimPO/KTO — 各种无需显式 reward model 的简化方案。TPO 论文定位在 RLHF 的 DPO 分支上，将偏好学习应用到视频时序理解领域。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models
