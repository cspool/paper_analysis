## WSD (Warmup-Stable-Decay) Learning Rate Scheduler

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WSD 是由 MiniCPM 论文（Hu et al., 2024）提出的学习率调度器，将训练分为三个阶段：(1) Warmup 阶段（s < W）：线性从 0 增加到峰值 η；(2) Stable 阶段（W ≤ s ≤ T）：保持恒定最大学习率 η；(3) Decay 阶段（T < s < S）：按函数 f(s-T) 衰减。核心优势：(1) 无需预定义总训练步数（与 Cosine 不同）；(2) 可从 Stable 阶段任意 checkpoint 恢复训练——恢复到相同高 LR 继续 Stable 或直接进入 Decay；(3) Decay 阶段仅需 ~10% 总 tokens 即可达到或超越 Cosine 最优性能；(4) Loss 在 Decay 阶段经历快速显著下降。Stuffed Mamba 使用 WSD 配合 10% decay steps、1000 步 linear warmup、50K 步 linear decay。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# WSD Scheduler 定义
def WSD(step, warmup_steps, stable_steps, decay_steps, peak_lr):
    if step < warmup_steps:                    # Warmup
        return (step / warmup_steps) * peak_lr
    elif step < warmup_steps + stable_steps:   # Stable
        return peak_lr
    else:                                      # Decay (linear)
        progress = (step - warmup_steps - stable_steps) / decay_steps
        return peak_lr * (1 - progress)

# Stuffed Mamba 的具体配置
warmup_steps = 1000
decay_steps = 50000
stable_steps ≈ 10 * decay_steps  # 10% decay ratio
peak_lr ∈ {1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3}  # sweep 选择最优
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WSD 特别适合持续预训练（continue pretraining）场景——Stuffed Mamba 从 Mamba-2 8K checkpoint 出发，使用 WSD 方便地在不同训练长度下恢复和继续训练。Decay 阶段损失快速下降，使得用较少的 tokens 即可完成。Stuffed Mamba 使用 linear decay（与 MiniCPM 原论文的 exponential decay 不同），1000 步 warmup + 50K 步 decay。学习率 sweep {1e-5,...,1e-3}，通过 Passkey Retrieval 验证选择最优——注意：不同 LR 的 validation loss 可能相似但 Passkey Retrieval 精度差异巨大。WSD 也被 DeepSeek 等模型采用，体现了从 Cosine 向多阶段调度的行业趋势。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
