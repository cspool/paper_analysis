## Warmup-Stable-Decay (WSD) Learning Rate Schedule

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warmup-Stable-Decay（WSD）学习率调度是 LLM 预训练中一种三阶段学习率调度策略。其将训练分为：(1) **Warmup 阶段**：学习率从 0（或极小值）线性增加到峰值 η_max（通常占训练步数的 1-2%）；(2) **Stable/Constant 阶段**：学习率保持恒定在 η_max，持续大部分训练步数（如 89% 步数）；(3) **Decay/Cooldown 阶段**：学习率从 η_max 线性衰减到 0（通常占步数的 10%）。与 Cosine Decay（学习率从峰值连续余弦衰减到底）不同，WSD 的核心特点是存在漫长的恒学习率阶段，这使得：(a) 可以在不同 token 预算处插入 cooldown 来灵活复检模型状态；(b) 恒学习率阶段模型被限制在同一个 loss basin 中迭代，促进探索而非收敛；(c) cooldown 衰减到零时模型才进入更尖锐的 loss 区域。WSD 由 Zhai et al. (2022) 提出用于 vision transformer，后由 Hu et al. (2024) (MiniCPM)、Bakouch et al. (2025) (SmolLM3) 等大规模 LLM 训练广泛采用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WSD 调度在 LLM 预训练中的典型实现伪代码：
```
def wsd_schedule(step, total_steps, peak_lr=3e-3, warmup_pct=0.01, cooldown_pct=0.1):
    warmup_steps = int(total_steps * warmup_pct)
    cooldown_steps = int(total_steps * cooldown_pct)
    stable_steps = total_steps - warmup_steps - cooldown_steps
    if step < warmup_steps:
        lr = peak_lr * (step / warmup_steps)       # 线性增长
    elif step < warmup_steps + stable_steps:
        lr = peak_lr                                # 恒定
    else:
        decay_progress = (step - warmup_steps - stable_steps) / cooldown_steps
        lr = peak_lr * (1.0 - decay_progress)       # 线性衰减到0
    return lr
```
论文的核心发现：在 Stable 阶段（恒学习率，长达数万亿 tokens），验证损失缓慢下降而量化误差几乎不变；一旦进入 Decay 阶段，验证损失急剧下降但量化误差同时激增——这是因为学习率衰减使模型进入更尖锐的 loss 区域（Hessian 最大特征值上升），对量化引起的权重扰动更敏感。论文据此认为 WSD 优于 Cosine Decay，因为 WSD 可以更好控制末期学习率，而 Cosine 的末期学习率由峰值 lr 和训练步数隐式决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WSD 在现代 LLM 训练框架中实现为自定义 LR scheduler：(1) PyTorch 中通过 `LambdaLR` 或 `SequentialLR` 组合 `LinearLR(during_warmup) → ConstantLR(during_stable) → LinearLR(during_decay)`；(2) 在 plainLM (https://github.com/Niccolo-Ajroldi/plainLM) 等训练代码库中直接实现。参数配置：峰值学习率通常在 1e-3 到 6e-3 之间，warmup 1-2% 步数（如 1900 steps for 100B tokens），cooldown 约 10% 步数，衰减到 0。基于论文发现，选择 WSD 时应在训练过程中持续监控 PTQ 误差作为附加超参数指标——如果两个学习率候选在验证损失上表现相似，应优先选择在衰减后 PTQ 误差更低的那个。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness
