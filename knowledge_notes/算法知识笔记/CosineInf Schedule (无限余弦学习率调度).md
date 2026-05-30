## CosineInf Schedule (无限余弦学习率调度)

术语解释
CosineInf (Cosine Infinite) Schedule 是由 Ibrahim et al. (2024) 提出的用于持续预训练（CPT）的学习率调度方案。与标准 Cosine Annealing（需要预先指定总训练步数）不同，CosineInf 在 cosine decay 后进入恒定学习率阶段（constant phase），允许无限期继续训练。当需要部署时，从 constant phase 执行 annealing（衰减到 η_min），然后可以从 pre-annealed constant-phase checkpoint 恢复继续训练。

术语是什么？
CosineInf 包含四个阶段：
$$ \eta(t) = \begin{cases} \text{Linear warmup} & 0 \le t < N_w \\ \text{Cosine cooldown: } \eta_{const} + (\eta_{max} - \eta_{const}) \cdot \frac{1 + \cos(\pi(t-N_w)/(N_c-N_w))}{2} & N_w \le t < N_c \\ \eta_{const} & N_c \le t < N_d \\ \text{Exponential decay to } \eta_{min} & t \ge N_d \end{cases} $$

本文的参数：
- Pre-training: total_iters=192720, η_max=3e-4, η_min=3e-5, η_const=1.65e-4, warmup=1%, cooldown=70%, const=10%
- CPT: total_iters=95370, η_max=3e-4, η_min=3e-5, η_const=1.65e-4, warmup=1%, const=80%, cooldown=0%

与标准 Cosine Decay 的关键差异：
- Cosine Decay（full re-training）：total_iters=288090，warmup 1% → decay to η_min，需预先知道总步数
- CosineInf（CPT）：不随时间衰减到 η_min，保持 constant phase 允许无限继续

从算法pipeline角度拆解术语：
```python
def cosine_inf_schedule(step, total_iters, eta_max, eta_min, eta_const,
                        warmup_pct, cooldown_pct, const_pct):
    N_w = int(total_iters * warmup_pct)
    N_c = int(total_iters * cooldown_pct)
    N_d = int(total_iters * (warmup_pct + cooldown_pct + const_pct))

    if step < N_w:                                    # Phase 1: Warmup
        return eta_max * step / N_w
    elif step < N_c:                                  # Phase 2: Cooldown
        progress = (step - N_w) / (N_c - N_w)
        return eta_const + (eta_max - eta_const) * (1 + cos(pi * progress)) / 2
    elif step < N_d:                                  # Phase 3: Constant
        return eta_const
    else:                                             # Phase 4: Annealing (optional)
        return eta_min + (eta_const - eta_min) * exp_decay(step - N_d)
```

术语一般如何实现？如何使用？
- **CPT 从 constant phase 恢复**：在 constant phase 保存 checkpoint（η=η_const），CPT 时直接从此 checkpoint 以 η_const 继续训练，无需 re-warming → 避免 re-warming 引起的遗忘
- **优于 Cosine Decay**：从衰减 checkpoint 开始的 Cosine Decay CPT 需要在 η_min 上 re-warm 到 η_max → 学习率大幅波动 → 遗忘更严重（本文 Figure 2 验证）
- **Smooth transition**：CosineInf 在 pre-training 和 CPT 之间的 LR 过渡平滑（η_const → η_const），无需大幅调整学习率
- **与 Cosine Decay CPT 的对比**：本文 ablation (Sec 5.1) 显示 CosineInf 从 non-decayed checkpoint 的 CPT 在 FineWeb 遗忘上显著优于 Cosine Decay 从 decayed checkpoint 的 CPT

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
