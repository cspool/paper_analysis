## WSD Learning Rate Scheduler（Warmup-Stable-Decay 学习率调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

WSD (Warmup-Stable-Decay) 学习率调度器是 MiniCPM (Hu et al., 2024) 提出的三阶段学习率策略：**Warmup 阶段**（前 10% steps，LR 从 0 线性增至 max_lr）→ **Stable 阶段**（中间 ~70% steps，LR 保持恒定为 max_lr）→ **Decay 阶段**（后 20% steps，LR 通过 cosine annealing 衰减至 0.1×max_lr）。与 cosine decay（全程衰减）相比，WSD 在 stable 阶段允许模型在最高学习率下持续学习，有利于 LLM 预训练中的充分优化。

从算法pipeline角度拆解术语：

```
// WSD Scheduler 伪代码
max_lr = sweep({1,2,5}×10^{-3,-4,-5}) on MHA baseline
total_steps = D_train / (batch_tokens)  // e.g. 20B / 512K
warmup_steps = 0.10 × total_steps
decay_steps  = 0.20 × total_steps
stable_steps = total_steps - warmup_steps - decay_steps

for step in range(total_steps):
    if step < warmup_steps:
        lr = max_lr × (step / warmup_steps)         // linear warmup
    elif step < warmup_steps + stable_steps:
        lr = max_lr                                   // stable plateau
    else:
        progress = (step - warmup_steps - stable_steps) / decay_steps
        lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
        // cosine decay to min_lr = 0.1 × max_lr
```

术语一般如何实现？如何使用？

本文在 GQA scaling experiments 中使用 WSD：10% warmup, 20% decay, 搭配 AdamW (β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0)。max_lr 对每个模型大小在 MHA baseline 上 grid search {1,2,5}×10^{-3,-4,-5} 获得，跨 GQA 配置复用。长上下文 adaption 阶段（T=4K→128K）使用更低的 max_lr=1e-5 + 新 optimizer state 防止 catastrophic forgetting。WSD 在 20:1 的 Chinchilla-optimal 数据比例下运行。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---
