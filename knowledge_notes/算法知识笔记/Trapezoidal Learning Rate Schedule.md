## Trapezoidal Learning Rate Schedule

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Trapezoidal Learning Rate Schedule（梯形学习率调度）是介于 constant 和 cosine 之间的 LR 调度策略，由 Hägele et al. (2024) 提出。形状为：warmup → constant plateau → linear decay → 0。核心优势：中间 checkpoint 可复用于不同训练时长的实验——短训练在 plateau 阶段结束时取 checkpoint，无需像 cosine schedule 那样因 LR 值不同而引入 bias。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 的具体配置：

```
total_tokens = D_total  # 计划训练的总 token 数

lr_schedule(tokens_trained):
    if tokens_trained < 130M:           # warmup phase
        lr = peak_lr * (tokens_trained / 130M)  # linear warmup
    elif tokens_trained < 0.8 * D_total:  # plateau: 前 80% 训练
        lr = peak_lr                       # constant
    else:                                  # decay: 最后 20%
        progress = (tokens_trained - 0.8*D_total) / (0.2*D_total)
        lr = peak_lr * (1 - progress)      # linear decay to 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 Scaling Laws 研究中特别有价值：需要训练多个不同 D 的模型时，短的 runs 可复用长的 runs 的中间结果，大幅降低总计算量
- Hoffmann et al. (2022) 指出 cosine schedule 的中间 checkpoint 会在 scaling law fitting 中引入系统性 bias（不同 D 的 checkpoint 处于不同 decay 阶段），trapezoidal 的 plateau 阶段避免了此问题
- Hägele et al. (2024) 证明 trapezoidal 的性能与 cosine 相当，但为 scaling law 研究提供了显著的实验效率优势

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
