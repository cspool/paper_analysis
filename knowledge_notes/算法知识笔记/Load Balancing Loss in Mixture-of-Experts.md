## Load Balancing Loss in Mixture-of-Experts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load Balancing Loss（负载均衡损失）是 MoE 训练中防止 expert 负载不均衡的辅助损失函数。由于 Router 在训练中可能收敛到总是选择少数几个 expert（导致其他 expert 不被训练而导致 collapse），Load Balancing Loss 约束每个 expert 处理大致等量的 tokens。标准形式为 `L_balancing = K * Σ_{i=1..K} F_i * G_i`，其中 F_i 是分配给 expert i 的 token 比例，G_i 是 expert i 的平均 routing probability。该损失通过鼓励均匀的 token 分配来避免 expert 过载/空闲问题。

LTDR 论文揭示了 Load Balancing Loss 在 multi-modal（vision-language）场景中的关键缺陷：vision tokens 服从 long-tailed distribution（大量低信息 background + 少量高信息 foreground），load balancing 会将 sparse 但关键的 foreground (tail) tokens 均匀分散到不同 expert，阻止 expert 对视觉关键信息进行专业化学习。实验数据显示移除 vision TER 的 load balancing 直接提升性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Standard Load Balancing (Switch Transformer, GShard)
# Applied to ALL tokens uniformly
for batch in training_data:
    # ... router forward ...
    for i in 1..K:  # for each expert
        F_i = count(tokens routed to expert i) / total_tokens
        G_i = mean(softmax(logits)[:, i])
    L_balancing = K * Σ_{i=1..K} F_i * G_i
    L_total = L_task + α * L_balancing  # α = coefficient (0.01)

# LTDR Modality-specific Load Balancing
# Only applied to language tokens
for batch in training_data:
    # ... router forward ...
    for i in 1..K:
        F_i = count(language_tokens routed to i) / N_lang
        G_i = mean(softmax(lang_logits)[:, i])
    L_balancing = K * Σ F_i(T) * G_i(T)  # vision tokens excluded!
    L_total = L_task + α * L_balancing
```

**消融实验验证**：
- Vision load balancing coefficient: 0.01 (standard) vs 0.001 (reduced) vs 0 (LTDR, removed)
- 结果：reduced (0.001) 不如 complete removal (0)，因为即使系数缩小 10x 仍然对 vision tail tokens 的 distribution 产生约束
- Strategy-swap 实验：将 MsDaR（移除 load balancing）分别应用于 vision 和 language 侧。Language+MsDaR 导致性能波动（因为语言确实服从 uniform distribution），Vision+MsDaR 稳定提升

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现方式**：在训练 loss 中作为 additive auxiliary loss，系数通常设为 0.01（Switch Transformer）。PyTorch 中作为额外 loss term 加到 `L_total.backward()`
- **使用方式**：MoE-LLaVA / GShard / Switch Transformer 均使用此机制。LTDR 的改动是在计算 F_i 和 G_i 时 filter 掉 vision tokens（仅在 token dispatch 时标记 modality type）
- **作用范围**：影响 expert 的 token 分配分布，从而影响 expert 专业化程度。对 language 有效（uniform distribution 适配），对 vision 有害（long-tailed distribution 不适用）

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
