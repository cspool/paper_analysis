## Expert Uniformity

术语解释
MoE tuning 中因复制 FFN 参数初始化 expert 导致的专家同质化现象：所有 expert 从相同的起点出发，经过训练后仍保持高度相似，并未发展出各自特化的功能，违背 MoE 架构的"不同 expert 专精不同任务"的核心设计理念。

术语是什么？
Expert Uniformity 由 EvoMoE 论文通过实验系统性揭示：在 MoE-tuning 训练完成后，随机打乱（shuffle）router 的各层 logits 进行推理，平均性能无明显下降（shuffle 5 次、8 次的 AVG 性能分别为 65.5 和 65.2，与原始 65.5 相当）。这说明 expert 之间没有实质性的功能差异——router 选哪个 expert 对结果影响不大。

根本原因：MoE-tuning 的 expert 初始化方式是"复制原始 dense model 的 FFN 参数 N 份"。所有 expert 起点相同、接收相同梯度、在相同数据上训练 → 训练后趋同。

EvoMoE 通过 Expert Evolution 解决：不同 β 值使各 expert 以不同速率吸收梯度更新 → 自然产生参数分化 → 功能分化。

从算法pipeline角度拆解术语：
Expert Uniformity 的验证实验：
```
# 推理时随机打乱 router logits 评估
for shuffle_trial in [1..8]:
    for layer in MoE_layers:
        # 原本每个 token 的 router logits 对应确定 expert
        logits = layer.router(token_hidden)  # [B, S, N]
        # 随机打乱 logits（在 expert 维度内）
        perm = random_permutation(N)
        logits = logits[:, :, perm]
        selected_expert = argmax(logits)
    # 评估：性能几乎不变 → Expert Uniformity 存在
```

在 MoE LLM 中也可通过以下指标诊断：
- Expert 间的参数 cosine similarity（越接近 1 越均匀）
- Expert 的激活分布差异（KL divergence）
- 不同 expert 在特定 benchmark 上的独立性能差异

术语一般如何实现？如何使用？
- Expert Uniformity 是一个需要诊断和避免的问题，而非可用的方法
- 诊断：shuffle router test、expert 相似度分析、独立 expert 评估
- 解决方案：Expert Evolution（EvoMoE）、Noise Initialization（效果不佳）、Dropout（效果不佳）、对比损失（NCE loss，各 benchmark 表现不一致）、local loss 增加 router entropy（效果有限）
- Expert Uniformity 与 Expert Collapse 不同：Collapse 指少数 expert 接收几乎所有 token（负载不均衡），Uniformity 指 expert 间参数趋同但负载可能均衡

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---
