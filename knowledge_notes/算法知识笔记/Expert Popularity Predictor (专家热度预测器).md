## Expert Popularity Predictor (专家热度预测器)

术语解释
APTMoE 提出的一种轻量级预测模块，在 MoE 层的 gate operation 之前若干层插入，结构与该层 gate operation 相同（linear + softmax），通过微调训练提前预测目标层的 expert 激活分布，为 inter-layer loading 提供决策依据。

术语是什么？
预测器是 gate operation 的"影子"副本，放置在目标层之前（如 Mixtral-8x7B 中提前 1 层，NLLB-MoE 中提前 4 层）。它接收当前层的中间 hidden states 作为输入，输出目标层各 expert 的预测激活概率。预测器不修改原模型结构，不影响原始 MoE 的计算结果。

从算法pipeline角度拆解术语。
```
# 假设 target_layer = l, predictor 插入在第 l-δ 层
# δ=1 for Mixtral (全 MoE layers)
# δ=4 for NLLB-MoE (每4层一个 MoE layer)

# Training (predictor fine-tuning)
predictor = copy_gate_structure(layer_l.gate)       # 复制结构
predictor.weights = layer_l.gate.weights.clone()    # 复制权重初始化
for step in range(training_steps):
    hidden_early = model.forward_up_to(layer_{l-δ})  # 提前若干层的 hidden states
    hidden_target = model.forward_up_to(layer_l)     # 目标层的 hidden states
    pred_probs = predictor(hidden_early)              # 预测的 expert 概率
    real_probs = layer_l.gate(hidden_target)          # 真实的 expert 概率
    loss = CrossEntropy(pred_probs, real_probs)       # 监督信号
    loss.backward(); optimizer.step()

# Inference (预测用于 inter-layer loading)
pred_probs = predictor(hidden_early)
sorted_experts = sort_by_predicted_popularity(pred_probs)
for expert in sorted_experts:
    if should_load_to_gpu(expert, Equation_1):
        interlayer_queue.add(expert)
```
预测器 overhead：FLOPs = 2sdE（gate 同结构），expert FLOPs = 8sdh，h >> E（Mixtral h=14336, E=8），因此 predictor 额外计算可忽略。训练收敛时间：Mixtral ~0.93s (700 steps)，NLLB ~0.18s。

术语一般如何实现？如何使用？
- PyTorch nn.Linear + Softmax 实现，与 gate 结构一致
- 使用 KL Divergence 或 Cross Entropy 作为训练损失
- 预测准确率取决于：expert 数量（越少越高）、预测提前量（越近越准）
- Expert 级准确率比 token 级准确率更有实际意义（只需知道哪些 expert 是 low-demand）
- 在 Mixtral-8x7B 上 least 2/8 experts 预测准确率 100%，在 NLLB-MoE 上 least 32/128 experts 准确率 94%

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
