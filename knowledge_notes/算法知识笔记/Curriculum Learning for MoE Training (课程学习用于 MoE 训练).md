## Curriculum Learning for MoE Training (课程学习用于 MoE 训练)

术语解释
Curriculum Learning 在 MoE 训练中特指基于 token 复杂度对训练数据重新排序的策略。由 Li et al. (EMNLP 2023) 在 Adaptive Gating 论文中首次应用于 MoE 训练场景，用于解决 adaptive gating 中不同 token 使用不同数量 expert 导致的 batch 内计算时间不均问题。

术语是什么？
在 adaptive gating 中，虽然多数 token 仅需 top-1 expert（计算量减半），但 Attention 层需要完整序列输入，导致训练 step 时间由 batch 中最慢的 top-2 token 决定。即使 80% token 已提前完成 MoE 计算，仍需等待剩余 20% top-2 token。课程学习通过将相似复杂度的训练样本分组，减少同 batch 内 top-2 token 比例的方差，缓解"快 token 等待慢 token"问题。

复杂度度量：对每个训练样本 d，定义复杂度向量 C_d = [r_0^d, r_1^d, ..., r_L^d]，其中 L 为 MoE 层数，r_i 为第 i 层中由 top-2 expert 处理的 token 占比。

从算法pipeline角度拆解术语。
```
# Curriculum Learning: Training Data Reordering
def reorder_training_data(all_samples, model, T):
    # Step 1: 计算每个样本的复杂度向量
    C_samples = []
    for sample in all_samples:
        C = []  # complexity vector
        for layer in model.moe_layers:
            gate_output = layer.gate(sample.embeddings)
            R = softmax(gate_output, dim=-1)
            top1_prob, top2_prob = R.topk(2, dim=-1).values[:, 0], R.topk(2, dim=-1).values[:, 1]
            prob_diff = top1_prob - top2_prob
            # r = 需 top-2 expert 的 token 比例
            r = (prob_diff <= T).float().mean().item()
            C.append(r)
        C_samples.append(C)

    # Step 2: 找到最简样本作为参考
    ref_idx = argmin([sum(C) for C in C_samples])
    ref_vec = C_samples[ref_idx]

    # Step 3: 按余弦相似度降序排列
    similarities = [cosine_sim(C, ref_vec) for C in C_samples]
    return [all_samples[i] for i in argsort(similarities, descending=True)]
```

术语一般如何实现？如何使用？
- 第一个 epoch 使用随机数据顺序让 model 产生初始 gate 决策
- 每 epoch 结束后重新计算复杂度向量并重排数据
- 排序依据：以最少 top-2 token 的样本为参考，余弦相似度降序排列
- 实验效果：去除 curriculum learning 后，训练时间平均膨胀 13.7%，推理性能最大下降 0.21 F1
- 适用于 adaptive gating 场景；固定 top-k MoE 训练不需要此策略
- 复杂度向量计算需完整前向传播一次，overhead 与一个 eval epoch 相当

涉及论文标题：
- Adaptive Gating in Mixture-of-Experts based Language Models

---
