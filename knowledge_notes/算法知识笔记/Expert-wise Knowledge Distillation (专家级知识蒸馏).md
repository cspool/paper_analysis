## Expert-wise Knowledge Distillation (专家级知识蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert-wise Knowledge Distillation 是 MoE-Pruner (Xie et al., 2024) 提出的剪枝后性能恢复方法。与标准 KD（仅在 logits 层计算 KL/MSE）不同，它在 MoE 所有 l 层、每层 n 个 expert 的输出层面逐 expert 计算 teacher（未剪枝 pretrained）和 student（剪枝后）的 MSE。损失 L_KD = L_CE + λ * Σ_{j=0}^{l-1} Σ_{i=0}^{n-1} MSE(E_it^j, E_is^j)，λ 初始化为 L_CE / L_expert 以平衡两条损失。

从算法pipeline角度拆解术语：
```
for batch in data (1000 C4 samples):
    # Teacher forward (no_grad)
    for layer j: teacher_out[j][i] = expert_i_j(x) for i in 0..n-1
    # Student forward
    L_CE = CrossEntropy(student_logits, labels)
    L_expert = Σ_j Σ_i MSE(teacher_out[j][i], student_expert_i_j_output)
    λ = L_CE.item() / L_expert.item()
    L_total = L_CE + λ * L_expert
    optimizer.step()  # sparsity mask 保持
```
Mixtral-8x7B 50% 稀疏度：Expert-wise KD 将 zero-shot 准确率从 67.23 恢复到 68.40（原始 69.16），维持 99% 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 Llama-Factory 中实现，full-parameter fine-tuning 保持 sparsity mask。lr=2e-5, cosine scheduler, 3 epochs, 16×H100-80GB, ~1小时。仅需 1000 条 C4 训练样本。
- Pretrained model 是天然 teacher（同结构直接对应蒸馏）。局限：需 16×H100 同时加载 teacher+student；full-parameter fine-tuning 计算不减；λ 仅启发式初始化。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
