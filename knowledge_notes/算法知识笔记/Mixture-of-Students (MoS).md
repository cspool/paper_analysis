## Mixture-of-Students (MoS)

术语解释
Mixture-of-Students (MoS) 是 DeepSpeed-MoE 提出的 MoE-to-MoE 知识蒸馏方法，通过 Staged Knowledge Distillation 将大 MoE 教师模型蒸馏到小 MoE 学生模型（减少层数，保持 MoE 架构），保留稀疏推理优势。与 MoE-to-dense 蒸馏不同，MoS 学生仍为 MoE 结构。

术语是什么？
传统 KD 大多用于 dense 模型或将 MoE 蒸馏为 dense（丢失稀疏推理加速）。MoS 的核心创新：(1) 教师和学生均为 MoE（PR-MoE），学生仅减少深度（如 24→21 层），保留专家结构；(2) 发现全程 KD 在预训练后期伤害精度（学生容量不足导致 underfitting：无法同时最小化 CE loss 和 KD loss），提出 **Staged KD**：前 400K steps 使用 KD + CE loss，之后停用 KD 仅优化 CE loss。

从算法pipeline角度拆解术语：
```
# MoS Staged KD 训练流程
# Teacher: 1.3B+PR-MoE+L24 (31B params, 24 layers)
# Student: 1.3B+PR-MoE+L21 (27B params, 21 layers, 12.5% depth reduction)

For step = 1 to total_steps:
    x, y = next_batch()
    
    teacher_logits = Teacher(x)                    # teacher inference (no grad)
    student_logits = Student(x)                    # student forward
    
    L_CE = CrossEntropyLoss(student_logits, y)     # 标准语言模型损失
    L_KD = KLDivergence(student_logits, teacher_logits)  # 蒸馏损失
    
    if step <= 400K:
        L = L_CE + α * L_KD                        # Staged KD Phase 1: 使用蒸馏
    else:
        L = L_CE                                    # Staged KD Phase 2: 仅标准 LM loss
    
    L.backward()
    optimizer.step()

# 关键发现：Full KD（全程使用 KD）最终精度低于 No KD 的 PR-MoE baseline
# Staged KD 解决了 underfitting，学生保留 99.1-99.5% 教师性能
```

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/DeepSpeed (DeepSpeed-MoE 组件)
- 知识蒸馏损失：L = L_CE + α·L_KD，其中 L_KD 为 KL 散度
- 关键超参数：KD 停止步数（如 400K steps）、权重 α
- 学生模型深度减少需配合 Staged KD 才能保持精度；直接减少深度不进行 KD 会导致显著精度损失（LAMBADA 下降 1.3 点，BoolQ 下降 7.5 点）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
