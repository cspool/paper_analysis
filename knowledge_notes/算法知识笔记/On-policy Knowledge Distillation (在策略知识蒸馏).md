## On-policy Knowledge Distillation (在策略知识蒸馏)

术语解释
On-policy KD 是使用 student 模型自身生成的输出（而非 teacher 生成的或数据集中固定的输出）作为蒸馏目标的训练策略。与 off-policy KD（蒸馏数据由 teacher 生成或来自固定数据集）相比，on-policy KD 直接针对 student 的推理分布进行优化，有效缓解 exposure bias。

术语是什么？
Off-policy vs On-policy：
- **Off-policy**: 蒸馏数据来自 teacher 输出或固定数据集 → student 的训练分布 ≠ 推理分布 → exposure bias
- **On-policy**: 蒸馏数据来自 student 自身采样 → student 在训练中遇到的错误模式与其推理时一致 → 蒸馏目标与推理分布一致

On-policy KD 流程（以 GKD 为例）：
1. Student 自回归生成 pseudo-target 序列 `y ~ qθ(·|x)`
2. Teacher 在相同 (x, y) 上计算 logits `p(y|x)`
3. Student 在相同 (x, y) 上计算 logits `qθ(y|x)`
4. 计算分布损失：`L = D_KL(qθ || p)` 或 `L = D_JS(qθ, p)`（使用 student 生成的 y 上的 token-level 分布）

从算法pipeline角度拆解术语：
```
# On-policy vs Off-policy KD comparison
def kd_comparison(teacher, student, x_dataset):
    # Off-policy (traditional KD, Sanh 2019)
    for x, y_golden in x_dataset:                # fixed data
        loss = KL_div(teacher(x, y_golden) || student(x, y_golden))
    
    # Off-policy (teacher-generated)
    for x in x_dataset:
        y_teacher = teacher.generate(x)           # teacher output
        loss = KL_div(teacher(x, y_teacher) || student(x, y_teacher))
    
    # On-policy (GKD, KA, SAR)
    for x in x_dataset:
        y_student = student.generate(x)           # student output!
        loss = KL_div(student(x, y_student) || teacher(x, y_student))
        # teacher evaluates student's own generation
```

术语一般如何实现？如何使用？
- GKD 使用 fixed + on-policy 混合数据训练
- KA 和 SAR 使用纯 on-policy 数据（student 生成 pseudo-target）
- On-policy 训练的关键：teacher 和 student 必须使用相同 tokenizer 以在相同 token 序列上进行分布比较
- 限制：on-policy 生成增加训练计算开销（每次迭代需 student autoregressive 生成）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
