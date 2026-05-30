## Generalized Knowledge Distillation (GKD)

术语解释
Generalized Knowledge Distillation (GKD) 是 Agarwal et al. (2024, ICLR) 提出的知识蒸馏框架，统一了 on-policy vs off-policy data 和 forward vs reverse KL divergence 的选择，通过广义 Jensen-Shannon (JS) divergence 提供灵活的 teacher-student 知识迁移机制。

术语是什么？
GKD 的两个核心维度：
1. **数据选择**：支持 teacher-generated (off-policy)、固定数据集 (fixed) 或 student-generated (on-policy) 的序列
2. **目标函数**：广义 JS divergence `D_JS^β(p||q) = β·D_KL(p||m) + (1-β)·D_KL(q||m)` where `m = β·p + (1-β)·q`。当 β=0 → forward KL (mode-seeking)，β=1 → reverse KL (mode-covering)，β=0.5 → 原始 JS divergence

GKD 的关键创新：使用 student-generated on-policy 输出替代教师输出进行蒸馏，解决 exposure bias——即 student 在推理时遇到训练中未见过的自回归错误积累。

从算法pipeline角度拆解术语：
```
# GKD (student on-policy, reverse KL, β=1)
def gkd_step(x, teacher, student):
    # 1. Student generates on-policy response
    y_student = student.generate(x, max_new_tokens=512)
    
    # 2. Teacher evaluates student-generated sequence
    teacher_logits = teacher(x, y_student)     # teacher distribution p(y|x)
    student_logits = student(x, y_student)     # student distribution qθ(y|x)
    
    # 3. Reverse KL divergence (or JS divergence)
    L = KL_div(student_logits || teacher_logits)  # reverse KL
    # or L = JS_div(teacher_logits, student_logits, β)
    
    student.backward(L)
```

术语一般如何实现？如何使用？
- Kim et al. (2025) 的 KA 和 SAR 均基于 GKD 的 on-policy + reverse KL 框架构建
- 对比 baseline KD (forward KL + teacher-generated data)，GKD 在 dense teacher 蒸馏中明显更好
- GKD 在 MoE teacher 上的表现不如 KA/SAR，验证了 MoE 专用 KD 方法的必要性
- 实现需要 teacher 和 student 共享 tokenizer（用于 token-level distribution matching）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
