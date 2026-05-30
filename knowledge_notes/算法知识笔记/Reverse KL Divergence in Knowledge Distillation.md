## Reverse KL Divergence in Knowledge Distillation

术语解释
Reverse KL divergence 是 KD 中的一种分布匹配目标，形式为 `D_KL(qθ || p)` 而非传统的正向 `D_KL(p || qθ)`。其中 p 是教师分布，qθ 是 student 分布。Reverse KL 具有"mode-seeking"行为：student 倾向于聚焦教师的少数高概率 mode，而非尝试覆盖教师的所有 mode（forward KL 的"mean-seeking"行为）。

术语是什么？
正向 KL vs 反向 KL：
- **Forward KL**: `D_KL(p || qθ) = Σ p(y) log(p(y)/qθ(y))` → 惩罚 qθ 在 p 有概率处给低概率 → mean-seeking → student 覆盖教师所有 mode → 可能生成"平均化"输出
- **Reverse KL**: `D_KL(qθ || p) = Σ qθ(y) log(qθ(y)/p(y))` → 惩罚 qθ 在 p 低概率处给高概率 → mode-seeking → student 聚焦教师的高置信度 mode → 生成更精准的输出

在 LLM 文本生成 KD 中，reverse KL 通常优于 forward KL，因为：生成任务有大量低概率但合理的长尾 token，forward KL 会迫使 student 学习这些长尾分布，导致"模糊"或"保守"的生成。

从算法pipeline角度拆解术语：
```
# Reverse KL KD (on-policy, per token)
def reverse_kl_kd_step(x, teacher, student):
    # Student autoregressive generation
    y_tokens = []
    for t in range(max_len):
        student_probs = student(x + y_tokens)         # qθ over vocab
        y_t = sample(student_probs)                    # on-policy sampling
        y_tokens.append(y_t)
    
    # Teacher evaluation on student-generated tokens
    teacher_probs = teacher(x + y_tokens)             # p over vocab
    
    # Reverse KL: D_KL(qθ || p) per token position
    loss = 0
    for t in range(len(y_tokens)):
        loss += sum(student_probs[t][w] * log(student_probs[t][w] / teacher_probs[t][w])
                    for w in vocab)
    return loss
```

术语一般如何实现？如何使用？
- MiniLLM (Gu et al., 2024) 使用 Policy Gradient 优化 reverse KL（因 student 采样不可微）
- GKD (Agarwal et al., 2024) 直接使用 reverse KL 作为可微损失（因 teacher 和 student 共享 tokenizer，在相同 token 序列上计算分布差异）
- KA/SAR (Kim et al., 2025) 使用 reverse KL + student on-policy generation
- 与 forward KL (Sanh, 2019) 不同，reverse KL 要求 KD 在 response tokens 上而非 full sequence 上计算，因为 prompt 部分 teacher 和 student 分布不具有可比性

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
