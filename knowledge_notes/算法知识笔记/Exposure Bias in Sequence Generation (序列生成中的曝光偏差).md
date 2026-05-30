## Exposure Bias in Sequence Generation (序列生成中的曝光偏差)

术语解释
Exposure bias（曝光偏差）是自回归序列生成模型的训练-推理不一致问题：训练时使用 ground-truth token 作为输入（teacher forcing），推理时使用自身生成的 token 作为输入。当 student 在推理时生成了一个训练中从未见过的错误 token，后续 token 的生成会基于这个错误前缀，导致错误级联放大 (error accumulation)。

术语是什么？
具体表现：
- 训练：`P_θ(y_t | y_{<t}^{gt})`——以 ground-truth history 为条件
- 推理：`P_θ(y_t | y_{<t}^{gen})`——以 self-generated history 为条件
- 差距：`y_{<t}^{gt}` ≠ `y_{<t}^{gen}` 导致分布的系统性偏移

在 KD 中的影响：如果 student 仅在 teacher 生成的 token 序列上训练（off-policy），它习惯于看到"完美"的 context，推理时遇到自己的"非完美"生成就会产生偏离。On-policy KD 通过让 student 在自己的生成序列上训练来消除此偏差。

从算法pipeline角度拆解术语：
```
# Exposure Bias Illustration
# Training (teacher forcing with ground truth):
#   Student sees:  "The cat sat on the..."
#   All previous tokens are correct from ground truth

# Inference (autoregressive generation):
#   Step 1: Student generates "The" ✓
#   Step 2: Student generates "dog" ✗ (wrong token!)
#   Step 3: Student now conditions on "...The dog..." → cascade of errors
#   Student has never trained on sequences starting with its own errors
```

```
# Mitigation via On-policy KD
for x in training_data:
    y_student_tokens = []
    for t in range(max_len):
        probs = student(x + y_student_tokens)    # student's own distribution
        next_tok = sample(probs)
        y_student_tokens.append(next_tok)         # student's own generation
    
    # Now train on student's own (possibly imperfect) generation
    teacher_probs = teacher(x + y_student_tokens)
    loss = KL_div(student(x + y_student_tokens) || teacher_probs)
```

术语一般如何实现？如何使用？
- 解决方案：(a) on-policy KD (GKD, KA, SAR)；(b) Scheduled Sampling——训练时以概率 ε 使用模型自身生成的 token 替代 ground truth；(c) Reverse KL divergence 的 mode-seeking 属性降低对"平均"分布的依赖
- KA 和 SAR 通过 student 自生成 pseudo-target 实现 on-policy 训练
- Exposure bias 在长序列生成中尤为严重（错误随序列长度累积）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
