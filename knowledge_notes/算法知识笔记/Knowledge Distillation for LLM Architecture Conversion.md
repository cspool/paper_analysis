## Knowledge Distillation for LLM Architecture Conversion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Knowledge Distillation（知识蒸馏）是将大模型（teacher）的知识迁移到小模型或架构修改后的模型（student）的训练方法。在 LLM 上下文中有两类主要蒸馏形式：(1) **Logit-based KD**：匹配 teacher 和 student 的输出概率分布（KL 散度），`L_KD = E[Σ_t KL(q(·|x, y_<t) || p_θ(·|x, y_<t))]`；(2) **Feature-based KD**：对齐中间层 hidden states，`L_L2 = Σ_i ||o_i(θ) - t_i||²`。

**Self-distillation** 是 KD 的特殊形式，teacher 和 student 共享相同的参数空间（或 student 从 teacher 初始化），teacher = 原始模型，student = 修改后的模型。FCSD（FarSkip-Collective Self-Distill）就是典型的 self-distillation 应用——teacher 和 student 的权重形状完全相同，仅连接性不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip FCSD 使用的 logit-based self-distillation：

```
# Teacher: 原始 MoE 模型, Student: FarSkip 修改后模型
teacher = load_checkpoint("original_moe")  # 冻结，eval mode
student = FarSkipModel(teacher_config)     # 从 teacher 权重初始化
student.load_state_dict(teacher.state_dict())  # 同一参数空间，直接拷贝

for batch in dataloader:
    # Forward
    with torch.no_grad():
        teacher_logits = teacher(batch)  # [B, seq_len, vocab_size]
    student_logits = student(batch)

    # KL divergence loss (temperature=1)
    # p_teacher = softmax(teacher_logits)
    # p_student = softmax(student_logits)
    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    loss.backward()
    optimizer.step()
```

FCSD 消融发现：
- KL vs SFT：KL 显著优于 SFT（DeepSeek-V2-Lite: 62.0 vs 55.0 Avg），因为 KL 提供完整的概率分布信号而非 one-hot label
- KL vs KL + Intermediate L2：仅 KL 更好（68.2 vs 65.4），intermediate L2 的刚性约束可能阻碍模型适应新连接性
- 训练稳定性：KL distillation 后期可能出现 mode collapse（小的 teacher-student 差异产生大梯度），early stopping 是最简单有效的解决方案

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LLM 架构转换场景中（如 FarSkip、ladder-residual、Kraken），self-distillation 是对比 SFT 微调更优的选择——因为 student 权重已经在 teacher 的大部分能力上预训练好，主要任务是适应新的连接性（而非学习新知识）。KL loss 让 student 的每步输出"模仿"teacher，保留了 teacher 的生成行为和内部表征。实践中 batch-size 和 learning rate 的 sweep 至关重要（影响收敛速度和稳定性）。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
