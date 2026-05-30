## FCSD (FarSkip-Collective Self-Distill)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FCSD（FarSkip-Collective Self-Distill）是将预训练 MoE 模型转换为 FarSkip-Collective 架构连接性的知识蒸馏方法。以原始模型（未修改连接性）作为 teacher，FarSkip 修改后的模型作为 student，使用 KL 散度 loss 训练约 10B tokens 即可恢复原始模型约 97.5-99% 的下游任务准确率。直接加载原始权重到 FarSkip 架构中会导致性能崩溃（MMLU 降至随机基线，HumanEval+ 降至 0%），因为模型接收到的输入激活值分布与训练时完全不同（OOD）。

FCSD 配方：
- **Loss**：`L_KD(θ) = E_x[Σ_t KL(q(·|x, y_<t) || p_θ(·|x, y_<t))]`，以原始模型 q 为 teacher
- **优化器**：AdamW + cosine-annealing LR scheduler + 1000-step warmup
- **Batch-size**：从 {2^16, 2^17, 2^18} 中 sweep 选择
- **Learning rate**：从 {2e-5, 4e-5, 8e-5} 中 sweep 选择
- **训练数据**：GenQA + Infinity Instruct SFT 数据
- **Early stopping**：使用 MBPP+ 每 1000 steps 评估，patience=20, delta=2%

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FCSD 训练流程：

```
teacher = load_original_moe_checkpoint()  # 冻结
student = convert_to_farskip_architecture(teacher)  # 仅修改 skip connections

# Hyperparameter sweep (各 2000 steps)
for bs in [2^16, 2^17, 2^18]:  # batch-size sweep, lr=2e-5
    train(student, teacher, bs, lr=2e-5, steps=2000)
    select by training loss
for lr in [2e-5, 4e-5, 8e-5]:  # lr sweep with best bs
    train(student, teacher, best_bs, lr, steps=2000)
    select by training loss

# Full training with best config, up to 10B tokens
for step in range(max_steps):
    x = next_batch(best_bs)  # SFT data
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    loss = KL(teacher_logits || student_logits)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        mbpp_score = evaluate(student, "MBPP+")
        if early_stop(mbpp_score, patience=20, delta=0.02):
            break
```

消融发现（Tab. 2, Qwen-3-30B MoE, 500M tokens）：
- KL alone 效果最好（Avg-11: 68.2 → 原始 75.9）
- KL + Intermediate L2 反而更差（65.4），可能是 intermediate 对齐过于刚性
- SFT only 显著劣于 KL（58.1），尤其在生成任务上（HEval+ 仅 1.2）
- 冻结 embedding/LM-head 无明显影响（67.6）
- 仅转换部分层（如 50%/75%/90%）使任务更容易，但减少了重叠机会

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FCSD 的优势：
- **高效**：仅需 < 10B tokens（vs 从头预训练需数万亿 tokens），约 100-1000× 更便宜
- **通用**：适用于任何 MoE 模型，不依赖强 teacher 模型（self-distillation）
- **鲁棒**：KL loss 提供细粒度训练信号，即使 SFT 数据质量不高也能恢复模型表征
- **稳定性挑战**：训练后期可能出现 mode collapse（teacher-student 差异导致大梯度），通过 MBPP+ early stopping 解决

FCSD 对比 SFT 的关键洞察：KL 散度匹配 teacher 的完整概率分布，提供比 one-hot SFT label 更丰富的训练信号。尤其在 FarSkip 场景中，student 的权重已在 teacher 的大部分任务上训练好，主要需要适应新的连接性——KL distillation 正是为此设计的"软对齐"方法。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
