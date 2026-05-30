## Expert Evolution

术语解释
EvoMoE 提出的一种 MoE expert 初始化策略：从单个可训练 FFN 通过动态混合先验参数和梯度更新，逐步演化出多个功能多样的 MoE expert，替代传统的"复制 FFN 初始化"方式。

术语是什么？
Expert Evolution 解决 MoE-tuning 中的 **Expert Uniformity** 问题——传统方法直接复制（replicate）dense model 的 FFN 参数来初始化多个 expert，导致 expert 在训练后趋同，失去 MoE 架构的多样化优势。Expert Evolution 的核心是 EMA（指数移动平均）形式的参数演化：

$$\theta_n \leftarrow \beta \cdot \theta_1 + (1 - \beta) \cdot \nabla \theta_1, \quad n = 2, 3, \dots, N$$

其中 $\theta_1$ 是唯一可训练的 Expert 1（使用 Stage I 输出初始化），$\theta_n$ 是演化生成的专家，$\nabla \theta_1$ 是 Expert 1 的梯度更新，$\beta \in [0,1]$ 为演化率。不同 expert 使用不同的 $\beta$ 范围（如 [0.9,0.99]、[0.8,0.89]、[0.7,0.79]），从而以不同速率吸收梯度信息，自然产生功能分化。演化后的 expert 和所有其他 LLM/MLP 参数保持冻结。

实验验证：独立评估每个演化后的 expert（不使用 router）发现 Expert 2/3/4 在多个 benchmark 上一致优于 Expert 1（原始 FFN），即使 $\beta=0.9$（仅保留 10% 梯度更新）也能提升性能，证明演化产生的多样性是有效的前非随机的。

从算法pipeline角度拆解术语：
```
# Stage II: Expert Evolution
# 输入：Stage I 输出的密集模型，θ_1 = FFN 参数（Expert 1）
# N = 4 个 expert，top-1 routing

for step = 1 to total_steps:
    # 前向：仅使用 θ_1 作为活跃 expert（Stage II 不开 MoE）
    h = MSA(LN(x)) + x
    y = FFN_1(LN(h)) + h   # FFN_1 即 θ_1
    loss = L_regressive + α * L_aux

    # 反向：仅更新 θ_1
    ∇θ_1 = backward(loss)
    θ_1 = optimizer_step(θ_1, ∇θ_1)

    # 演化其他 expert（每个 step 都执行）：
    for n in [2, 3, 4]:
        β_n = random_uniform(low_n, high_n)
        θ_n ← β_n * θ_1 + (1 - β_n) * ∇θ_1  # EMA 混合

# 输出：4 个具有功能差异的 FFN expert 参数
```

关键设计：
1. 仅 Expert 1 有 optimizer state 和梯度，极大减少训练参数量（vs 全部 expert 参与训练）
2. β 值每步随机采样，增强泛化性
3. β > 0.5 时 expert 才表现差异化，β < 0.5 时趋近于 β=0（纯梯度更新）导致退化
4. 与"加噪声初始化"、"Dropout"、"对比损失"等其他 diversity 策略对比，Expert Evolution 在所有 benchmark 上均优

术语一般如何实现？如何使用？
- 实现于 MoE-tuning 框架的 Stage II，替换原有的 FFN 复制步骤
- 使用 PyTorch 实现：Expert 1 正常参与 backward → optimizer.step()，Expert 2-4 通过 `param.data = beta * expert1_param.data + (1-beta) * expert1_param.grad` 在每个 optimizer step 后更新
- β 从多个预定义范围随机采样，同一范围的 β 通常对应一个特定的 expert（如 Expert 2→[0.9,0.99]）
- 可扩展至任意数量 expert（论文测试 2/4 expert），每增加一个 expert 只需分配一个新的 β 范围

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---
