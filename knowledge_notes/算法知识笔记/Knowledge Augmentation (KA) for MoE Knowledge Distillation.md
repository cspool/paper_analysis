## Knowledge Augmentation (KA) for MoE Knowledge Distillation

术语解释
Knowledge Augmentation (KA) 是 Kim et al. (2025) 提出的 MoE 专用知识蒸馏方法。核心思想：在蒸馏过程中，对同一输入进行 M 次教师前向传播，每次以概率 λ 从 gate probability 分布中随机采样 N-1 个 expert（以 1-λ 概率取 Top N-1），从而增广来自不同 expert 组合的多样化知识。解决传统 KD 仅使用 Top-k activated experts 而遗漏 non-activated experts 中知识的问题。

术语是什么？
KA 基于以下观察设计：(1) MoE 教师中 non-activated experts 的 gate probabilities 总和超过 50%，即大部分 expert 知识未被传统 KD 利用；(2) 增加 activated experts 数量（k→N-1）可提升 student 性能但不一定提升 teacher 性能，说明 non-activated experts 有独特知识；(3) Load balancing 使同一输入在不同迭代可能激活不同 expert 集合，知识分散在多个 expert 中。

KA 机制：使用 N-1 个 expert（而非原始 Top-k），通过混合策略（以概率 λ 采样、概率 1-λ 取 Top N-1）平衡 knowledge diversity 和 consistency。

从算法pipeline角度拆解术语：
```
# KA Distillation Pipeline (per training step)
def ka_distillation_step(x, teacher_moe, student, M, lambda_, N):
    # x: input request
    # teacher_moe: MoE teacher with Noise Top-k Gating
    # M: number of augmented forward passes
    # lambda_: sampling probability (typ. 0.05)
    
    # Student generates pseudo-target (on-policy)
    y_pseudo = student.generate(x)
    
    # Step 1: KA-based teacher forward (M times)
    teacher_logits_list = []
    for m in range(M):
        # Choose expert selection strategy
        if random() < lambda_:
            # Random sampling from gate prob distribution
            gate_logits = teacher_moe.compute_gate_logits(x)  # H(x)
            gate_probs = softmax(gate_logits)
            E_selected = sample_without_replacement(gate_probs, N-1)
        else:
            # Top N-1 selection (deterministic)
            gate_logits = teacher_moe.compute_gate_logits(x)
            E_selected = topk_indices(gate_logits, N-1)
        
        # Forward with selected experts
        KA_logits = compute_KA_logits(gate_logits, E_selected)
        G_KA = softmax(KA_logits)
        y_teacher = sum(G_KA[i] * E_i(x) for i in E_selected)
        teacher_logits_list.append(y_teacher)
    
    # Step 2: Distillation loss (reverse KL, M times)
    for teacher_logits in teacher_logits_list:
        L = KL_div(student(y_pseudo|x) || teacher_logits)  # reverse KL
        student.backward(L)
    
    return student
```
超参数：M（增广次数，典型值 2）、λ（采样概率，典型值 0.05）。M 过大会导致过度多样化的知识（"nonsense knowledge"），反而降低性能。

术语一般如何实现？如何使用？
- 适用于 MoE teacher → dense student 的蒸馏场景
- 需要 teacher 支持灵活修改 activated expert 数量（从 k 到 N-1）
- 与 GKD (Agarwal et al., 2024) 框架兼容：使用 student on-policy 生成 + reverse KL divergence
- 计算开销：M 次额外教师前向（M=2 时约 2× teacher forward cost per step）
- KA 在 Llama-MoE (3.5B/3.0B) → Sheared-Llama (1.3B) 蒸馏上实现 +4.8 ROUGE-L over conventional KD

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
