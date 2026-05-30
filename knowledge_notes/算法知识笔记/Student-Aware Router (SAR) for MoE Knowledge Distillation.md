## Student-Aware Router (SAR) for MoE Knowledge Distillation

术语解释
Student-Aware Router (SAR) 是 Kim et al. (2025) 提出的第二种 MoE 专用知识蒸馏方法。与 KA 的"采样增广"策略不同，SAR 直接优化 MoE 教师的路由器，使其根据 student 的反馈调整 expert 权重，从而为 student 提供更优的知识聚合。SAR 的核心创新在于将"student-friendly teacher"思想应用于 MoE 的 routing 机制。

术语是什么？
SAR 的每次迭代包含两个阶段：
1. **Router 更新阶段**：使用 student 反馈（reverse KL divergence + auxiliary load balancing loss β·L_b）仅更新 MoE 教师的路由器参数 `W_g` 和 `W_noise`，所有其他参数冻结，所有 expert 全激活
2. **知识蒸馏阶段**：使用更新后的路由器激活所有 expert 并加权聚合输出，通过 reverse KL divergence 蒸馏到 student

SAR 的 motivation：简单激活所有 expert（ALL baseline）已优于传统 KD，但不如 SAR，说明通过 student feedback 调整 expert 权重比简单全激活更有效。

从算法pipeline角度拆解术语：
```
# SAR Training Pipeline (per step)
def sar_step(x, teacher_moe, student, beta):
    N = teacher_moe.num_experts
    
    # Student generates pseudo-target (on-policy)
    y_pseudo = student.generate(x)
    
    # === Phase 1: Router Update ===
    # Teacher forward with ALL experts (full activation)
    gate_logits = teacher_moe.compute_gate_logits(x)  # H(x)
    gate_probs = softmax(gate_logits)                 # no top-k masking
    y_teacher_all = sum(gate_probs[i] * teacher_moe.E_i(x) for i in range(N))
    
    # Router loss: reverse KL + load balancing
    L_router = KL_div(student(y_pseudo|x) || y_teacher_all) + beta * L_b
    
    # Update only router parameters (W_g, W_noise)
    teacher_moe.W_g -= lr * grad(L_router, teacher_moe.W_g)
    teacher_moe.W_noise -= lr * grad(L_router, teacher_moe.W_noise)
    
    # === Phase 2: Knowledge Distillation ===
    # Teacher forward with updated router, ALL experts
    gate_logits_new = teacher_moe.compute_gate_logits(x)
    gate_probs_new = softmax(gate_logits_new)
    y_teacher_sar = sum(gate_probs_new[i] * teacher_moe.E_i(x) for i in range(N))
    
    # Student update
    L_student = KL_div(student(y_pseudo|x) || y_teacher_sar)
    student.backward(L_student)

# Auxiliary Load Balancing Loss
def L_b(m, P):
    # m: token counts per expert [N]
    # P: summed router probabilities per expert [N]
    CV = lambda v: std(v) / mean(v)
    return CV(m)^2 + CV(P)^2
```
β = 0.01 (遵循 Llama-MoE 原始设置)。

术语一般如何实现？如何使用？
- 适用于 MoE teacher (Noise Top-k Gating) → dense student 蒸馏
- Router 更新仅修改 W_g 和 W_noise，不影响 expert 参数
- 效果验证：SAR 的 KL divergence (原始 router vs 更新后 router) 随层深增加而增加→深层 expert 权重变化更显著→student 获益更大
- SAR 计算开销：每次迭代多一次 router 前向更新
- 在有 16 expert 的 Llama-MoE-3.5B 上，SAR 实现 25.91 avg ROUGE-L (vs KD 20.92)

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
