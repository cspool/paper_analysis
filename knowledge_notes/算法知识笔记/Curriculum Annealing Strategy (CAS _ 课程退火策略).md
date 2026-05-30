## Curriculum Annealing Strategy (CAS / 课程退火策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curriculum Annealing Strategy (CAS) 是 VisionSelector 提出的训练策略，用于桥接 soft token selection（训练时的 sigmoid soft mask）与 hard token selection（推理时的 Top-K binary mask）之间的 gap。总损失: L_total = L_CE + λ_t·L_constraint。其中 L_CE 是下游任务交叉熵损失，L_constraint = BCE(M_soft, M_hard) 衡量 soft mask 与 hard mask 的二值交叉熵（引导 M_soft 向 0/1 极化），λ_t 是动态权重系数。λ_t 从初始小值 λ_start 线性增加到最终值 λ_end：λ_t = λ_start + (λ_end − λ_start)·min(t_current/t_total, 1.0)。早期 λ_t 较小（如 0.1），模型优先学习下游任务；后期 λ_t 较大（如 2.0），强化 soft mask 向 hard mask 的逼近。这与 Gumbel-Softmax 的 τ (temperature) annealing 互补：CAS 操作损失权重空间，τ annealing 操作 softmax 平滑度空间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Curriculum Annealing Strategy (CAS) ===
# 参数: λ_start=0.1, λ_end=2.0, t_total (总训练步数)

for step t in 1..t_total:
    # 标准前向
    s = LIS(V)                    # 重要性得分
    M_soft = DiffTopK(s, k)       # soft mask ∈ (0,1)^N
    V_pruned = M_soft ⊙ V
    loss_ce = CE(LLM(V_pruned, text), targets)

    # 约束损失: 引导 M_soft 趋近 M_hard
    M_hard = standard_TopK(s, k)  # hard binary mask (无梯度)
    loss_constraint = BCE(M_soft, M_hard)

    # 动态权重
    λ_t = λ_start + (λ_end - λ_start) * min(t / t_total, 1.0)

    # 总损失
    loss_total = loss_ce + λ_t * loss_constraint
    loss_total.backward()
    optimizer.step()

# === 消融关键结论 ===
# Config 4 (λ_t = 固定 3.0, no annealing): Avg = 88.94%  (崩塌)
# Config 3 (λ_t = 0.1→3.0):                Avg = 95.37%
# Config 5 (λ_t = 0.1→2.0):                Avg = 95.75%  (更温和)
# VisionSelector (λ_t = 0.1→2.0):           Avg = 95.96%  (全局最优)
```

Annotations: 固定高 λ (Config 4) 导致模型过早被迫极化 token 得分而非学习下游任务，性能崩塌（88.94% vs 95.96%）。更温和的终点 λ_end=2.0 (vs 3.0) 进一步改善约 0.2pp。CAS 的核心是平衡"学习什么重要"和"学习二值化选择"两个有时冲突的目标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现完全在训练循环中：λ_t 随 step 线性插值，无需额外超参调度器。λ_start 和 λ_end 通过消融实验确定（λ_start ∈ {0.1}, λ_end ∈ {2.0, 3.0}）。与 Gumbel-Softmax τ annealing 可叠加使用但 VisionSelector 未采用 τ annealing——DTS 的 sigmoid 斜率固定，仅通过 CAS 调节 selection 硬度。CAS 的普适性：任何使用 soft-hard mismatch training 的场景（如可微分剪枝、可微分量化、可微分架构搜索等）均可使用类似策略。关键原则：早期让模型"理解任务"（低 λ），后期让模型"压缩精化"（高 λ）。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs
