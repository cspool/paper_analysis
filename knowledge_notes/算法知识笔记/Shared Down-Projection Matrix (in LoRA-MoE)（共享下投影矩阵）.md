## Shared Down-Projection Matrix (in LoRA-MoE)（共享下投影矩阵）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Down-Projection Matrix 是 MoDE/LoRA-MoE-SD 的核心设计：在 multi-task LoRA-MoE 中，所有 expert 共享同一个 down-projection 矩阵 A ∈ R^{P×r}，仅 up-projection 矩阵 B^i ∈ R^{Q×r} 保持 expert-specific。设计动机来自 PCA 分析：对 15 个独立训练的 LoRA 模块进行 PCA 可视化，发现不同任务的 down-projection 向量（A 的列向量 a_j）按 rank 维度高度聚类（task-agnostic），而 up-projection 向量（B 的列向量 b_j）分散分布（task-specific）。这意味着为每个 expert 学习独立 A^i 矩阵是参数冗余的——多个任务可以用同一个 A 完成输入特征提取（down-projection），而任务特定性仅通过 B^i 表达。

从算法pipeline角度拆解术语：
```
# LoRA-MoE (传统): m 个 expert，每个有独立 A^i, B^i
# y = x@W0 + Σ_{i=1}^m R^i(x) * (x@A^i@B^{iT})
# 参数量: m × r × (P + Q)

# LoRA-MoE-SD (共享 down-projection): 共享 A, 各自 B^i
# y = x@W0 + Σ_{i=1}^m R^i(x) * (x@A@B^{iT})
# 参数量: r×P + m×r×Q ≈ r×P + m×r×Q (节省 m×r×P - r×P)

# 实际效果 (Gemma 2B, r=4, m=4):
# MoLORA 16×4: 7.62% 额外参数, ROUGE-L 57.77
# MoLORA-SD 16×4: 2.71% 额外参数, ROUGE-L 58.28
# 参数节省 64%, 性能提升 0.88% ROUGE-L
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与标准 LoRA 类似：A ∈ R^{P×r} 唯一，B^i ∈ R^{Q×r} 有 m 个副本。Training 时仅更新 A 和 B^i。
- Merge 到 backbone 时的等效权重：W_eff = W0 + Σ_i R^i(x) · (A @ B^{iT})，即 m 个不同的 ΔW^i 均通过相同的 A 生成。
- 适用场景：multi-task PEFT，任务数多时收益最大（无需为每个新任务分配独立的 down-projection 参数）。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---
