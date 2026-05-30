## Two-Level Fusion Mechanism

术语是什么？
Two-Level Fusion Mechanism（两级融合机制）是 M3oE 框架中用于精确控制多域多任务信息聚合的设计，分为两个层级：(1) 第一级融合：在 Domain Expert Module 和 Task Expert Module 内部，通过 β_d 控制当前域专家与其他域专家输出的加权平衡（域间融合），通过 β_t 控制当前任务专家与其他任务专家输出的加权平衡（任务间融合）；(2) 第二级融合：通过 α_d 和 α_t 控制 Shared Expert、Domain Expert 和 Task Expert 三个模块之间的贡献比例（模块间融合）：h̄_d = S(h_d) + α_d·T(h_d) + α_t·D(h_d)。两级融合权重的乘积（如 α_d·β_d）共同决定了域特定专家最终对预测的贡献比例，实现了对每一对 (d,t) 信息源的精细逐样本调控。

从算法pipeline角度拆解术语：
两级融合的权重建模：
```
// 第一级：专家内部融合（expert-level）
Domain_fused = β_d·expert_d + (1-β_d)/(D-1)·Σ_{k≠d} expert_k
Task_fused   = β_t·expert_t + (1-β_t)/(T-1)·Σ_{k≠t} expert_k

// 第二级：模块间融合（module-level）
h_bar = Shared_gated + α_d·Task_fused + α_t·Domain_fused

// 所有权重由可训练标量生成
w = Sigmoid(e_w)  // e_w ∈ {e_αd, e_αt, e_βd, e_βt} 是一维可训练张量

// 权重语义：
// β_d ∈ (0.5,1) → 当前域专家主导；β_d ∈ (0,0.5) → 其他域知识传递主导
// α_d 大 → 域模块贡献高（相比共享和任务模块）
```

术语一般如何实现？如何使用？
两级融合通过可训练标量 + Sigmoid 激活实现，微分友好，可直接通过梯度下降优化。与使用门控网络（如 fully-gated variant）相比，该方法参数量极小（仅 4 个标量参数），但通过解耦设计实现了比统一 gate 更精准的融合控制。消融实验（论文 Table 3）表明：两级融合优于直接拼接（Concat modules）和全门控融合（Fully gated modules），验证了显式解耦+可控融合优于隐式学习的结论。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---
