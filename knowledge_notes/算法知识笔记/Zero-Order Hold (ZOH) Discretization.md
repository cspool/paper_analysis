## Zero-Order Hold (ZOH) Discretization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zero-Order Hold (ZOH) 是将连续时间 SSM 转换为离散时间 SSM 的离散化方法。连续系统 h'(t) = Ah(t) + Bx(t) 通过 ZOH 离散化后得到 h_t = A_bar·h_{t-1} + B_bar·x_t，其中转换公式为 A_bar = exp(ΔA)，B_bar = (ΔA)^{-1}(exp(ΔA) - I)·ΔB。ZOH 假设在两个采样点之间输入的 x(t) 值保持不变（即"零阶保持"），物理上等价于在采样间隔 Δ 内保持信号恒定。在 Mamba 中，Δ 不再是固定的常数，而是由输入 x_t 通过 s_Δ(x) = Broadcast_D(Linear_1(x)) 和 τ_Δ = softplus 动态生成，使离散化步长具有输入依赖性——这是 selection mechanism 的核心数学基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ZOH 离散化在 Mamba 中的具体计算：
```
# 输入: Δ_t ∈ R^{B×L×D}, A ∈ R^{D×N}, B_t ∈ R^{B×L×N}

# Step 1: 计算离散化步长（selection mechanism 的关键）
Δ_t = softplus(Linear_1(x_t) + bias_Δ)  # 输入依赖的步长

# Step 2: ZOH 离散化公式
A_bar_t = exp(Δ_t ⊙ A)        # ⊙ 广播 element-wise 乘
# A ∈ R^{D×N} 是对角矩阵，A_bar_t ∈ R^{B×L×D×N}

# B_bar 的计算:
B_bar_t = (ΔA)^{-1}(exp(ΔA) - I)·ΔB
# 由于 A 是对角的，此运算可逐元素简化
# 等价于: B_bar_t = Δ_t ⊗ B_t  （在 A→0 或简单初始化下）

# Step 3: 离散递归
h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊗ x_t
```

与 RNN gating 的联系（Theorem 1）：当 N=1, A=-1, B=1, s_Δ=Linear_1, τ_Δ=softplus 时，ZOH 离散化使 selective SSM 退化为经典 gated RNN：
```
Δ_t = softplus(Linear_1(x_t))
A_bar_t = exp(-Δ_t) = σ(-Linear_1(x_t)) = 1 - σ(Linear_1(x_t))
B_bar_t = 1 - A_bar_t = σ(Linear_1(x_t))
→ g_t = σ(Linear(x_t))
→ h_t = (1-g_t)·h_{t-1} + g_t·x_t
```
这证明了 selective SSM 中的 discretization 是 RNN gating 机制的原则性数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZOH 是最常用的离散化规则，替代方案包括双线性变换（bilinear/Tustin）和欧拉方法。在 Mamba 的 fused kernel 中，离散化和递归/scan 在 GPU SRAM 中融合完成，不将 A_bar_t 和 B_bar_t 写入 HBM，减少 IO 传输 O(N) 倍。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---
