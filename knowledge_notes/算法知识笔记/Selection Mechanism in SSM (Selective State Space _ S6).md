## Selection Mechanism in SSM (Selective State Space / S6)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Selection Mechanism（选择机制）是 Mamba (Gu & Dao, 2023) 的核心创新，将 SSM 参数 (Δ, B, C) 从静态改为输入依赖，使模型沿序列维度"选择性"传播或遗忘信息。Δ_t = softplus(Parameter + s_Δ(x_t)) 控制关注当前 vs 保持历史的平衡；B_t = Linear_N(x_t) 和 C_t = Linear_N(x_t) 提供输入↔状态↔输出的细粒度内容调制。Theorem 1 证明当 N=1, A=-1, B=1 时选择机制退化为 gated RNN: g_t = σ(Linear(x_t)), h_t = (1-g_t)h_{t-1} + g_t·x_t。Ablation (Table 7): 仅 Δ 选择性 PPL 10.93→9.81，三者全开 8.71。Δ 是核心——大 Δ≈关注当前输入并重置状态，小 Δ≈忽略输入保持历史。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 选择性 SSM (S6) 前向 (单通道):
输入: x ∈ R^{B×L×D}, A ∈ R^{D×N} (diagonal, N≈16)
输出: y ∈ R^{B×L×D}

# Step 1: 计算输入依赖参数
Δ = softplus(Linear_R(x) + bias)   → (B, L, D)  # R=64, broadcast to D
B = Linear_N(x)                     → (B, L, N)  # 选择性输入投影
C = Linear_N(x)                     → (B, L, N)  # 选择性输出投影

# Step 2: ZOH 离散化 (fused in SRAM, 不物化完整张量)
Ā_t = exp(Δ_t ⊙ A)                  # element-wise, A 为 diagonal
B̄_t = Δ_t ⊙ B_t                    # 一阶 Taylor 近似

# Step 3: 选择性循环 (parallel scan, 硬件感知实现)
h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t   # time-varying! 每步参数不同

# Step 4: 输出
y_t = C_t ⊙ h_t
```
对比 LTI (S4): Ā, B̄ 对所有 t 相同 → h_t = Ā h_{t-1} + B̄ x_t → 等价于全局卷积 y = x ∗ K̄，可用 FFT。LTI 无法做内容感知推理（Selective Copying 失败），选择机制打破 LTI 约束获得内容感知但损失卷积可用性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba。实现需增加 3 组小型投影（Δ/B/C），参数增量极小（~1%）。关键工程挑战：选择性导致 time-varying → 卷积不可用 → 必须用循环，但朴素循环需物化 (B,L,D,N) 中间状态（大 N 倍）。Mamba 通过硬件感知 fused selective scan kernel 解决（见 kernel调度 层对应条目）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
