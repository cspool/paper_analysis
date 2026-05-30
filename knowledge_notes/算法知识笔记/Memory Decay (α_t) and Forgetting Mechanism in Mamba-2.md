## Memory Decay (α_t) and Forgetting Mechanism in Mamba-2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 中的 Memory Decay（记忆衰减）由 α_t = exp(-Δ_t · exp(A)) 控制，其中 Δ_t = Softplus(W_Δ u_t + b_Δ) 是输入依赖的门控标量，A 是可学习的标量参数。α_t ∈ (0,1) 决定每个时间步保留多少历史状态信息：α_t → 1 完全保留（h_t ≈ h_{t-1}），α_t → 0 完全遗忘（h_t ≈ Δ_t·B_t·x_t）。衰减通过乘法累积：第 i 个 token 在 t 时刻的记忆强度为 α_{i:t} = ∏_{j=i+1}^{t} α_j。整个状态可写为加权和 h_t = Σ_{i=1}^{t} α_{i:t} · B̄_i · x_i，这是 Sliding Window 方法和遗忘分析的关键性质。Stuffed Mamba 论文发现：在 8K 训练长度下，某些 head 的 α_t 始终接近 1（如首 token 累积 α_{1:t} > 0.997），模型未学会在必要时遗忘——即"遗忘机制失效"。这正是长度泛化失败的根本原因。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba-2 的记忆衰减和遗忘诊断：
```
# Mamba-2 head 中的遗忘机制
Δ_t = Softplus(W_Δ @ u_t + b_Δ)    # 输入依赖的门控
α_t = exp(-Δ_t * exp(A))          # 单个时间步的衰减因子

# 累积衰减（第 i 个 token 在时间 t 的保留强度）
α_{i:t} = ∏_{j=i+1}^{t} α_j       # t-i 次乘法累积

# 状态加权和形式（关键性质）
h_t = Σ_{i=1}^{t} α_{i:t} · (Δ_i · B_i) · x_i
    = Σ_{i=1}^{t} α_{i:t} · B̄_i · x_i

# Stuffed Mamba 的诊断发现
# 问题：α_t 始终 ≈ 1 → α_{1:T_train} > 0.997 → 几乎不遗忘
# 结果：超训练长度后 memory interference 导致检索失败
# 检索误差（公式 7）：
y_t = α_{s:t} · (C_t·B̄_s) · x_s + Σ_{i≠s} α_{i:t} · (C_t·B̄_i) · x_i
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                               当 token 过多且 α_{i:t} ≈ 1 时，此项急剧增大
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba-2 中 α_t 由 Softplus 激活的 Δ_t 经 exp(-Δ_t·exp(A)) 计算。训练时模型通过梯度下降学习 W_Δ, b_Δ, A 来控制衰减行为。Stuffed Mamba 的发现：短训练长度下模型"学会"设置 α_t ≈ 1（不遗忘）因为状态容量足够大。解决方向：(1) 训练长度 > 遗忘阈值（T_forget ∝ N_S）；(2) 推理时干预：RRI 缩放 α_t' = α_t^{0.9999} 强制轻微加速衰减，B_t' = 0.75·B_t 减弱插入强度。Sliding Window 利用加权和性质直接截断：h_t^{(w)} = h_t - α_{t-w+1:t}·h_{t-w}。该机制适用于所有可写为加权和的 RNN（GLA: G_t∈(0,1)^d 门控衰减；RWKV: e^{-w} channel-wise decay；RetNet: γ 固定衰减）。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
