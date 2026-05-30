## Attention-guided Sparse Token Distillation (STD, 注意力引导的稀疏Token蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention-guided Sparse Token Distillation（STD）是 S²Q-VDiT 提出的 token 级量化损失重加权技术。核心观察：V-DMs 的全空间-时间注意力呈现显著稀疏模式——每层仅约 10% 的 token 拥有高注意力权重，其余 90% 对最终输出影响微弱。传统 block-wise PTQ 使用均匀加权 MSE L_quant = (1/n) Σ_j ||θ^f(x_j) - θ^q(x_j)||²，将所有 token 等权处理，浪费了有限校准数据对高影响力 token 的优化能力。STD 通过每个 Transformer Block 的多头注意力图 A ∈ R^{H×n×n} 计算每个 token j 的全局重要性：S_j = Σ_{h,i} A_{h,i,j}（token j 作为被关注对象从所有 query token 和所有 head 收到的注意力权重之和），经 min-max 归一化并映射到 [λ_min, λ_max] 得到权重 λ_j。最终损失 L_quant = (1/n) Σ_j λ_j · ||θ^f(x_j) - θ^q(x_j)||²，使高影响力 token（λ_j → λ_max=1）获得完整优化力度，低影响力 token（λ_j → λ_min）放松约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention-guided Sparse Token Distillation (STD)
# 在 block-wise PTQ 的每 block 优化中应用

# 1. 用 FP 模型前向计算当前 block 的 multi-head attention map
# A ∈ R^{H×n×n}, H = num_attention_heads
A = block.attention(Q_fp, K_fp, V_fp)  # softmax(Q @ K^T / sqrt(d))

# 2. 计算每个 token 的全局重要性得分
# S_j = sum of attention weight received by token j from ALL query tokens and ALL heads
S = zeros(n)
for h in range(H):
    for i in range(n):   # query token i
        for j in range(n):  # key/value token j
            S[j] += A[h, i, j]
# 优化: S_j = sum(A[:, :, j])  # 沿 head 和 query 维度求和

# 3. 归一化到 [λ_min, λ_max]
S_min, S_max = min(S), max(S)
λ = zeros(n)
for j in range(n):
    λ[j] = (S[j] - S_min) / (S_max - S_min) * (λ_max - λ_min) + λ_min

# 4. 重加权的量化损失
y_fp = block_fp(x)  # FP block 输出
y_q = block_q(x)    # 量化 block 输出
L_quant = (1/n) * sum(λ[j] * ||y_fp[j, :] - y_q[j, :]||² for j in range(n))
L_quant.backward()
```

超参数 λ_max=1（默认），λ_min=0.5 为最佳平衡点（控制低影响力 token 的放松程度）。Ablation 显示所有 λ_min ∈ {0.3, 0.5, 0.7} 均能提升性能，证明 STD 的鲁棒性。STD 可集成到已有 block-wise PTQ 方法：PTQ4DiT + STD 将 Aesthetic Quality 从 45.49 提升至 47.27。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Attention map 在校准前用 FP 模型一次性前向计算并预存储（per block per sample），校准时按数据索引直接检索，几乎不增加校准时间（CogVideoX-2B 仅从 2.82h 增至 2.84h）。推理时无需 attention map 或额外计算，零推理开销。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---
