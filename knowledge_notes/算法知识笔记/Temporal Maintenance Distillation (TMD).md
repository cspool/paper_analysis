## Temporal Maintenance Distillation (TMD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Maintenance Distillation (TMD) 是 Q-VDiT (ICML 2025) 提出的用于视频 DiT 量化优化的知识蒸馏损失函数。它解决标准 MSE（L_task = ||S^{FP}−S^{Q}||²）在视频生成量化中的核心缺陷：MSE 对每帧独立计算，忽略了视频帧间的时空相关性。TMD 通过构建 FP 模型（教师）中帧间相似度分布 D^{FP}_i = softmax([cos_sim(S^{FP}_i, S^{FP}_1),...,cos_sim(S^{FP}_i, S^{FP}_t)]) 作为先验知识，用 KL 散度对齐量化模型（学生）的帧间分布 D^{Q}_i，使每个 frame 的优化受所有帧共同引导。总损失 L_total = L_task + γ·L_temporal（γ=100）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TMD 在 PTQ 校准训练中的计算流程：
```
# 输入: S^{FP}, S^{Q} ∈ R^{n×d}  (n=s×t 个token)
# 步骤1: 提取每帧的token序列
for frame i in [0..t-1]:
    S_fp_i = S^{FP}[i*s : (i+1)*s, :]   # ∈ R^{s×d}
    S_q_i  = S^{Q}[i*s : (i+1)*s, :]

# 步骤2: 计算帧间余弦相似度矩阵 (Eq. 13)
for i, j in [0..t-1]:
    T_fp[i,j] = cos_sim(S_fp_i, S_fp_j) = (S_fp_i · S_fp_j) / (||S_fp_i||·||S_fp_j||)
    T_q[i,j]  = cos_sim(S_q_i, S_q_j)

# 步骤3: 构建每帧的时序分布 (Eq. 14)
for frame i in [0..t-1]:
    D_fp_i = softmax([T_fp[i,0], ..., T_fp[i,t-1]])   # ∈ R^t
    D_q_i  = softmax([T_q[i,0], ..., T_q[i,t-1]])

# 步骤4: KL散度对齐 (Eq. 15)
L_temporal = Σ_{i=1}^{t} KL(D_fp_i || D_q_i)
           = Σ_i Σ_k D_fp_i[k] * log(D_fp_i[k] / D_q_i[k])

# 步骤5: 梯度分析 (Eq. 16-18)
# ∂L_temporal/∂S^{Q}_i = 双向梯度流:
#   Σ_j [∂L_temporal/∂T^{Q}_{i,j} · ∂T^{Q}_{i,j}/∂S^{Q}_i + ∂L_temporal/∂T^{Q}_{j,i} · ∂T^{Q}_{j,i}/∂S^{Q}_i]
# 其中 ∂L_temporal/∂T^{Q}_{i,j} = D^{Q}_{i,j} - D^{FP}_{i,j} (Eq. 17 的简化)
# 因此任意帧对的相关性受所有帧共同数值影响
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TMD 作为额外的损失项叠加到 PTQ 校准损失上，γ=100 是论文通过实验确定的平衡值（对 γ∈{0.1,1,10,100,500,1000} 均有效，不敏感）。TMD 适用于任何视频生成模型的 PTQ/微调场景，因为：(1) 无额外推理开销——TMD 仅在训练时计算，推理时不需计算帧间分布；(2) 普适性强——不依赖模型架构细节，仅需视频帧的 latent token 序列。在 W3A6 设置下，TMD 将 Scene Consistency 从 22.00 (仅 TQE) 提升到 22.58 (TQE+TMD)，VQA-Technical 从 29.58 (SOTA) 提升到 59.10。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---
