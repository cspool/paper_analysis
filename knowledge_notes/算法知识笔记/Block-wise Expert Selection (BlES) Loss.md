## Block-wise Expert Selection (BlES) Loss

术语解释
Block-wise Expert Selection (BlES) Loss 是 CoSMoEs (Huber et al., 2025) 提出的训练阶段辅助损失函数，鼓励 MoE 模型在连续 token 上选择相同的 expert 集合，从而在端侧推理时减少 expert offloading 次数，降低延迟。

术语是什么？
BlES Loss 的核心洞察：标准 MoE 训练中，router 对每个 token 独立选择 expert，导致连续 token 间 expert 频繁切换。在端侧 offloading 场景下（GPU 仅保留 active experts），每次 expert 切换触发 CPU↔GPU 数据传输，引入显著延迟（4-20×）。BlES 通过在训练时惩罚 expert 切换，使模型学会在连续 token 保持一致的 expert 选择。

损失函数由两部分乘积构成：
1. **Hard Expert Replacement (H_norm)**：统计连续 token 间 top-k expert selection 的实际变化次数（不可微分）
2. **Soft Expert Selection Difference (L_norm)**：计算连续 token 间 softmax 路由概率的 L1 变化（可微分）

从算法pipeline角度拆解术语。
```
# Input: R ∈ R^{B×T×E} (router logits), τ: temperature, K=2

# Soft routing weights (differentiable)
W = softmax(τ * R)                                # [B, T, E]

# Hard expert selection (non-differentiable, for H computation only)
S = top_k(W, K)                                    # [B, T, K]

# Hard expert replacement count
# For each expert e, count transitions between active/inactive
H_e = Σ_{b=1}^{B} Σ_{t=1}^{T-1} |(S[b,t+1]==e) - (S[b,t]==e)|
H = Σ_{e=1}^{E} H_e                               # each switch counted 2x
H_norm = floor(H/2) / (B * K * (T-1))              # ∈ [0, 1]

# Soft expert selection difference (differentiable)
L = Σ_{b} Σ_{t=1}^{T-1} Σ_{e=1}^{E} |W[b,t+1,e] - W[b,t,e]|
L_norm = L / (B * T)

# BlES loss = product of hard and soft signals
loss_BlES = H_norm * L_norm
```

需要配合 **Sequence-Level Load Balancing**：标准 load balancing 在 model level 计算（所有层总和），可被 exploit。例如 2 experts, 2 layers：layer 0 只用 expert 0、layer 1 只用 expert 1 → model level 50:50 完美均衡 + minimal BlES（每层内无切换）。改为 sequence level 后消除此 exploit。

```
L_total = L_NLL + α * L_load_balancing(seq_level) + β * L_BlES
```

术语一般如何实现？如何使用？
- 实现：在标准 MoE 训练循环的 loss 计算中添加 BlES loss 项，需要访问每层的 routing weights 计算连续 token 间差异
- 效果：Expert Replacement Ratio 43.82% → 6.55%（6.7× reduction），生成速度 15.02 → 23.10 tok/s（1.54× speedup）
- 质量 trade-off：Phone-sized -0.43% avg，Wearable-sized -1.87% avg（小模型更易受影响）
- 与推理时优化正交：BlES 是训练时优化，可与 MoE-Infinity、EdgeMoE 等推理时预取/缓存方法叠加
- 局限：对极小模型质量影响更明显；效果与 batch size 相关；必须与 sequence-level load balancing 配合

涉及论文标题：
- CoSMoEs Compact Sparse Mixture of Experts
