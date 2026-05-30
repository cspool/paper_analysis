## DS-MoE (Dense Training, Sparse Inference for MoE)

术语解释
DS-MoE 是 Pan et al. (2024) 提出的 MoE 训练范式创新——训练阶段所有 expert 全激活（dense training），推理阶段仅激活 top-K 或超阈值 expert（sparse inference），搭配 Mutual Information (MI) Loss 实现负载均衡和 sparsity 塑造。核心发现：传统 sparse training 中仅 top-K expert 接收梯度导致 Router 参数更新不完整，是 MoE 参数效率低的根本原因。

术语是什么？
DS-MoE 重新定义了 MoE 的训练-推理关系：
- **Dense Training**：前向时计算所有 N 个 expert 的输出 O = Σ_{i=1..N} S_i · E_i(X)；反向时 Router 梯度包含所有 expert 贡献 ∇S = [E_1(X), ..., E_N(X)]^T ∇O，每个 expert 梯度为 ∇e_i(X) = S_i ∇O。训练框架：PyTorch + FSDP + activation checkpointing。
- **Sparse Inference**：Router 计算 scores S 后，仅激活 top-K 或超阈值 expert：O = Σ_{i∈A} S_i · E_i(X) where A = {i | topK(S, K) or p_i > ε}。使用 SimpleMoE 的 ParallelLinear 操作执行稀疏 expert computation。
- **与 Sparse Upcycling 的区别**：Sparse Upcycling 从 dense checkpoint 初始化后转为 sparse training（train dense → train sparse）；DS-MoE 始终保持 dense training（train dense → deploy sparse）。DS-MoE 的参数效率来自训练阶段的完整梯度信号，而非已有 dense checkpoint 的"遗产"。
- **与 DefaultMoE 的区别**：DefaultMoE 使用 EMA default vector 近似 dense gradient 同时保持前向 sparse；DS-MoE 在前向直接执行 dense computation（所有 expert 全激活）。DS-MoE 更精确（无 EMA 近似误差），但训练开销更大。

从算法pipeline角度拆解术语：
```
# DS-MoE Training Pipeline (per layer, per token)
# 1. Router
S = Softmax(h(X))              # [N], Router 计算所有 expert scores

# 2. Dense Forward (ALL experts)
O = zeros(d_h)
for i in 1..N:
    # Expert FFN: GeLU(X @ W_up_i) @ W_down_i
    E_i = GeLU(X @ W_up_i + b_up_i) @ W_down_i + b_down_i
    O += S[i] * E_i            # weighted sum of ALL experts

# 3. Dense Backward
# Router gradient (dense, no mask)
dL/dS = [E_1, ..., E_N]^T @ dL/dO   # all N experts contribute
# Expert gradient
for i in 1..N:
    dL/dW_up_i, dL/dW_down_i = backprop through E_i, scaled by S[i]

# 4. MI Loss (per batch)
P = mean(S, dim=0)                     # [N], expert probability per batch
H_e = -sum(P * log(P))                 # expert entropy (maximize for balance)
H_cond = mean(-sum(S * log(S), dim=-1)) # per-token conditional entropy (minimize for concentration)
L_MI = -H_e + H_cond                   # MI Loss
L_total = L_LM + alpha * L_MI

# 5. Sparse Inference (post-training)
S = Softmax(h(X))
# Option A: Fixed TopK
A = topK(S, K)                # K = 4 or 6 depending on model/sparsity
# Option B: Threshold
p_norm = S * N                # normalized probability
A = where(p_norm > epsilon)   # epsilon = 0.48 (default)
# Sparse compute
O = sum_{i in A} S[i] * E_i(X)  # ParallelLinear (SimpleMoE)
```

术语一般如何实现？如何使用？
- **训练配置**：DS-MoE-1B (1067M params, N_ffd=32, D_ffd=256, N_att=16), DS-MoE-3B (2846M, N_ffd=32, D_ffd=384, N_att=8), DS-MoE-6B (6343M, N_ffd=32, D_ffd=512, N_att=8)
- **训练成本**：H100×8 (1B, 24h), H100×32 (3B, 64h / 6B, 124h)。训练数据 30B tokens (1B) / 100B tokens (3B/6B)
- **推理 sparsity**：DS-MoE-1B 激活 41% hidden, DS-MoE-3B 激活 34%, DS-MoE-6B 激活 29%（趋势：更大模型 → 更高 sparsity）
- **推理性能**：DS-MoE-6B 在 vLLM 上达到 2.00 req/s (A100), 2.30 req/s (H100)，比 Mistral-7B 快 1.86×，比 DeepSeekMoE-16B 快 1.50×，GPU 内存仅 12.6 GiB
- **代码未开源**（截至 2024.4），使用 SimpleMoE (开源) 进行稀疏推理

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

---
