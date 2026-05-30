## MoE Scaling Laws

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Scaling Laws 是 Hunyuan-Large 提出的针对 MoE 模型的计算预算-性能关系。不同于传统 dense 模型的 C = 6ND（Kaplan et al., 2020）或 Chinchilla 的 C ≈ 6ND（Hoffmann et al., 2022），MoE 模型由于稀疏激活和长序列 attention 复杂度，计算预算公式需要修正：

$$C \approx 9.59ND + 2.3 \times 10^8 D$$

其中 N 为激活参数量（非总参数量），D 为训练 token 数。常数项 2.3×10^8 D 来自 attention 计算的开销。

考虑 batch size 影响，通过临界 batch size B_crit(L) 修正为最小计算预算：

$$C_{min} = \frac{C}{1 + \frac{B}{B_{crit}(L)}}$$

通过拟合 isoFLOPs 曲线，得到：
- N_opt = 5.9×10^{-3} × C_min^{0.5305}（最优激活参数量-计算预算关系）
- D_opt = 3.2 × C_min^{0.50}（最优训练数据量-计算预算关系）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 的 MoE Scaling Law 探索流程：

```
# 1. 训练系列 MoE 模型（10M-1B 激活参数）
for N_active in [10M, 50M, 100M, 300M, 1B]:
    for D in [10B, 30B, 50B, 100B]:
        C = 9.59 * N_active * D + 2.3e8 * D     # 计算预算
        C_min = C / (1 + B/B_crit(L))            # 修正 batch size
        train_model(N_active, D)                  # 记录 loss L(N, D)

# 2. 拟合 isoFLOPs 曲线
# N_opt = N_c * C_min^α  →  N_c = 5.9e-3, α = 0.5305
# D_opt = D_c * C_min^β  →  D_c = 3.2, β = 0.50

# 3. 确定最优配置
# N_opt ≈ 58.1B activated → 选择 52B (smooth curve trade-off)
# D_opt ≈ 5.6T tokens → 选择 7T (maximize within optimal range)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE Scaling Laws 的实现需要：(1) 训练多组不同规模的 MoE 模型（小模型代理），(2) 记录每个配置的 loss 和 FLOPs，(3) 拟合参数化公式 N_opt = N_c × C_min^α 和 D_opt = D_c × C_min^β，(4) 根据计算预算选择最优的激活参数量和训练数据量。Hunyuan-Large 受 Llama 3 启发，利用抛物线在最优值附近的平滑特性，从理论最优值 58.1B 调整到 52B（工程可行范围）。此方法适用于任何 MoE 模型的预训练规模规划。

---

**Joint MoE Scaling Laws (Ludziejewski et al., 2025)** 提出了统一的 Dense+MoE scaling law，将 expert 数 E 纳入 Chinchilla 形式：

$$\mathcal{L}(N_{\text{act}}, D, \hat{E}) = a\hat{E}^{\delta}N_{\text{act}}^{\alpha + \gamma \ln(\hat{E})} + b\hat{E}^{\omega}D^{\beta + \zeta \ln(\hat{E})} + c$$

其中 Ê 是 E 的单调变换（Eq.4: 1/Ê = 1/(E-1+(1/E_start-1/E_max)^(-1)) + 1/E_max），α,β,γ,ζ,δ,ω 为拟合系数。核心洞察是：exponent μ(E)=α+γ·ln(Ê) 和 ν(E)=β+ζ·ln(Ê) 中的对数项捕捉了 E 与 N_act 和 D 的交叉效应——γ>0 意味着更多 expert 会使 N_act 的 exponent 更负（更大模型时 MoE 收益递减），ζ<0 意味着更多 expert 会使 D 的 exponent 更负（需要更多 data）。

该 scaling law 的关键推导和发现：
1. **Compute optimality with E**：给定 budget F=6·N_act·D，compute-optimal 配置为 N_act_opt = G·(F/6)^(ν/(μ+ν))，D_opt = G^(-1)·(F/6)^(μ/(μ+ν))。E 增加 → 应减少 N_act、增加 D（Table 1: E=1→16 时 N_act 降 52%, D 增 113%）。
2. **Memory optimality**：引入 total params 约束 N_total ≤ M（含 KV-cache），在 3D 空间 {N_act, D, E} 求 argmin L。发现 E≤8 的 MoE 用 E× tokens 训练可超越 compute-optimal dense——Rule of Thumb。
3. **Inference optimality**：将 inference cost 2·N_act·D_inf 纳入 joint budget，揭示 MoE 的 inference 优势（每 token FLOPs = dense 的 39-64%）。
4. **LR scaling for MoE**：LR = exp(8.39 - 0.81·ln(N_act\e) - 0.25·ln(E))，更多 expert → 更低的 optimal LR（E 系数为负）。

拟合方法：LBFGS 优化 Huber loss (δ=0.01) on log-space，280+ 模型 runs（N_act 最高 2.7B, N_total 最高 5B, E∈{1,2,4,8,16,32}），RMSE_v=0.0039。该 scaling law 的特点是将 dense (E=1) 和 MoE 统一在同一框架下，使得跨 E 的公平比较成为可能。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
