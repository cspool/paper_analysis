## CAPO (Cost-Aware Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CAPO（Cost-Aware Policy Optimization，成本感知策略优化）是 ResAdapt 提出的 RL 奖励塑形方法，解决 naive accuracy-cost Lagrangian penalty 导致 RL 策略向最小预算崩溃的问题。核心矛盾：若直接使用 R = Q(x,y) − λ·C(s)，任何成本降低都获得等量奖励（无论是否破坏答案），策略会无条件坍缩至 smin。CAPO 通过三项机制稳定训练：(1) Dynamic Cost Pivot τ_dyn = κ_mix·c̄_group + (1−κ_mix)·τ_fix，在组内均值和全局目标间插值；(2) Asymmetric Reward Shaping — 正确且低成本 → 中等奖励 λ_+，错误且高成本 → 强惩罚 λ_−（λ_− > λ_+ > 0）；(3) 对正确 rollout 施加正下限 ε_+，确保正确低成本 rollout 始终获得正向学习信号。消融实验证实：移除 CAPO（仅用 direct cost penalty）→ 策略坍缩至 smin；移除 cost 完全（仅 accuracy）→ 策略饱和至 smax。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CAPO 的核心计算流程（per prompt x with M=16 allocations, N=1 rollout each）：
```
# Step 1: 计算 proxy cost（用线性 proxy 避免二次方差）
c_m = (s̄_m - s_min) / (s_max - s_min)   # s̄_m = mean_t(s_m[t])

# Step 2: 动态 cost pivot（同时利用组内对比和全局锚点）
c̄_group = mean(c_1, ..., c_M)
τ_dyn = κ_mix * c̄_group + (1 - κ_mix) * τ_fix   # κ_mix=0.5, τ_fix=0.1

# Step 3: 非对称塑形（核心防止崩溃的机制）
for each rollout (m, n):
    if u_{m,n} == 1:   # 正确
        S_{m,n} = λ_+ * σ((τ_dyn - c_m) / τ_s)    # sigmoid 温度 τ_s=0.1
    else:               # 错误
        S_{m,n} = -λ_- * σ((c_m - τ_dyn) / τ_s)   # λ_- >> λ_+

# Step 4: 组合 base advantage
A_base_{m,n} = GRPO_normalize(R_task_{m,n})
Ã_{m,n} = A_base_{m,n} + λ_capo * S_{m,n} - γ * c_m   # γ: 残差全局成本压力

# Step 5: 正确 rollout 正下限保护
A_{m,n} = max(Ã_{m,n}, ε_+) if u_{m,n} == 1 else Ã_{m,n}
```
关键超参数：κ_mix 控制动态枢轴中组内 vs 全局的比例；λ_+ / λ_- 比值决定非对称程度（λ_- > λ_+ 是关键，否则策略向低成本崩溃）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAPO 作为 GRPO advantage 计算的替换层实现，在标准 GRPO 训练循环中替换 group-normalized advantage 的计算步骤。Correctness indicator u_{m,n} 定义：exact-match QA 直接使用二元结果；连续指标（ROUGE-L, temporal IoU）使用 0.35 阈值。与 GRPO 的集成：CAPO 计算 per-allocation aggregated advantage A_m_CAPO = mean_n(A_{m,n})，用于 Allocator 的 PPO update；同时各 rollout advantage A_{m,n} 用于 backbone 的 token-level PPO update（可选）。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning
