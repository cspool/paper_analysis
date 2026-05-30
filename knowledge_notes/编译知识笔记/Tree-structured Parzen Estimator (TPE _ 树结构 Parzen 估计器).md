## Tree-structured Parzen Estimator (TPE / 树结构 Parzen 估计器)

术语解释
一种用于超参数优化的贝叶斯优化算法，是 Optuna 框架的默认采样器。TPE 通过分别建模"好"试验和"差"试验的参数分布，用概率密度比 ℓ(x)/g(x) 作为采集函数指导搜索，具有 O(n) 计算复杂度和天然支持类别变量的优势。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TPE 是 Bergstra et al. (2011) 提出的贝叶斯优化方法，由 Watanabe (2023) 进一步细化和分析。其核心机制与 Gaussian Process-based Bayesian Optimization 不同：

1. **分布建模**：不直接建模 P(y|x)（目标函数），而是建模 P(x|y)（给定性能下的参数分布）：
   - ℓ(x) = P(x|y < y*) — "好"试验（top γ 分位，通常 γ≈0.25）的参数分布
   - g(x) = P(x|y ≥ y*) — "差"试验（其余）的参数分布
2. **采集函数**：Expected Improvement (EI) 简化为 EI(x) ∝ ℓ(x)/g(x)，选择 ℓ(x)/g(x) 最大的点作为下一候选
3. **核密度估计**：ℓ(x) 和 g(x) 通过一维 Parzen 窗（核密度估计）建模，而非多维高斯过程

相比 GP 的优势：
- 时间复杂度 O(n) vs GP 的 O(n³)
- 天然支持类别变量（每类独立建模）和条件参数（tree-structured）
- 高维空间更稳定
- 支持并行建议（constant liar 等策略）

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RoE 中，TPE 用于搜索每层 MoE 的 Gumbel 温度 τ_l。虽然不是传统的编译优化，但 TPE 在此处扮演了**自动调优（auto-tuning）框架**的角色——类似 TVM AutoScheduler 搜索最优 kernel schedule，TPE 搜索最优推理超参数配置。

TPE 在 RoE 温度搜索中的运转流程：

```
# TPE Auto-tuning Pipeline for RoE Temperature Search
输入: MoE 模型 M (L 层, 其中 L_moe 层为 MoE 层)
      验证集 D_val
      任务类型 task_type ∈ {math, commonsense, code}
输出: 最优温度配置 {τ_l}

# ===== 搜索空间定义 =====
search_space = {}
skip_layers_k = get_skip_layers_k(task_type)  # e.g., 1 for math, 3 for commonsense
for l in range(L_moe):
    if l < skip_layers_k or l >= L_moe - skip_layers_k:
        τ[l] = 0.0           # 首尾层固定为确定性路由（启发式剪枝）
    else:
        τ[l] = suggest_float(f"tau_{l}", 0.0, 0.5)  # 仅搜索中间层

# ===== TPE 优化循环 =====
study = optuna.create_study(
    sampler=TPESampler(
        n_startup_trials=10,          # 初始随机探索
        n_ei_candidates=24,           # 每步 EI 候选数
        gamma=lambda _: 25            # top 25% 为"好"组
    ),
    direction="minimize" if use_ppl else "maximize"
)

for trial in range(n_trials):  # e.g., 50 trials
    # Step 1: TPE 建议下一候选点
    params = study.ask()  # TPE 内部: 用 ℓ(x)/g(x) 最大化选择 τ

    # Step 2: 评估候选
    τ_config = {l: params[f"tau_{l}"] for l in searched_layers}
    metric = evaluate_roe(M, D_val, τ_config)  # PPL 或 Accuracy

    # Step 3: 反馈结果，更新 ℓ(x) 和 g(x)
    study.tell(params, metric)

# Step 4: 返回最优配置
best_τ = study.best_params
return best_τ
```

TPE 在此场景的关键作用：
- 搜索空间为 L_moe 维连续空间（每层一个 τ ∈ [0,0.5]），L_moe 可达 26+（OLMoE），grid search 不可行
- PPL 评估成本低（单次 forward），但 Accuracy 评估成本高（需完整生成），TPE 在 50 trials 内收敛
- 启发式剪枝（首尾层 τ=0）将有效搜索维度降低约 30-40%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用 TPE 的典型方式（Optuna）：
```python
import optuna

# TPE 是 Optuna 默认 sampler
study = optuna.create_study(direction="minimize")  # 内部使用 TPESampler

# 或显式配置
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        consider_prior=True,          # 使用先验分布
        prior_weight=1.0,             # 先验权重
        consider_magic_clip=True,     # 魔数裁剪改善数值稳定性
        consider_endpoints=False,     # 不强制考虑边界
        n_startup_trials=10,          # 随机启动试验数
        n_ei_candidates=24,           # EI 优化候选数
        multivariate=True,            # 多维联合建模
        warn_independent_sampling=True
    )
)
study.optimize(objective, n_trials=100)
```

在 MoE/LLM 推理领域的其他应用场景：
- Expert offloading 阈值搜索
- Speculative decoding draft token count 搜索
- KV-cache eviction 策略参数搜索
- 量化位宽分配优化

论文使用 TPE 而非 Grid Search 或 Random Search 的原因：
- 50 trials 预算有限（尤其 Accuracy 评估昂贵）
- 搜索空间连续且维度较高（每层一个温度）
- TPE 的 O(n) 复杂度适合 50-300 trials 的中等规模

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
