## Bayesian Optimization with Multi-dimensional ε-Greedy Search for MoE Deployment（面向MoE部署的多维ε-Greedy贝叶斯优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bayesian Optimization (BO) 是一种用于黑盒函数全局优化的统计方法，由 surrogate function（通常为 Gaussian Process）模拟目标函数、acquisition function 决定下一采样点。该论文将 BO 扩展用于 MoE 模型在 serverless 平台上的部署优化，目标是最小化所有 MoE 层的 billed cost。

传统单维度 ε-greedy search (ε-GS) 作为 acquisition function，以概率 1-ε 选择当前最佳变量（exploitation），以概率 ε 探索新值（exploration），ε 随迭代衰减。但 MoE 部署优化需要同时调整 Q 个 key-value pair（多维变量），单一 ε 无法充分平衡不同维度的探索与利用。该论文提出 **Multi-dimensional ε-Greedy Search**：将 ε 扩展为 Q 维向量 ε ∈ R^Q，对前 μQ 个低性能 key-value pair 使用更慢的衰减速率（乘以 ρ_1/ρ_2/ρ_3 > 1），鼓励更多探索；后 (1-μ)Q 个 key-value pair 使用正常衰减，在正常范围 P 内平衡探索与利用。

从编译框架角度拆解术语：
多维 ε-GS 在 BO 框架中的运转流程（等价于自动调优框架的搜索算法）：

```
输入: MoE model, key-value table Ω_0, constants Q, μ, α, ρ, ρ_1, ρ_2, ρ_3, λ, ζ
输出: optimal key-value pairs {ẑ_q, v̂_q}

1. Initialize Q key-value pairs, ε_0 ∈ R^Q, limited range L, historical set B_0 = {}, τ = 1
2. repeat:
3.   ε_τ = ε_0 / (1 + ρτ)                    // ε衰减
4.   Ω_τ(z_{τ-1,q}) = v_{τ-1,q}              // 更新key-value table
5.   Predict expert selection via Ω_τ          // 专家预测
6.   Solve 3 MIQCP sub-problems → c_{τ,a,e}   // 三种通信方法各求解一次
7.   (â_e, x, y, β)_τ = ODS(c_{τ,a,e})        // ODS算法选最优配置
8.   for each batch j:
9.     if |r_{e,i} - R^{real}_{e,i}| > α:      // 预测vs实际差异大
10.      Record token IDs in L_τ               // 缩小调整范围
11.      if memory insufficient: ρ'=ρ_1, replicate expert
12.      elif payload exceeded:   ρ'=ρ_2, replicate expert
13.      else:                    ρ'=ρ_3, no replication
14.      ε_{τ,1:μQ} = (1 + ρ'τ)·ε_{τ,1:μQ}    // 减缓前μQ维的衰减
15.    Deploy model, measure billed cost c_τ
16.  B_τ ← B_{τ-1} ∪ ({z, v}, c_τ)
17.  Update key-value pairs via ε-GS over B_τ, L_τ (前μQ维) and P (后维)
18.  τ += 1
19. until min c_τ change < ζ over λ consecutive iterations
20. return argmin c_τ
```

收敛条件：`τ > (1+ρ)/(ρ-ρ_1) · (1 - δ/max(ε_{0,q}))`，其中 δ 为任意小正常数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：基于 Optuna 框架（https://optuna.org/），Gaussian Process 作为 surrogate function。
- 关键参数：Q=1000（每轮调整的 key-value pairs 数量），μ（前 μQ 维慢衰减的比例），ρ（衰减常数），ρ_1 < ρ_2 < ρ_3（三种情况下的衰减速率）。
- 与标准 BO 对比：该论文实验表明 multi-dim ε-GS 在 billed cost 降低上优于 single ε-greedy sampler、TPE（Tree-structured Parzen Estimator）和 random search。
- 场景适用：当优化变量是多维离散的配置参数、且不同维度对目标函数的影响不对称时，multi-dim ε-GS 通过差异化探索速率提升收敛效率。

涉及论文标题：
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing
