## Visual Information Flow-Guided Pruning Optimization（视觉信息流引导的剪枝优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Information Flow-Guided Optimization 是 VFlowOpt 提出的自动搜索最优剪枝策略超参数的方法。理论基础：LMM 可解释性研究揭示，视觉信息从 vision tokens → query text tokens → 最后位置 last token 逐层聚合，last token 是 text-visual interaction 最具代表性的信号。VFlowOpt 将剪枝策略设计建模为优化问题：max_s f(s) = CosineSim(h_f, g_s(h_f))，其中 h_f 为无剪枝时最后 token 的表示，g_s(h_f) 为应用剪枝策略 s 后最后 token 的表示。最大化 Cosine Similarity 等价于最小化视觉信息流在剪枝前后的差异——差异越小说明剪枝对 LMM 内部信息处理的扰动越小。

关键洞察：(1) 不同 LMM 有不同的 internal information flow 特征，统一的手工策略无法最优适配；(2) 该优化仅需 30 个无标签样本 + 50 次 Bayesian Optimization 迭代，约 30 分钟完成搜索；(3) 优化目标是 task-agnostic（任务无关）的，因为在优化过程中不涉及任何下游任务标签，仅依赖 LMM 内部表示。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Bayesian Optimization for Pruning Strategy
# Search space: s = (R1, R2, R3, t, α, a)
# R1, R2 ∈ [0, 1]: stage retention ratios
# t: attention calibration sensitivity
# α: entropy weight in importance score
# a: token recycling grid size
# Constraint: R̄ = (R1*L1 + R1*R2*L2 + R1*R2*R3*L3) / L

GP = GaussianProcess(kernel=Matern52)                 # Surrogate model
acquisition = ExpectedImprovement(xi=0.01)             # Acquisition function

# Initial random sampling
X0 = uniform_sample(valid_ranges, n_init=10)
for s in X0:
    R3 = (R_target*L - R1*L1 - R1*R2*L2) / (R1*R2*L3)
    y = sum([CosineSim(LLM_last_token_no_prune(d_i),
                        LLM_last_token_with_prune(d_i, s))
             for d_i in D_unlabeled])                  # D_unlabeled = 30 samples
    data.append((s, y))

# BO iterations
for iter in 1..50:
    GP.fit(data)
    s_next = argmax ExpectedImprovement(s; GP)         # 平衡 exploration/exploitation
    R3 = solve_constraint(s_next.R1, s_next.R2, R_target)
    y_next = evaluate_f(s_next)
    data.append((s_next, y_next))

s_opt = argmax y in data                              # 最优策略
```

Annotations: 目标函数 f(s) 在每次评估时对全部 30 个无标签样本计算 cosine sim 后求和。实验证实 last token 优化优于 mean pooling（MMStar 57.8 vs 56.1）、first token（57.8 vs 54.2）、top-3 tokens（57.8 vs 56.8）。数据选择独立于任务（随机样本 vs MathV360K-GEOS 训练数据效果相当），证明优化的是模型特定信息流而非任务特定特征。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：scikit-optimize 或 BoTorch 的 Gaussian Process + Expected Improvement。LMM forward pass 需捕获最后 token 表示（hooks on final layer hidden states）。使用场景：(1) 为任何 LMM 定制剪枝策略——LLaVA-OneVision-7B、LLaVA-NeXT-7B、Qwen2-VL-7B 三者的最佳超参数经 BO 搜索后各不相同；(2) 可扩展到任何有超参数配置的 token 压缩方法（非仅 VFlowOpt 自身的剪枝）。限制：需约 30 分钟 GPU 时间 per model，对需要频繁切换模型的场景开销较高。开源实现见 https://github.com/sihany077/VFlowOpt。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization
