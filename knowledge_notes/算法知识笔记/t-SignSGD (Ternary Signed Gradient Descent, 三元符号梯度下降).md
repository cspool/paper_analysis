## t-SignSGD (Ternary Signed Gradient Descent, 三元符号梯度下降)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
t-SignSGD 是 LoTA-QAF 为优化三值适配器 {-1,0,1} 而专门设计的优化器。受 SignSGD 在离散/约束域参数更新中表现优异的启发（Balles et al. 2020; AutoRound Cheng et al. 2023），t-SignSGD 使用符号梯度 + 动态百分位阈值选择性地更新三元适配器权重。核心更新规则：A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)，其中 g_t = ∇_{A_T} L 为当前梯度，τ 为固定最小梯度阈值（如 1e-9），σ_t 为动态百分位阈值（基于梯度幅值分布，初始 top-5%，线性衰减至 0.01%）。关键设计：(1) 无学习率——选中更新的权重直接翻转（+1→0→-1 或反向），翻转方向由 sign(g_t) 决定；(2) 百分位阈值起自适应选择性作用——仅梯度幅值最大的 top-k% 权重被更新，小梯度被视作噪声而过滤；(3) σ_t 的线性衰减实现粗到细的搜索策略（早期高阈值聚焦关键参数做"大调整"，后期低阈值允许"精细调整"）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# t-SignSGD 单步更新（以 A_T 为例）
# 输入: A_T (当前三元适配器, 值 ∈ {-1,0,1})
#       g (梯度, same shape), τ (最小阈值, 1e-9)
#       σ (当前动态百分位阈值)
#       step, total_steps

# 1. 线性衰减 σ_t (前 80% 训练衰减)
if step < 0.8 * total_steps:
    σ_t = σ_init * (1 - step / (0.8 * total_steps))  # 5% → 0.1%
else:
    σ_t = 0.0001  # 后 20% 固定 0.01%

# 2. 计算百分位阈值（基于梯度幅值分布）
|g|_flat = abs(g).flatten()
σ_t_value = percentile(|g|_flat, (1 - σ_t) * 100)

# 3. 选择更新位置并更新
threshold = max(τ, σ_t_value)
update = (abs(g) > threshold)       # bool mask
A_T[update] -= sign(g[update])      # +/-1 翻转
A_T = clip(A_T, -1, 1)              # 保持三值约束
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
t-SignSGD 在 LoTA-QAF 中通过 PyTorch 自定义优化器实现（t_signSGD.py），继承 torch.optim.Optimizer。百分位阈值计算使用 torch.quantile。由于无动量机制和自适应学习率，t-SignSGD 的收敛性依赖 σ_t 的衰减调度设计：噪声过滤（低幅值梯度被阈值过滤，防止离散空间的震荡）+ 退火式搜索（粗→细的探索-利用平衡）。当前实现局限：(1) 无一二阶动量；(2) 仅线性衰减调度；(3) 未探索余弦退火或循环调度。LoTA-QAF 实验显示，在 4-bit/3-bit 量化下 t-SignSGD 收敛损失与 LoRA 差距 < 0.01；2-bit 下差距较大（0.132 vs 0.375），因三值调整在仅 4 个可能值的量化空间中更不稳定。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning
