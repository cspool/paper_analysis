## Optimized Hyperparameter Search (OHS, 优化超参数搜索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OHS 是 LOGART 的多级超参数搜索策略，快速确定对数 PTQ 最优量化网格。三组件：(1) ABS: tensor-wise, calibration-free, 确定 l_a；(2) SFS: block-wise, calibration-based, 搜索 s_of 抵御 outlier；(3) DBS: block-wise, calibration-based, 搜索 n₁:n₂ 权衡精度与硬件效率。联合优化：argmin_{s_of,n₁,n₂} E[||ΔW·X||_F²]。

OHS 与 LLR 协同效应（核心 insight）：三角不等式分解量化误差 ||ΔW·H^{1/2}||² ≤ (E₁(OHS)+E₂(LLR))²。E₁ 是网格固有离散化误差（OHS 通过最优 θ*={s_of,n₁,l_a} 最小化），E₂ 是理想投影与 LLR 学习结果的残差。OHS 先建优质网格，LLR 再精细化舍入。实验：OHS+LLR 500 iters (PPL 31.15, 1.25 min) > 纯 LLR 2000 iters (PPL 36.27, 4.00 min)。

从算法pipeline角度拆解术语：
```
# OHS Pipeline
# 1. ABS: compute l_a per-channel (no calib)
l_a = asymmetric_bound(w_max, w_min)  # Eq.14-16
# 2. SFS+DBS: joint block-wise search with 32 calib samples
for each block B:
    best_loss = inf
    for s_of in [0.6, 0.7, ..., 1.0]:
        for n1 in [0, 1, ..., 2^{N-1}-1]:
            n2 = 2^{N-1} - 1 - n1
            Ŵ = log_quant(W_B, l_a, s_of, n1, n2)
            loss = ||(W_B - Ŵ)·X_B||_F²
            if loss < best_loss: best_config = (s_of, n1)
```

术语一般如何实现？如何使用？
OHS 三组件独立可组合。消融显示每组件 additive gain：DBS alone PPL 170.64→66.63；+SFS→36.10；+ABS→34.29；+LLR→31.15 (OPT-125M 3-bit)。Calib 数据仅 32 segments × 2048 tokens (LLM) 或 2048 images (vision)。多级设计避免暴力联合搜索的大搜索空间，将 OHS 运行时控制在数秒（小模型）到数分钟（大模型）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION
