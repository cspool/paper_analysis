## Difficulty-aware Data Sampling for RLVR（面向RLVR的困难感知数据采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Difficulty-aware Data Sampling 是 RLVR 训练中的一种数据筛选策略：根据当前模型对训练样本的掌握程度，优先选择模型尚未掌握的困难样本进行训练。在 TimeLens 中实现为：先用待训练模型对全部训练数据进行 offline inference，计算每个样本的 difficulty score `d_i = 1 - IoU(Ŝ_i, S*_i)`（d_i 越高表示越困难），然后以 Gaussian 分布 `g(d; μ, σ²)` 为目标进行 weighted sampling。为了确保采样后的 difficulty 分布符合目标 Gaussian 而非被原始数据分布 bias，使用 density-corrected weight：`w_i = g(d_i; μ, σ²) / p̂(d_i)`，其中 `p̂(d_i)` 是原始数据中 difficulty d_i 的经验密度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TimeLens 中 difficulty-aware sampling 的完整流程：
```
# Step 1: Offline inference — 评估每个样本的 difficulty
for each (v_i, q_i, S*_i) in D_train:  # D_train = TimeLens-100K
    Ŝ_i = π_θ(v_i, q_i)  # 用待训练模型做推理
    d_i = 1 - IoU(Ŝ_i, S*_i)  # difficulty ∈ [0, 1]

# Step 2: 估计原始数据的 difficulty 经验密度 p̂(d)
# 使用直方图或 KDE 估计 p̂(d_i) for each i

# Step 3: 计算 density-corrected sampling weight
# 目标: sample difficulty ~ Gaussian(μ=0.05, σ=0.2)
# 即 prefer d_i ≈ 0.05 (IoU ≈ 0.95 的极困难样本)
for each i:
    g_i = (1 / sqrt(2πσ²)) * exp(-(d_i - μ)² / (2σ²))
    w_i = g_i / p̂(d_i)  # density correction

# Step 4: 按权重采样 ~12K 样本
D_sampled = weighted_sample(D_train, w_i, size=12000)

# Step 5: 在采样数据上做 RLVR 训练
train_GRPO(D_sampled)
```

TimeLens 通过改变 μ 值实验显示（Fig. 7）：样本难度越高（μ 越小 → d_i 越小 → IoU 越高），模型性能越好，直到难度极高时性能趋于 plateau（IoU > 0.75）。最优配置：μ=0.05, σ=0.2，即优先选择 difficulty 接近 0.05（高度困难）的样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) Offline inference 需在 RLVR 训练前完成，因对全量训练数据做推理耗时较长，需计入总训练时间；(2) Density correction 是关键步骤：由于原始数据中困难样本天然偏少，直接按 Gaussian weight 采样可能导致实际采样到的仍是大量容易样本，density correction 确保采样后的 difficulty 分布跟随目标 Gaussian；(3) 该策略在 TimeLens 中贡献了显著的性能增益（Fig. 2b 中 "Difficulty Sampling" 为最终性能提升的关键组件之一）；(4) 不同于 curriculum learning（从易到难），difficulty-aware sampling 直接采样最困难的样本进行高效训练。类似策略也见于 GLM-4.1-V-Thinking、VL-Cogito 等工作。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
