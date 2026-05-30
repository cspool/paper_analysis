## Multi-round Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-round Training（多轮训练）是一种针对小语言模型的训练策略，解决小模型在单轮大规模数据训练中面临的灾难性遗忘（catastrophic forgetting）问题。传统大模型训练通常对所有数据仅遍历一次（single-pass/one-epoch training），因为大模型容量大、学习新知识时不易覆盖旧知识。但小模型（≤1.5B参数）容量有限，在1.6T tokens的序列训练中，后期数据会严重覆盖早期的学习成果——论文通过"重新计算已训练数据的loss"实验验证了这一点：早期数据的loss从训练时的低值大幅反弹。Multi-round Training的核心思想是：第一轮全量训练后，基于每batch历史loss计算采样概率（困难样本被采样概率更高），第二轮有放回地采样部分数据（如50%）进行继续训练，强化对困难样本的学习。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 训练数据 D, 模型参数 θ, 训练轮数 R=2, 采样率 r=0.5

# Round 1: 全量标准训练
K = 8  # 将数据均匀分为K个part
D_parts = random_split(D, K)  # D = {P1, P2, ..., PK}
loss_history = {k: [] for k in range(K)}  # 记录每个batch的loss

for k in range(K):  # 顺序训练每个part
    for batch in P_k:
        l = compute_loss(θ, batch)
        loss_history[k].append(l)
        θ = AdamW_step(θ, l)  # 标准优化步骤

# 验证forgetting: 重新计算Round 1早期数据在最终θ上的loss
recomputed_losses = {}
for k in range(K):
    L_k = []
    for batch in P_k:
        L_k.append(compute_loss(θ_final_round1, batch))
    recomputed_losses[k] = mean(L_k)
# 观察: 早期part的recomputed loss远高于对应训练时的loss → forgetting证据

# Round 2: Loss-guided数据采样
D_round2 = []
for k in range(K):
    L = loss_history[k]  # batch-wise losses for part k
    # Softmax归一化: p_i ∝ exp(l_i)
    probs = softmax(L)  # p_i = exp(l_i) / Σ_j exp(l_j)
    # 按概率采样 r*N 个batch
    N_k = len(P_k)
    sample_count = int(r * N_k)
    sampled_indices = multinomial_sample(probs, sample_count)
    D_round2.extend([P_k[i] for i in sampled_indices])

# Round 2: 在采样数据上继续训练
shuffle(D_round2)
for batch in D_round2:
    l = compute_loss(θ, batch)
    θ = AdamW_step(θ, l)

输出: 最终模型参数 θ (Round 2)
```

关键公式：
- 采样概率: p_i = exp(l_i) / Σ_{j=1}^{N_k} exp(l_j)，其中l_i为第i个batch在Round 1中记录的loss
- 效果：Round 2 r=50% Avg=54.46 vs Single Round Avg=51.61 (+2.85)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **采样率选择**：论文实验显示r=50%可获得r=100%的~94%收益，推荐r=50%作为效率与性能的平衡点。
2. **训练轮数**：Round 2收益最大(+2.85)，Round 3开始饱和(Avg几乎不变)，推荐总共两轮。
3. **数据组织**：Round 1中数据按时间顺序分为K=8个part，每个part独立计算采样概率（保证数据多样性和loss分布的局部性）。
4. **相关技术对比**：
   - mix-cd (2025): 优先采样"collateral damage"样本（之前正确、当前错误的样本），通过复用已有inference避免额外前向计算
   - Self-Generated Replay (2026): 使用模型自身生成的历史数据伪样本作为replay数据，KL散度替代NTP损失
   - Forecasting Forgetting (2024): 使用NTK-style kernel预测哪些样本会被遗忘，仅replay预测遗忘的样本
5. **适用场景**：小模型（≤3B参数）在大规模语料上的预训练；模型容量不足以单轮充分学习时。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
