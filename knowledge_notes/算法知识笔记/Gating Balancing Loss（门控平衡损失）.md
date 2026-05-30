## Gating Balancing Loss（门控平衡损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gating Balancing Loss ($\mathcal{L}_{\text{balance}}$) 是 MOLE 提出的辅助损失函数，用于防止可学习 gating function 在训练过程中坍塌到仅激活少数 LoRA expert。问题根源：训练初期表现较好的 LoRA 会获得越来越高的 gating 概率，形成正反馈循环 → gating 熵持续下降 → 最终 68% 权重集中在单个 LoRA。该损失鼓励 gating 分布在所有 block × 所有 LoRA 上的联合分布尽可能均匀。

从算法pipeline角度拆解术语：
```
# 输入: M 个 block 的 gating 输出, N 个 LoRA
for i in 1..N:                    # 对每个 LoRA
    q_i = (1/M) * Σ_{k=1}^{M} exp(ε_i^k / τ) / Σ_j exp(ε_j^k / τ)

L_balance = -log(Π_{i=0}^{N} q_i)  # Eq.14
          = - Σ_i log(q_i)         # 等价形式
```

数学性质：当所有 q_i = 1/N 时 L_balance 最小。对数积形式对极端不平衡施以极强惩罚（某个 q_i → 0 时 log → -∞ 使 L_balance → +∞）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 总训练目标：L = L_D + α·L_balance，α=0.5（论文所有实验统一取值）。L_D：V&L 用 CLIP guidance，NLP 用 FLAN-T5 cross-entropy。
- 替代方案对比（Table 7）：仅调大 τ 虽可缓解不平衡但会丧失 gating 区分能力——MOLE^{τ1/τ2/τ3}（温度递增）性能单调下降（78.07→77.45→76.71→76.35），均低于带 L_balance 的 MOLE（78.07）。
- MOLE w/o L_balance 在 NLP NLI 任务上平均 77.57 vs MOLE 78.07（-0.50）。

涉及论文标题：
- Mixture of LoRA Experts
