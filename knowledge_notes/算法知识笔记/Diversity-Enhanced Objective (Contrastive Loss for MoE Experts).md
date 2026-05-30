## Diversity-Enhanced Objective (Contrastive Loss for MoE Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diversity-Enhanced Objective 是 LEGO 提出的确保 Graph MoE 中不同 expert 学习互补动力学模式的对比学习损失函数。核心思想：对于每个节点 i，同一 expert k 在不同训练样本中产生的激活表征 hᵢᵏ 应当相互靠近（正样本对），而不同 expert 产生的表征应当相互远离（负样本对）。通过此损失，各 expert 被迫专业化于不同的动力学模式（如某些 expert 擅长高能量场景、某些擅长低能量场景），从而为 LLM Judge 提供多样化的候选预测。损失函数为 InfoNCE 的变体（Eq. 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: K 个 experts 在训练集上所有节点的激活表征
输出: diversity loss ℒ_div

定义: S_i^k = {样本中节点 i 被 expert k 激活的所有表征}
      S_i   = ∪_{k=1}^K S_i^k  (所有 experts 的表征集)

for each node i and expert k:
    选取两个不同的表征 h, h̃ ∈ S_i^k (正样本对)
    sim_pos = exp(h · h̃ / τ)                    // τ: temperature
    
    计算所有表征的相似度和
    sim_all = Σ_{h' ∈ S_i} exp(h · h' / τ)      // 包含正样本和负样本
    
    ℓ_i^k = -(1/C) · log(sim_pos / sim_all)     // Eq.9: 对比损失

ℒ_div = 1/(KN) · Σ_k Σ_i ℓ_i^k                  // Eq.10: 平均所有 node/expert

// 最终损失（Eq.11）
ℒ = ℒ_mse + ℒ_div
```

损失函数的直观解释：
- sim_pos/sim_all 大 → 正样本在语义空间中靠近（同一 expert 的表征一致）→ 损失小
- sim_pos/sim_all 小 → 正样本被负样本淹没关系（不同 expert 的表征混在一起）→ 损失大
- 优化目标：不同 expert 学习到可区分的表征空间，使 LLM Judge 有真正的"选择余地"

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) 在训练过程中维护每个 expert 的激活表征集合 S_i^k（或使用 mini-batch 内的动态对比）；(b) 温度参数 τ 控制表征空间的距离敏感度（小 τ → 严格区分，大 τ → 宽松）；(c) C 为归一化常数
- 理论基础：基于 Contrastive Learning（Chuang et al. 2020, MoCo/SimCLR 等）在多 expert 场景的扩展。类似 MoELora（Luo et al. 2024）中 contrastive learning 引导 expert 专业化
- 在 LEGO 中的作用：没有 diversity loss 时，多个 experts 可能 converge 到类似的解（mode collapse），diversity loss 确保 expert specialization 是真正的多模态覆盖
- 局限：(a) 需要足够大的训练集来构建有意义的正样本对；(b) τ 的选择影响分类粒度；(c) 在极少数 expert（K=2）场景下 diversity 收益有限

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
