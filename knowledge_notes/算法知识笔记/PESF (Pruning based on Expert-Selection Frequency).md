## PESF (Pruning based on Expert-Selection Frequency)

术语解释
PESF 是 EAC-MoE 提出的 MoE-LLM 动态专家剪枝方法，在推理时基于当前输入序列中每个 expert 被选中的频率，动态剪枝不频繁被选的 expert，直接跳过其全部计算。与 EES/ODP 等逐 token 剪枝低权重 expert 的方法不同，PESF 从 expert 粒度（而非 token 粒度）进行剪枝，可实现更显著的加速。

术语是什么？
PESF 的核心机制：
- **剪枝阈值**：对每层 MoE（N 个 expert，每 token 选 K 个），序列长度 l，剪枝阈值 $c < \frac{l \times K}{N} \times \alpha$，其中 $\alpha \in (0, 1]$ 是超参数
- **直觉**：如果某 expert 被选中的次数低于"均匀选择期望值 × α"，说明它对当前任务不重要
- **动态性**：基于当前序列实时统计（非静态先验），适应不同任务类型的 expert 偏好
- **限制**：仅在 prefill 阶段使用（需要多个 token 的统计信息），不适用于逐 token 的 generate 阶段

两个操作点：(1) α=0.3（保守）：几乎无损准确率（<0.5%），加速 1.08-1.14×；(2) α=0.7（激进）：加速 1.30-1.47×，准确率下降~1.5%。Mixtral-8x7B 对激进剪枝敏感（expert 选择更均衡，稀疏性弱），仅适合 α=0.3。

从算法pipeline角度拆解术语：
```
=== PESF 动态 Expert 剪枝 ===
输入: 输入序列 seq[l], MoE 模型, 阈值 α
超参数: N (expert数量), K (per-token选择数)

For each MoE layer:
    # Phase 1: 统计阶段（prefill）
    c = [0] * N                           # expert 选择计数
    for each token t in seq:
        logits = router_W @ h_t
        probs = Softmax(logits)
        selected = TopK(probs, K)
        for expert_id in selected:
            c[expert_id] += 1
    
    # Phase 2: 剪枝决策
    threshold = (l * K / N) * α           # 均匀期望 × α
    active_experts = []
    for i in range(N):
        if c[i] >= threshold:
            active_experts.append(i)
        # else: expert i 被跳过
    
    # Phase 3: 仅计算未剪枝的 expert
    for each token t:
        logits = router_W @ h_t
        probs = Softmax(logits)
        selected = TopK_over_active(probs, K, active_experts)  # 仅在 active set 中选
        output = Σ norm_probs[i] * ExpertFFN_i(h_t) for i in selected
```

术语一般如何实现？如何使用？
- 完全在线、无训练：仅需一次额外的遍历统计 expert 选择计数，延迟开销可忽略
- 可与 QESC 量化组合使用（EAC-MoE = QESC + PESF），在 3.03-bit 量化基础上额外获得 1.09-1.13× 加速
- 核心依据：同一任务类别内 expert 选择频率高度相似（cosine similarity >0.8），因此序列级统计能准确反映任务偏好
- 对比 EES/ODP：PESF 从 expert 角度剪枝，直接跳过整个 expert 计算（而非仅减少某 expert 的部分输入），加速比更显著
- 局限：仅适用 prefill（generate 阶段仅单个 token，无法统计频率）；Mixtral-8x7B expert 选择分布较均匀，不适合激进剪枝

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
