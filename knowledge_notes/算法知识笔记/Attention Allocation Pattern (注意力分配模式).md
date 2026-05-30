## Attention Allocation Pattern (注意力分配模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Allocation Pattern（注意力分配模式）是 SnapKV 论文（Li et al., 2024）通过系统实验发现的一项关键观察：在 LLM 自回归生成过程中，每个 attention head 对 prompt 中各 token 的注意力分配表现出高度一致的规律——仅有少数 prompt token 是真正对回答生成"重要"的，且这些重要 token 的集合在生成过程开始之前就可以被识别。

SnapKV 通过两项实验验证了这一模式：(1) 将 prompt 末尾多个 window 的 queries 选出的 "重要 attention features"（高 attention weights 的 KV 位置）与生成阶段实际使用的重要 features 计算 overlap rate，发现 prompt 最后一个 window 与生成阶段的 overlap rate 最高（Fig. 2）；(2) 将生成过程分为多个 window，计算各 window 选出的重要 features 与 prompt 最后一个 window 选出的 overlap rate，发现 overlap rate 在生成全过程中保持高位（Fig. 3），说明模式稳定。

这一发现的核心含义是：**LLM 在生成之前就知道哪些 prompt tokens 对其回答至关重要**（LLMs know what you are looking for before generation），因此可以在 prefill 阶段完成 KV cache 压缩，而不需等待生成过程。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**注意力分配模式的分析流程**：

```
# 实验设置：利用 Ultrachat 数据集，筛选 response > 512, prompt > 3K
# 每个 window = 128 tokens

# === 实验1: 模式是否可在生成前识别 ===
for layer in model.layers:
    # 取 prompt 最后 20 个 window 的 queries
    for w in last_20_windows:
        Q_w = Q[layer, w_start:w_end, :]     # [128, D]
        scores = Q_w @ K_prefix^T / sqrt(D)  # [128, L_prefix]
        avg_weights = scores.mean(dim=0)      # [L_prefix] 平均注意力权重
        important_w = avg_weights > threshold  # 标记的重要 features

    # 实际生成中使用的重要 features
    for gen_step in generation:
        Q_gen = Q[layer, gen_step, :]
        scores_gen = Q_gen @ K_prefix^T / sqrt(D)
        important_gen = scores_gen > threshold

    # 计算 overlap rate
    overlap_rate = |important_w ∩ important_gen| / |important_gen|

# 结果：最后一个 window 的 overlap rate 最高 → 模式可在生成前识别

# === 实验2: 模式是否在生成中保持稳定 ===
# 取 prompt 最后一个 window 的重要 features
important_last_window = get_important(Q_last_window, K_prefix)

# 将生成过程分为 4 个 window，每 window 128 tokens
for gen_window in [1..4]:
    important_gen_w = get_important(Q_gen_window, K_prefix)
    overlap = |important_last_window ∩ important_gen_w| / |important_gen_w|
    # 结果：overlap rate 在所有 window 中保持高位 → 模式稳定
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

注意力分配模式的具体实现基于对 attention weights 的分析。关键步骤：(1) 将 prompt 划分为 prefix 和 observation window（末尾部分）；(2) 用 observation window 内的 queries 计算对所有 prefix keys 的注意力权重；(3) 沿 query 维度聚合得到每个 prefix token 的"重要性投票分数"；(4) 通过 TopK 选出得分最高的 token 位置。这一模式被 SnapKV 用于驱动 KV cache 压缩——仅保留被选中的 prefix KV pairs 和完整的 observation window KV pairs。

该模式的适用条件：(a) 要求模型具备长上下文理解能力；(b) prompt 末尾的指令/问题能有效驱动对不同 prefix 区域的差异化注意力；(c) 不同指令下注意力模式会变化（Fig. 4），因此需要动态识别而非使用静态重要位置。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
