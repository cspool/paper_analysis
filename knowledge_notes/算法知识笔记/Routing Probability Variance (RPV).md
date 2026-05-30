## Routing Probability Variance (RPV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Routing Probability Variance (RPV) 是 LTDR 论文提出的衡量每个 token 的 routing distribution 离散度的指标。定义：对于 token x 的 routing probabilities `P(x) ∈ R^K`（softmax 归一化后的 K 维向量），`RPV(x) = Variance(P(x)) = (1/K) * Σ_i (P(x)_i - μ)²`，其中 μ = (1/K) * Σ_i P(x)_i。RPV 反映 Router 对 token 的"路由置信度"：低 RPV 表示 token 被均匀分配给各 expert（router 不确定该由谁处理），高 RPV 表示 token 集中分配给少数 expert（router 有明确偏好）。

LTDR 利用 RPV 实现了两个功能：(1) 通过 RPV distribution 分析 vision token 的 long-tailed 特性；(2) 用 Mean(RPV) 作为动态阈值区分 vision head tokens（低 RPV）和 tail tokens（高 RPV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# RPV 计算 (per vision token)
def compute_RPV(probs_v):  # probs_v: [M, K]
    mu = mean(probs_v, dim=1)        # [M], mean routing prob
    var = mean((probs_v - mu)^2, dim=1)  # [M], variance per token
    return var  # [M] = RPV for each vision token

# Tail/Head Classification via Mean RPV
RPV_v = compute_RPV(softmax(V @ W_g))  # [M]
threshold = Mean(RPV_v)                 # scalar, dynamic
is_tail = RPV_v > threshold             # ~13% of vision tokens
is_head = RPV_v <= threshold            # ~87% of vision tokens

# RPV-L2 Norm Analysis (验证 RPV 与信息量的关联)
# Top-13% RPV tokens:  mean L2 norm = 0.3158
# Top-13%-26% tokens:   mean L2 norm = 0.2124  
# Top-26%-39% tokens:   mean L2 norm = 0.1475
# → Higher RPV correlates with richer vector representations
```

**RPV 分析结果**：
- Language tokens: RPV 接近均匀分布（load balancing 适配）
- Vision tokens without load balancing: long-tailed RPV distribution
- Vision tokens with load balancing: 高 RPV token 数量被抑制（biased long-tailed）
- LTDR 显著提升 vision tail tokens 的 mean RPV，head tokens 不受影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：在每次前向传播中计算 softmax(logits) 后直接求 per-token variance，计算开销极小（O(M*K) per MoE layer）
- **用途 1 — Tail token 识别**：用 Mean(RPV) 作为动态阈值，避免固定比例阈值（fixed 10%/15%/20%）对不同数据分布不鲁棒。实验表明 adaptive mean-RPV 优于所有 fixed-ratio 方案
- **用途 2 — 分布分析**：通过 RPV 分布可视化验证 TER 策略效果（training steps evolution、cross-router comparison）
- **替代方案对比**：VsDEA 中用 Instruction-Aware Tokens (IATs, attention-based selection) 替代 RPV-based selection → 效果较差（跨模态 attention 噪声干扰）

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
