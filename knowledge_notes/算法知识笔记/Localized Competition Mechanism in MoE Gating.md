## Localized Competition Mechanism in MoE Gating

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Localized Competition Mechanism 是 GatePro 的核心操作——不在所有 expert 之间引入全局竞争，而是仅在最相似的 expert 对之间进行 targeted pairwise competition。设计原理：(1) 通过 cosine similarity 找到每个 expert i 的最相似 counterpart j*(i)；(2) 对每个 token，比较 pair 中两方的 logits；(3) 仅对 loser 施加固定惩罚 λ=10^{-4}。关键优势：仅干扰功能冗余的 expert 对（dissimilar experts 不受影响），竞争粒度是 pairwise 而非 global（每个 expert 最多一个直接对手），惩罚是 token-specific 的（同一 expert 在不同 token 上可能赢也可能输，保持路由灵活）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Vectorized localized competition
l_competitor = logits[j_star]       # gather competitor logits
loser_mask = (logits < l_competitor) # True for losers
logits_tilde = logits + loser_mask * (-1e-4)

# Example (N=8, λ=1e-4):
# logits:        [0.1,  0.05, 0.8,  0.79, 0.3,  0.2,  0.15, 0.4]
# j_star:        [2,    0,    1,    0,    7,    3,    0,    4]
# l_competitor:  [0.8,  0.1,  0.05, 0.1,  0.4,  0.79, 0.1,  0.3]
# loser_mask:    [T,    F,    F,    F,    T,    T,    F,    F]
# logits_tilde:  [0.0,  0.05, 0.8,  0.79, 0.2,  0.1,  0.15, 0.4]
# Note: expert 0 penalized (0.1→0.0), expert 4 (0.3→0.2), expert 5 (0.2→0.1)
```

设计选择：(1) λ=10^{-4} — 足够改变 top-k 排序（logit 差异通常在 10^{-3} 到 10^{-1} 量级），但对数值稳定性无影响；(2) 每对 expert 最多惩罚一个 — 避免 both penalized 导致 none selected；(3) penalty 在 logit space — 比 probability space 更稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

适用于 N≥64 的 MoE 架构（larger pool → more redundancy → 更大收益），深层 MoE 层（specialization 更难），pretrain 和 continuous training 阶段。可与任何 top-k routing（softmax/sigmoid）集成。256 experts 下 GatePro 的优势比 128 experts 更显著。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
