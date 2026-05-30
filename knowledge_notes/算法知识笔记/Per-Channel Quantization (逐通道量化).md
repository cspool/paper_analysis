## Per-Channel Quantization (逐通道量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Channel Quantization 是将张量沿 channel（特征/隐藏维度）方向分组量化的策略。对于 KV Cache $X \in \mathbb{R}^{l \times d}$，per-channel 量化意味着沿 d 维度分组，每若干 channel 共享一组量化参数（scale, zero-point）。每个 channel 的量化参数独立计算，使得误差被限制在各自的 channel 内。

KIVI 论文的核心发现：key cache 中少量固定 channel 存在极大的 magnitude outlier，使用 per-token 量化时这些 outlier 会污染同一 group 内的所有正常 channel，导致 attention score 相对误差高达 47%。而 per-channel 量化将 outlier channel 的量化误差隔离在自身 channel 内，不干扰其他 channel，attention score 相对误差降至 9.6%（约 5× 更低）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 中 per-channel key cache 量化（KeyQuant 函数）：

```
procedure KeyQuant(X_K ∈ R^{l × d}):
    r = l % R                    # 不能被R整除的余数
    X_K_g = X_K[:l - r]          # grouped部分
    X_K_r = X_K[l - r:]          # residual部分 (FP16)
    Q(X_K_g) = GroupQuant(X_K_g, dim=channel, numGroup=l // G)
    # dim=channel: 沿channel维度分组，每G个连续token为一组
    # 每组内共享 scale 和 zero-point
    return Q(X_K_g), X_K_r
```

与 per-token 对比：
- per-token: X[l × d] → 沿 token 维分 d/G 组，每 G 个 channel 共享 scale
- per-channel: X[l × d] → 沿 channel 维分 l/G 组，每 G 个 token 共享 scale

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 使用 Triton 实现 per-channel group-wise quantization kernel。实现时每组 G=32 个 token，计算组内 min/max → scale=(max-min)/(2^B-1), zero-point=min → round((x-z)/s) → clamp。由于 per-channel 量化跨 token 维度，新到达 token 无法直接 append，KIVI 通过 grouped+residual split 解决流式兼容问题。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
