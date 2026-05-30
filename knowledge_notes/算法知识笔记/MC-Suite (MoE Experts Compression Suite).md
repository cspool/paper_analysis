## MC-Suite (MoE Experts Compression Suite)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MC-Suite（MoE Experts Compression Suite）是 Jaiswal et al. (2025) 提出的首个全面的 MoE expert 重要性估计 benchmark，从四个维度（Weight/Inference/Activation/Gradient）系统化设计了 16 种 task-agnostic 准则来识别"最可丢弃"的 expert。实验验证最优准则为 **Min-EAN**（最小激活范数）和 **Min-EGE**（最小梯度熵），因为它们同时考虑了 input tokens 和 weight parameters，比仅基于 expert usage frequency 的传统准则（EUF, ECC 等）更精确。

四个维度：
- **Weight-Guided（4种）**：仅需 expert weights 本身，无需 calibration data。Expert Weight Similarity (EWS)、Router Weight Norm (RWN)、Expert Weight Stable Rank (WSR)、Expert Weight Norm (EWN)。最佳为 Max-RWN（Table 1 中 50% sparsity pp=10.70）
- **Inference-Guided（4种）**：依赖 calibration data forward pass 统计 routing 行为。Expert Usage Frequency (EUF)、Expert-Expert Collaboration (ECC)、Expert Vocabulary Coverage (EVTC)、Expert Input Token Similarity (ETS)
- **Activation-Guided（4种）**：calibration data forward pass + hooks 收集 expert 输出。Expert Activation Similarity (EAS)、Expert Activation Entropy (EAE)、Expert Activation Outliers (EAO)、Expert Activation Norm (EAN)。**Min-EAN 为最优准则**（50% sparsity pp=9.99 vs full 7.82）
- **Gradient-Guided（4种）**：forward + backward pass 收集梯度。Expert Gradient Similarity (EGS)、Expert Gradient Entropy (EGE)、Expert Gradient Outliers (EGO)、Expert Gradient Norm (EGN)。**Min-EGE 为次优准则**（50% sparsity pp=10.45）

从算法pipeline角度拆解：

```
# Min-EAN (Minimum Expert Activation Norm) — 最优准则
def min_ean_score(M, l, e, X_calib):
    A_e = []
    for batch in X_calib:
        h = M.embed(batch)  # hidden states
        topk = topk(softmax(h @ M.layers[l].W_G), k=2)
        mask = (topk == e).any(dim=-1)  # tokens routed to expert e
        if mask.any():
            x_e = h[mask]  # (t_e, d)
            # SwiGLU FFN: SiLU(gate) * up → down
            a = silu(x_e @ W_gate) * (x_e @ W_up) @ W_down
            A_e.append(a)
    A_all = concat(A_e, dim=0)  # (total_tokens_e, d)
    return sum(norm_l2(A_all, dim=0))  # ||A_e||_2, lower→more droppable

# Min-EGE (Minimum Expert Gradient Entropy) — 次优准则
def min_ege_score(M, l, e, X_calib):
    for batch in X_calib:
        loss = cross_entropy(M(batch), batch_labels)
        loss.backward()  # accumulate gradients
    grad_W = M.layers[l].experts[e].weight.grad
    # H ∝ Σ_j log[σ(W_grad^j)]
    stds = [std(grad_W[j,:]) for j in range(grad_W.shape[0])]
    return sum(log(s) for s in stds if s > 0)
```

**实验发现**：Activation entropy 和 gradient entropy 强正相关；dominant expert 具有较高 entropy（信息量大，适合 downstream adaptation）；layers 1-2 中有 2 个 expert 的 gradient entropy 极高，丢弃它们导致 abrupt 性能崩溃；activation entropy 跨层逐渐增长（initial→terminal layers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Calibration data：256 C4 validation samples, max_seq_len=2048 即可获得稳定估计
- Activation criteria 需 forward hooks（开销低），gradient criteria 需 backward pass（开销较高）
- 推荐 Min-EAN：最优性能 + 最低开销（仅 forward pass）
- 准则选择不敏感于 calibration dataset 选择（cross-dataset robust）

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
