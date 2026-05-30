## Dual-Modality Thresholding (DMT)

术语是什么？
Dual-Modality Thresholding (DMT) 是 MoDES 提出的针对多模态 MoE 模型（MLLM）的 modality-aware expert skipping 策略。传统 expert skipping 方法对所有 token 采用统一阈值，忽略了 text token 和 vision token 在 MoE FFN 中的行为差异。DMT 为 text token 和 vision token 分别设置独立的跳过阈值 $\tau_t$ 和 $\tau_v$，允许根据不同 modality 的 expert 冗余度进行差异化跳过。

从算法pipeline角度拆解术语：
DMT 的决策过程：

```
# 离线搜索最优阈值对
B = sorted grid of D=100 candidates in (0, 1)
target_skip_ratio = rho (e.g., 83%)
前端搜索 (Frontier Search) 在 O(ND) 时间内找到最优 (tau_t*, tau_v*)
满足 g(tau_t*, tau_v*) >= rho 且最小化 f(tau_t*, tau_v*)

# === 在线推理：DMT 决策 ===
for token x at layer l:
    modality = is_text_token(x) ? "text" : "vision"
    threshold = (modality == "text") ? tau_t : tau_v

    for i in topk_indices(pi, k):
        s_i = alpha_tilde[l] * pi[i]   # GMLG score
        if s_i < threshold:
            skip Expert_i              # Eq.(5)
```

DMT 的设计依据（论文 Motivation 节 Fig. 3）：
- **(Middle)** Vision token 的 pre-FFN 与 post-FFN cosine similarity 高于 text token → FFN 对 vision token 的更新幅度更小 → vision expert 冗余度更高。
- **(Right)** Vision token 与 FFN 权重的夹角更接近 90°（更正交）→ 向量投影小 → 更新幅度受限。
- 实际运行时，$\tau_v > \tau_t$（vision 阈值更高），使得 vision expert 被更激进跳过。MoDES 在 83% 总 skipping ratio 下，vision token 的 skipping ratio 显著高于 text token（Fig. 8）。

术语一般如何实现？如何使用？
- 阈值对 $(\tau_t, \tau_v)$ 由 Frontier Search 离线确定，推理时作为预加载常量。
- DMT 决策在 router kernel 内部实现：计算 importance score 后，使用 branch-free masked comparison 与 modality-specific threshold 比较，不引入额外 kernel launch。
- 仅适用于 MLLM（同时处理 text + vision token）。对于 text-only LLM，只需单一阈值（退化为 Thresholding baseline）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
