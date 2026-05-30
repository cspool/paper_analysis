## Modality-specific Distribution-aware Router (MsDaR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modality-specific Distribution-aware Router (MsDaR) 是 LTDR 的第一个核心模块。基于 vision tokens 服从 long-tailed distribution、language tokens 服从 uniform distribution 的观察，MsDaR 修改了 MoE 的 load balancing 策略：保留 language TER 的 load balancing（适配其 uniform distribution），移除 vision TER 的 load balancing（让 vision tokens 按天然 long-tailed 分布路由到专业化 expert）。具体实现是将 L_balancing 公式中的 F_i 和 G_i 计算限定为：`L_balancing = K * Σ F_i(T) * G_i(T)`，仅对 language tokens T 计算，vision tokens 完全不受负载均衡约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MsDaR: Modality-specific Distribution-aware Router
def moe_layer_forward(V, T, W_g, experts):
    # V: [M, D] vision tokens, T: [N, D] language tokens
    x = concat([V, T], dim=0)  # [(M+N), D]

    # Shared router forward
    logits = x @ W_g            # [(M+N), K]
    probs = Softmax(logits)     # [(M+N), K]

    # MsDaR: Language-only load balancing
    probs_t = probs[M:, :]      # [N, K], language portion
    F_t = argmax(probs_t, dim=1)  # which tokens go where
    for i in 1..K:
        F_i = sum(F_t == i) / N    # fraction of language tokens to expert i
        G_i = mean(probs_t[:, i])  # mean routing prob to expert i
    L_balancing = K * Σ F_i * G_i  # ONLY language tokens

    # Vision tokens: NO load balancing constraint
    # → naturally follow long-tailed distribution
    # → tail tokens get higher RPV → route to specialized experts

    return moe_output, L_balancing
```

**与 Modality-aware MoE 的区别**：
- Modality-aware MoE（MoMa, Eve）：将 experts 硬性划分为 vision group 和 language group → 损失模型容量
- MsDaR：保持所有 experts 共享，仅通过 routing 策略实现模态差异 → 更灵活、不损失 expert pool 容量
- 实验验证：MoE-LLaVA-v2Top1-t2Top1 (modality-split experts) 性能 57.7 vs baseline 57.6（几乎无提升）；添加 MsDaR 后提升至 58.2

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现复杂度**：极低。只需在 L_balancing 计算前添加 modality mask（根据 token 在 concatenation 中的位置区分 V 和 T）
- **兼容性**：与任何基于 load balancing 的 MoE 训练框架兼容（HuggingFace, Megatron, DeepSpeed-MoE）
- **训练 overhead**：计算 F_i 和 G_i 时少处理 vision tokens → 略微减少计算量
- **独立贡献**：MsDaR alone 提供 0.6%（StableLM-1.6B）和 0.5%（Phi2-2.7B）的平均提升

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
