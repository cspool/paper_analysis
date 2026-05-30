## Capacity-Aware Expanded Drop

术语解释
在 Token Drop 施加 expert 容量约束前，将 token 的候选 expert 集从 top-k 扩展为 top-k+m（m 为本地设备 expert 数），使溢出 token 能被有剩余容量的低负载 expert 吸收处理，在容量约束内同时提升负载均衡和模型表示能力。

术语是什么？
Expanded Drop 利用低负载 expert 的剩余容量：Token Drop 仅丢弃超载 expert 的溢出 token → 低负载 expert 容量未被利用 → Expanded Drop 先扩展候选集再施加容量约束 → 溢出 token 被重分配到有容量的 expert 而非丢弃。扩展仅限本地设备 expert（无跨设备 All-to-All 通信开销），利用 gating score 分布的长尾平坦特性（Figure 8：top-k 外 expert 的 score 与 top-k 内末尾 expert 接近）。不强制限制每 token 最多 k 个 expert（w/o max 优于 w/ max, Table 11）。

从算法pipeline角度拆解术语：
```python
def expanded_drop(x, k, gamma, local_ids):
    scores = softmax(gate(x))                       # [N, E]
    topk_scores, topk_idx = scores.topk(k, dim=1)   # [N, k]
    
    # 扩展候选: top-k + 本地所有 expert
    local_idx = local_ids.repeat(N, 1)               # [N, m]
    exp_idx = cat([topk_idx, local_idx], dim=1)      # [N, k+m]
    local_scores = scores[:, local_ids]              # [N, m]
    exp_scores = cat([topk_scores, local_scores], dim=1)
    
    exp_mask = scatter(zeros(N,E), 1, exp_idx, 1)   # [N, E]
    masked_scores = scores * exp_mask
    
    # 逐 expert 容量约束
    cap = int(gamma * (N * k) / E)
    _, keep_idx = masked_scores.topk(cap, dim=0)     # per-expert top-cap
    cap_mask = scatter(zeros(N,E), 0, keep_idx, 1)
    
    final_map = exp_mask * cap_mask
    return scores * final_map, final_map
```

术语一般如何实现？如何使用？
在 Megatron-LM MoE forward 中，Gate 之后、All-to-All dispatch 之前插入。本地 expert ID 从 EP group 获取。与 Token Drop 相比，Expanded Drop 在 Mixtral 提升 Avg 0.7 点（74.5 vs 73.8, Table 2）；多模态场景下 Image First 策略 + Expanded Drop 在 γ=0.5 时性能接近 baseline。

涉及论文标题：
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

---
