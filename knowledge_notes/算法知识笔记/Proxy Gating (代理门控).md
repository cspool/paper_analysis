## Proxy Gating (代理门控)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Proxy Gating 是 MoE-Prism 提出的一种无需训练的门控机制重建策略。在将 monolithic expert 分解为 N 个子 expert 后，原始 router（为原 expert 设计）失效。朴素的解决方案是对每个 token 执行所有子 expert 以计算其输出范数再做选择，但这完全抵消了细粒度分解的性能收益。Proxy Gating 的解决方案：从每个子 expert 中选择 r 个 gate neuron（通过 co-activation matrix 和 centrality 排序选出），推理时仅计算 gate neurons 的激活，用这些 neuron 的平均 L1 范数作为整个子 expert 输出贡献的廉价代理估计。由于 gate neuron 是其子 expert 的"功能中心"（与子 expert 内其他 neuron 共激活频率最高），其激活模式能有效代表整个子 expert 的行为。

从算法pipeline角度拆解术语：
```
# Inference with Proxy Gating
h = input_hidden_state  # [d_model]
# 仅计算gate neurons的中间激活 (极低成本)
for each sub_expert S_n:
    gate_h = h @ W_gate[:, gate_neurons[S_n]]  # [r]
    gate_up = h @ W_up[:, gate_neurons[S_n]]    # [r]
    proxy_score[S_n] = mean(|SiLU(gate_h) * gate_up|)  # avg L1 norm

# Router选择
top_k_sub_experts = top_k(proxy_score, k)
# 仅执行选中的sub-experts (完整前向)
output = sum(softmax(proxy_score[n]) * execute_sub_expert(S_n, h) 
             for n in top_k_sub_experts)
```
对比 baseline：原始 MoE router 对 N 个 expert 输出 N 维 logits→top-k 选择。Proxy Gating 的等效操作：对每个子 expert 仅计算 r 个 gate neuron 的 L1 范数（O(r·d_model)），相比执行完整子 expert（O(C·d_model)），gate neuron 的开销可忽略不计（r ≪ C）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 中与 training-free Linear Gate (LG w/o FT) 结合使用。Proxy gating 的 proxy score 输入到新的 linear router（Linear(d_model, N_sub_experts)），router 按 softmax 归一化后选 top-k。
- 相关技术：Confidence-Guided Gate (2025) 用 token-level confidence 替代 softmax routing 解决 expert collapse；ASMG (2025) 用 Generalized Hebbian Algorithm 学习 adaptive routing subspace。
- Proxy Gating 是 training-free 的，适合快速部署。若追求最大保真度，可搭配 Low-cost Router Finetuning（仅微调 router，冻结 99.9%+ 参数）。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
