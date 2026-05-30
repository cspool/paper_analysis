## Expert Activation Frequency in MoE（MoE 中的专家激活频率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Activation Frequency（专家激活频率）是衡量 MoE 推理过程中每个 expert 被 gating network 选择（激活）次数的指标。在推理时，每个 token 经过 router 后选择 top-k 个 expert，统计所有 token 在整个推理过程中对每个 expert 的选择次数，即得到每 expert 的 activation frequency。这个指标反映了：(a) 哪些 expert 被频繁使用（热门 expert），哪些几乎不被使用（冷门 expert）；(b) expert 负载分布的均匀程度；(c) 训练时负载均衡损失（auxiliary load balancing loss）的实际效果。MoE-Inference-Bench (Section 8.3) 通过 activation frequency heatmap 可视化每层每个 expert 的激活次数，对比了 DeepSeek-VL2 系列（经过精心训练的负载均衡）和 MolmoE-1B 模型的 activation pattern。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Activation Frequency 的计算过程：

```
# Activation frequency counting during MoE inference
# Method 1: Inference-time counting (used in MoE-Inference-Bench)
activation_counts = zeros(num_layers, num_experts_per_layer)  # [L, E]

for each batch in evaluation_dataset:
    hidden_states = model.forward_first_layers(batch)  # up to first MoE layer

    for layer in moe_layers:
        router_logits = layer.router(hidden_states)     # [B, S, E]
        _, topk_indices = topk(softmax(router_logits), k)

        for e in range(num_experts):
            activation_counts[layer][e] += count(topk_indices == e)

        hidden_states = layer.forward(hidden_states, topk_indices)

# Normalize to activation frequency (per 1000 tokens or absolute counts)
activation_freq = activation_counts / total_tokens * 1000
```

MoE-Inference-Bench 的关键发现（Section 8.3, Figure 15）：(a) DeepSeek-VL2 系列模型显示相对均匀的 activation pattern，各 expert 和 layer 间的激活分布接近——这是因为 DeepSeek-V2 在训练时加入了 auxiliary loss 来平衡 expert 利用率；(b) MolmoE-1B 显示更稀疏的 activation pattern，某些 expert 的激活次数远超其他（最高达 1M 次，而 DeepSeek-VL2 最高约 290K），形成明显的"热门-冷门"expert 分布；(c) **关键洞察**：Activation frequency 在 well-balanced 模型中不是评估 expert 重要性的可靠指标——因为均匀分布下所有 expert 被激活次数接近，无法通过 frequency 区分哪些 expert 对模型质量更重要。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **离线统计**：在代表性 benchmark 或 validation dataset 上运行一次完整推理，记录路由决策。MoE-Inference-Bench 使用 MME benchmark dataset 对 VLM 模型进行 activation frequency 统计。
- **在线监控**：在 production inference 中持续监控 activation frequency，用于检测：(a) 某些 expert 长期不被激活（可考虑剪枝）；(b) expert 负载漂移（可能由于 data distribution shift）。
- **可视化**：通常以 heatmap 形式呈现，x 轴为 layer index，y 轴为 expert index，颜色深度代表激活次数/频率。MoE-Inference-Bench 的 Figure 15 是典型例子。
- **与 expert importance 的关系**：在负载不均的模型中，activation frequency 低可能意味着 expert 可被安全剪枝；在 well-balanced 模型中（如 DeepSeek-VL2），activation frequency 不能单独作为重要性指标，需要结合 weight magnitude、gradient-based importance 等多维指标（参见 MoE-I² 的 ϕ·w multi-factor importance）。
- 局限：(a) 依赖于具体的 evaluation dataset（不同数据集的 token 分布不同，activation pattern 也不同）；(b) 需要额外的 hook/callback 机制来捕获每层 router 的 topk_indices 输出。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD 的理论扩展**：MoESD 从概率角度推导了激活专家数的闭式表达式 N(t) = E × (1 - ((E-K)/E)^t)，其中 E 为总 expert 数，K 为每 token 激活的 expert 数，t 为输入 token 数。推导假设各 expert 激活独立同分布（均匀路由），MoESD 实验验证与实际模型行为高度一致。进一步定义全激活阈值 T_thres = ⌈log_{(1-ρ)}(1-τ)⌉（τ 通常取 0.95，ρ=K/E 为 sparsity），当 batch size B ≥ T_thres 时几乎所有 expert 同时激活。此时每 expert 平均处理 token 数 Texp(t;ρ) = ρt/(1-(1-ρ)^t)，证明 ρ 越小（越稀疏）→ Texp 越小 → 系统更 memory-bound → SD 验证的计算增量近乎免费。该分析是 MoESD 证明"中等 batch size 下 SD 对稀疏 MoE 更有效"的理论基础。
