## Expert-Choice Gating (EC Routing)

术语解释
Expert-Choice Gating 是 Zhou et al. (NeurIPS 2022) 提出的门控策略，反转传统 Token-Choice Gating 逻辑：由每个专家独立选择 top-k tokens 来处理，而非每个 token 选择 top-k 专家。

术语是什么？
传统 MoE 路由（Token-Choice）令每个 token 从 N 个专家中选 top-k 个，导致负载不均和 token 丢弃。Expert-Choice 则令每个专家从 T 个 tokens 中选 top-k 个。具体：
- 计算 token-expert affinity 矩阵 S ∈ R^{T×N}
- Token-Choice: TopK(S, k) 沿 expert 维度（每行选 k 列）
- Expert-Choice: TopK(S^T, k) 沿 token 维度（每行/每专家选 k 个 tokens）
- 每个专家恰好处理 k 个 tokens，天然负载均衡，无需 auxiliary loss

Expert-Choice 允许变长专家激活：简单 token 可能只被 0-1 个专家选中，复杂 token 可能被 3-4 个专家选中。实验显示 74% tokens 路由到 1-2 个专家，23% 到 3-4 个，3% 到 >4 个。

从算法pipeline角度拆解术语。
```
# Expert-Choice Gating
affinities = x @ W_gate.T                    # [T, N]
topk_vals, topk_idx = TopK(affinities.T, k)  # [N, k], 沿 token 维度

# 每个专家 i 处理 topk_idx[i, :] 指定的 tokens
for i in range(N):
    selected_tokens = topk_idx[i, :]
    y[selected_tokens] += FFN_i(x[selected_tokens]) * gate_weights[selected_tokens, i]
```
与 Token-Choice 的对比：
- Token-Choice: 需要 capacity factor（1.25x~2x）和 auxiliary load balancing loss（w=0.01）
- Expert-Choice: 天然负载均衡，无需 auxiliary loss，训练 2x 更快收敛

术语一般如何实现？如何使用？
- 训练时 2x 更快的收敛（达到相同 perplexity），~20% 更快的 step time
- GLUE + SuperGLUE 11 任务上平均比 Switch top-1 和 GShard top-2 高 ~2%
- 被 Brainformers 等多种后续架构采用，可从 16 扩展到 128 专家

**Brainformers 中的 Expert Choice Gating 使用**：
Brainformers 的演化搜索将 gating function 作为搜索维度之一（搜索空间包含 Top-2 和 Expert Choice），搜索到的 Brainformer Block 1 选择 Expert Choice gating + capacity factor=1。相比于 Top-2 routing，Expert Choice 在 Brainformers 中的优势：
- **Perfect load balance**：无需 auxiliary load balancing loss，训练更稳定
- **更稀疏的专家激活**：capacity factor=1 时每 token 平均路由至 1 个 expert（vs Top-2 的固定 2 个）
- **更快的 step time**：通信量减半（top-1 dispatch vs top-2 dispatch），配合更小的 expansion ratio 实现 5x step time speedup at 8B scale
- 但需要训练时可访问全部 token（双向 attention 或 encoder 场景），不适合 decoder-only 自回归推理的 naive 实现

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Brainformers Trading Simplicity for Efficiency
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

**ETR 论文中的 Expert-Choice Gating 使用**：
ETR 将 ECR 与 TCR 组合为双向路由。在 ETR 中，ECR 不是独立使用的，而是作为 TCR 之后的第二阶段：TCR 先让 token 选 top-ℓ experts，然后 ECR 让每个 expert 从已分配的 token 中按 affinity score δ 选择 top-C tokens（Bottom-C 保留最高分 token）。ETR 的 ECR 与传统 Expert-Choice 的一个关键区别：传统 EC 每个 expert 固定选 k 个 token；ETR 的 EC 使用自适应容量 C = max(C_min, s/n)，C 随训练进度动态调整（后期降低 ~40%）。ETR 理论证明 (Theorem 5)：在 expert 获得判别能力后 (q_i << 1)，ECR 成功率 ≥ 1-e^{-3C/16}，接近 1，而 TCR 仍受限于 C/s。

---
